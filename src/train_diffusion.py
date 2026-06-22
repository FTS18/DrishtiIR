"""
train_diffusion.py
------------------
Training script for IR→RGB Conditional Diffusion Model.
Optimized for small datasets (50-200 images) on a single GPU.

Active Optimizations:
  - Cosine Noise Schedule       : Better texture learning vs linear
  - Classifier-Free Guidance    : 10% unconditional dropout forces stronger conditioning
  - EMA Weights                 : Saved alongside regular weights for smooth inference
  - Spectral Augmentation       : Random band brightness scaling for sensor robustness
  - GeoAug                      : Random flips for rotation invariance
  - Gradient Accumulation       : Simulates large batch sizes within VRAM budget
  - Mixed Precision AMP         : FP16 training for 2x memory efficiency
  - DDIM Sampling               : 50-step preview generation (vs 1000-step = 20x faster)
  - Progressive Resolution      : Train at 128px first, scale to 256px for faster convergence
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model_diffusion import (
    ConditionalDiffusionModel,
    EMAModel,
    get_ddpm_scheduler,
    get_ddim_scheduler,
)
from dataset import get_dataloader, denormalize


# ─── Augmentations ────────────────────────────────────────────────────────────

def spectral_augment(ir_batch: torch.Tensor) -> torch.Tensor:
    """
    Randomly scales the brightness of each spectral band independently.
    Simulates different atmospheric conditions and sensor calibration variations.
    """
    if ir_batch.shape[1] < 2:
        return ir_batch
    B, C, H, W = ir_batch.shape
    scales = (torch.rand(B, C, 1, 1, device=ir_batch.device) * 0.4 + 0.8)
    return torch.clamp(ir_batch * scales, -1.0, 1.0)


def geospatial_augment(ir_batch: torch.Tensor, rgb_batch: torch.Tensor) -> tuple:
    """Random flips and 90-degree rotations for maximum spatial invariance."""
    if torch.rand(1).item() > 0.5:
        ir_batch  = torch.flip(ir_batch,  dims=[3])
        rgb_batch = torch.flip(rgb_batch, dims=[3])
    if torch.rand(1).item() > 0.5:
        ir_batch  = torch.flip(ir_batch,  dims=[2])
        rgb_batch = torch.flip(rgb_batch, dims=[2])
        
    # Extreme Augmentation: Random 90, 180, 270 degree rotations (0% runtime cost)
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        ir_batch = torch.rot90(ir_batch, k, [2, 3])
        rgb_batch = torch.rot90(rgb_batch, k, [2, 3])
        
    return ir_batch, rgb_batch


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(epoch: int, model: nn.Module, ema: EMAModel, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"diffusion_epoch_{epoch:03d}.pth"))
    torch.save(model.state_dict(), os.path.join(out_dir, "diffusion_latest.pth"))
    torch.save(ema.state_dict(), os.path.join(out_dir, "diffusion_ema_latest.pth"))
    print(f"  [CKPT] Saved epoch {epoch:03d} + EMA to {out_dir}")


# ─── Sampling with DDIM (50 steps instead of 1000) ───────────────────────────

def save_sample_images(
    model: nn.Module,
    ir_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    epoch: int,
    out_dir: str,
    device: str,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
):
    """
    DDIM Sampling + Classifier-Free Guidance:
    Generates a sample image and saves a side-by-side panel [IR | Generated | Ground Truth].
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    ddim = get_ddim_scheduler(num_inference_steps=num_inference_steps)

    with torch.no_grad():
        ir_sample   = ir_batch[:1].to(device)
        real_rgb    = rgb_batch[:1].to(device)
        uncond_ir   = torch.zeros_like(ir_sample)

        # Start from pure noise
        noisy_rgb   = torch.randn(1, 3, ir_sample.shape[2], ir_sample.shape[3], device=device)

        for t in tqdm(ddim.timesteps, desc=f"DDIM Sampling (Ep {epoch})", leave=False):
            t_batch = torch.tensor([t], device=device, dtype=torch.long)

            # Conditional noise prediction (with IR conditioning)
            noise_cond   = model(noisy_rgb, ir_sample, t_batch, is_unconditional=False)
            # Unconditional noise prediction (blank IR)
            noise_uncond = model(noisy_rgb, uncond_ir, t_batch, is_unconditional=True)

            # Classifier-Free Guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            noisy_rgb = ddim.step(noise_pred, t, noisy_rgb).prev_sample

        fake_rgb = noisy_rgb

    # Denormalize and save side-by-side panel
    ir_disp   = denormalize(ir_sample[0, 0].cpu().numpy())
    fake_disp = denormalize(fake_rgb[0].permute(1, 2, 0).cpu().numpy())
    real_disp = denormalize(real_rgb[0].permute(1, 2, 0).cpu().numpy())

    ir_bgr   = cv2.cvtColor(ir_disp,   cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_disp, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.cvtColor(real_disp, cv2.COLOR_RGB2BGR)

    panel = np.concatenate([ir_bgr, fake_bgr, real_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"diff_sample_epoch_{epoch:03d}.png"), panel)
    model.train()


# ─── Advanced Losses (Holy Trinity) ──────────────────────────────────────────

import torchvision.models as models

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        pred_norm = (pred_01 - self.mean) / self.std
        target_norm = (target_01 - self.mean) / self.std
        return F.mse_loss(self.vgg(pred_norm), self.vgg(target_norm))

# ─── Training Loop ────────────────────────────────────────────────────────────

from accelerate import Accelerator

def train(args):
    # ── L40S Hardware Optimizations ───────────────────────────────────────────
    # BF16: L40S has dedicated BF16 tensor cores (faster + more numerically stable than FP16)
    # TF32: NVIDIA A/L series GPUs run matmul in TF32 automatically when enabled
    import torch.backends.cudnn as cudnn
    torch.backends.cuda.matmul.allow_tf32 = True   # 8x faster matmul on L40S vs FP32
    cudnn.allow_tf32 = True                         # Faster conv via TF32
    cudnn.benchmark  = True                         # Auto-tune fastest conv algorithm

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16"   # L40S native: more stable than fp16, no loss scaling needed
    )
    device = accelerator.device
    print(f"\n{'='*60}")
    print(f"  DrishtiIR Diffusion Training  |  Device: {device}")
    print(f"  Precision: BF16 | TF32: ON | cuDNN Benchmark: ON")
    print(f"{'='*60}\n")

    # ── Progressive Resolution ────────────────────────────────────────────────
    # Phase 1: train at 128px for first 30% of epochs (fast, learns structure)
    # Phase 2: fine-tune at 256px for remaining 70% (sharpens textures)
    total_epochs = args.num_epochs
    phase1_epochs = max(10, int(total_epochs * 0.30))
    phase2_start = phase1_epochs + 1

    # Phase 1 loader at 128x128
    loader_128 = get_dataloader(
        ir_dir=args.ir_dir,
        rgb_dir=args.rgb_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tile_size=128,
        val_split=0.0,
    )
    # Phase 2 loader at 256x256 with 10% validation split
    loader_256, val_loader = get_dataloader(
        ir_dir=args.ir_dir,
        rgb_dir=args.rgb_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tile_size=256,
        val_split=0.10,
    )

    # Detect in_channels from first batch
    sample_ir, sample_rgb = next(iter(loader_256))
    in_channels = sample_ir.shape[1]
    print(f"  Input channels (IR bands): {in_channels}")
    print(f"  Dataset size: {len(loader_256.dataset)} training images")
    print(f"  Progressive Resolution: Phase 1 (128px) → Epoch {phase1_epochs} | Phase 2 (256px) → Epoch {total_epochs}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalDiffusionModel(
        ir_channels=in_channels, rgb_channels=3, image_size=256
    ).to(device)

    # ── Speed Optimization: xFormers (Flash Attention) ────────────────────────
    try:
        import xformers
        model.unet.enable_xformers_memory_efficient_attention()
        print("  [SPEED] Enabled xFormers memory efficient attention (Flash Attention)!")
    except ImportError:
        print("  [SPEED] xFormers not installed, using standard attention.")

    # ── Speed Optimization ────────────────────────────────────────────────────
    # Note: torch.compile() is currently disabled. It triggers a known PyTorch/Sympy
    # compiler bug (pow_by_natural) in this specific Python 3.12 environment.
    # The L40S is fast enough with BF16 + TF32 + batch_size=32 anyway!
    # if hasattr(torch, "compile"):
    #     print("  [SPEED] Compiling model with torch.compile() for max throughput...")
    #     model = torch.compile(model)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # EMA Model
    ema = EMAModel(model, decay=0.9999)

    # Cosine noise scheduler
    noise_scheduler = get_ddpm_scheduler()

    # ── Checkpoint Resume ─────────────────────────────────────────────────────
    start_epoch = 1
    ckpt_path = os.path.join(args.checkpoint_dir, "diffusion_latest.pth")
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"  [RESUME] Loaded checkpoint: {ckpt_path}")
            saved = sorted(
                [f for f in os.listdir(args.checkpoint_dir) if f.startswith("diffusion_epoch_")],
            )
            if saved:
                last_epoch = int(saved[-1].split("_")[-1].replace(".pth", ""))
                start_epoch = last_epoch + 1
                print(f"  [RESUME] Continuing from Epoch {start_epoch}\n")
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint ({e}), starting fresh.\n")

    # ── Optimizer: 8-bit AdamW ────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print("  [SPEED] Enabled 8-bit AdamW optimizer (Saves VRAM, trains faster)!")
    except ImportError:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
    from diffusers.optimization import get_cosine_schedule_with_warmup
    # Stable cosine scheduler (Diffusion models melt at high learning rates)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=total_epochs
    )

    # Prepare everything via Accelerator
    model, optimizer, loader_128, loader_256, lr_scheduler = accelerator.prepare(
        model, optimizer, loader_128, loader_256, lr_scheduler
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # Initialize Advanced Losses
    print("  [LOSS] Initializing Edge-Preservation and Perceptual (VGG) Losses...")
    edge_criterion = EdgeLoss().to(device)
    perc_criterion = PerceptualLoss().to(device)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0

        # Progressive resolution: switch loader at phase2_start
        loader = loader_128 if epoch < phase2_start else loader_256
        if epoch == phase2_start:
            print(f"\n  [PROG-RES] Switching to 256×256 resolution at Epoch {epoch}\n")
        optimizer.zero_grad()
        
        expected_steps = len(loader)
        if args.limit_batches is not None and args.limit_batches < expected_steps:
            expected_steps = args.limit_batches
            
        pbar = tqdm(loader, total=expected_steps, desc=f"Epoch {epoch:03d}/{total_epochs}", leave=False)
        
        steps_taken = 0

        for step, (ir_batch, rgb_batch) in enumerate(pbar):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            
            steps_taken += 1
            ir_batch  = ir_batch.to(device)
            rgb_batch = rgb_batch.to(device)

            # ── Augmentations ─────────────────────────────────────────────────
            ir_batch = spectral_augment(ir_batch)
            ir_batch, rgb_batch = geospatial_augment(ir_batch, rgb_batch)

            # ── Classifier-Free Guidance: 10% unconditional dropout ───────────
            is_uncond = torch.rand(1).item() < 0.10
            if is_uncond:
                ir_cond = torch.zeros_like(ir_batch)
            else:
                ir_cond = ir_batch

            # ── Diffusion Noise ───────────────────────────────────────────────
            noise     = torch.randn_like(rgb_batch)
            bsz       = rgb_batch.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_rgb = noise_scheduler.add_noise(rgb_batch, noise, timesteps)

            # ── Forward & Backward via Accelerate ─────────────────────────────
            with accelerator.accumulate(model):
                noise_pred = model(noisy_rgb, ir_cond, timesteps, is_unconditional=is_uncond)

                # ── Google's Min-SNR-Gamma Weighting ──────────────────────────
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                
                # Calculate Signal-to-Noise Ratio (SNR) for current timesteps
                snr = alpha_t / (1.0 - alpha_t)
                snr_gamma = 5.0
                
                # For epsilon (noise) prediction, Min-SNR weight is min(snr_gamma, snr) / snr
                min_snr_weight = torch.clamp(snr, max=snr_gamma) / snr
                
                # 1. Min-SNR Weighted MSE Noise Loss
                raw_noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
                noise_loss = (raw_noise_loss * min_snr_weight).mean()

                # 2. Explicit Color Alignment Loss (L1 on predicted x0)
                # Recover x0 estimate from predicted noise to penalize global color drift
                pred_x0 = (noisy_rgb - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                color_loss = F.l1_loss(pred_x0, rgb_batch)

                # 3. Advanced Losses (Holy Trinity)
                edge_loss = edge_criterion(pred_x0, rgb_batch)
                perc_loss = perc_criterion(pred_x0, rgb_batch)

                # Dynamically scale perc_loss weight down by 4x at 256 resolution
                current_perc_weight = 0.0002 if epoch < phase2_start else 0.00005

                # Combine: Min-SNR noise_loss for structure, color_loss for global tint
                loss = noise_loss + (0.5 * color_loss) + (0.1 * edge_loss) + (current_perc_weight * perc_loss)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            display_loss = loss.item()
            total_loss  += display_loss
            pbar.set_postfix(loss=f"{display_loss:.4f}")

        avg_loss = total_loss / max(1, steps_taken)
        lr_scheduler.step()

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0:
            save_checkpoint(epoch, accelerator.unwrap_model(model), ema, args.checkpoint_dir)
            # ── Validation Pass ──────────────────────────────────────────────
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_ir, val_rgb in val_loader:
                        val_ir  = val_ir.to(device)
                        val_rgb = val_rgb.to(device)
                        v_noise  = torch.randn_like(val_rgb)
                        v_steps  = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_rgb.shape[0],), device=device).long()
                        v_noisy  = noise_scheduler.add_noise(val_rgb, v_noise, v_steps)
                        v_pred   = model(v_noisy, val_ir, v_steps)
                        val_loss += F.mse_loss(v_pred, v_noise).item()
                val_loss /= max(len(val_loader), 1)
                print(f"  [VAL] Epoch {epoch:03d} | Val Loss: {val_loss:.4f}")
                model.train()

        if epoch % args.sample_every == 0:
            s_ir, s_rgb = next(iter(loader_256))
            save_sample_images(
                model=model,
                ir_batch=s_ir,
                rgb_batch=s_rgb,
                epoch=epoch,
                out_dir=args.sample_dir,
                device=device,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.ddim_steps,
            )

    # Save final checkpoint
    save_checkpoint(total_epochs, accelerator.unwrap_model(model), ema, args.checkpoint_dir)
    print(f"\n  Training complete! Final EMA weights saved to: {args.checkpoint_dir}/diffusion_ema_latest.pth")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiIR Diffusion Training — optimized for L40S (48GB BF16)")
    parser.add_argument("--ir-dir",         type=str,   default=None)
    parser.add_argument("--rgb-dir",        type=str,   default=None)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--sample-dir",     type=str,   default="samples_diffusion")
    parser.add_argument("--num-epochs",     type=int,   default=150)
    parser.add_argument("--limit-batches",  type=int,   default=None, help="Limit number of batches per epoch for speed")
    # L40S has 48GB VRAM — batch 32 at 256px fits easily and trains 2x faster than 16
    parser.add_argument("--batch-size",     type=int,   default=32)
    # With batch=16 we don't need grad accumulation
    parser.add_argument("--grad-accum",     type=int,   default=1,   help="Gradient accumulation steps (1 = off)")
    parser.add_argument("--lr",             type=float, default=5e-4,  help="LR scales with batch size: sqrt(16/4)*2e-4")
    parser.add_argument("--save-every",     type=int,   default=10)
    parser.add_argument("--sample-every",   type=int,   default=5)
    # L40S has 16+ CPU cores — use 8 workers for parallel data loading
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--guidance-scale", type=float, default=3.0,  help="CFG guidance scale")
    parser.add_argument("--ddim-steps",     type=int,   default=50,   help="Number of DDIM inference steps")
    args = parser.parse_args()
    train(args)
