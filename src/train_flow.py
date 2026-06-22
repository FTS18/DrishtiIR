"""
train_flow.py
-------------
Flow Matching (Rectified Flow) training script for IR→RGB colorization.
Same U-Net architecture as the DDPM version — only the scheduler and loss change.

Why Flow Matching instead of DDPM?
  - DDPM: learns to denoise across 1000 noisy steps (curved path)
  - Flow Matching: learns a straight velocity field from noise→image
  - Straight path = 4-8 inference steps (vs 50 DDIM / 1000 DDPM)
  - Training: 2-3x faster (simpler loss landscape, no SNR weighting needed)
  - Quality: Equal or better than DDPM

Loss: MSE on velocity v = noise - rgb  (constant target, no SNR balancing)

Usage (Lightning AI):
    python src/train_flow.py \
        --ir-dir data/train_massive/ir_multiband \
        --rgb-dir data/train_massive/rgb \
        --batch-size 32 --num-epochs 80 --save-every 5 --limit-batches 70

Resume:
    Automatically resumes from checkpoints/flow_latest.pth if it exists.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model_diffusion import (
    ConditionalDiffusionModel,
    EMAModel,
    get_flow_train_scheduler,
    get_flow_inference_scheduler,
)
from dataset import get_dataloader, denormalize
from train_diffusion import spectral_augment, geospatial_augment, EdgeLoss, PerceptualLoss


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(epoch: int, model: nn.Module, ema: EMAModel, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"flow_epoch_{epoch:03d}.pth"))
    torch.save(model.state_dict(), os.path.join(out_dir, "flow_latest.pth"))
    torch.save(ema.state_dict(),   os.path.join(out_dir, "flow_ema_latest.pth"))
    print(f"  [CKPT] Saved flow epoch {epoch:03d} + EMA")


# ─── Sample Images ────────────────────────────────────────────────────────────

def save_sample_images(model, ir_batch, rgb_batch, epoch, out_dir, device,
                       guidance_scale=3.0, num_steps=4):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    scheduler = get_flow_inference_scheduler(num_inference_steps=num_steps)

    with torch.no_grad():
        ir_s    = ir_batch[:1].to(device)
        real    = rgb_batch[:1].to(device)
        uncond  = torch.zeros_like(ir_s)

        # Start from pure noise
        x = torch.randn(1, 3, ir_s.shape[2], ir_s.shape[3], device=device)

        for t in scheduler.timesteps:
            t_val = t.item() if torch.is_tensor(t) else t
            t_b = torch.tensor([int(t_val)], device=device, dtype=torch.long)
            v_cond   = model(x, ir_s,   t_b)
            v_uncond = model(x, uncond,  t_b)
            v_pred   = v_uncond + guidance_scale * (v_cond - v_uncond)
            x        = scheduler.step(v_pred, t, x).prev_sample

    ir_disp   = denormalize(ir_s[0, 0].cpu().numpy())
    fake_disp = denormalize(x[0].permute(1, 2, 0).cpu().numpy())
    real_disp = denormalize(real[0].permute(1, 2, 0).cpu().numpy())

    ir_bgr   = cv2.cvtColor(ir_disp,   cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_disp, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.cvtColor(real_disp, cv2.COLOR_RGB2BGR)

    panel = np.concatenate([ir_bgr, fake_bgr, real_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"flow_sample_epoch_{epoch:03d}.png"), panel)
    model.train()


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    import torch.backends.cudnn as cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32  = True
    cudnn.benchmark   = True

    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"  DrishtiIR Flow Matching Training  |  Device: {device}")
    print(f"  Precision: BF16 | 4-step inference | Straight paths")
    print(f"{'='*60}\n")

    # ── Progressive Resolution ────────────────────────────────────────────────
    total_epochs  = args.num_epochs
    phase1_epochs = max(5, int(total_epochs * 0.25))
    phase2_start  = phase1_epochs + 1

    loader_128 = get_dataloader(
        ir_dir=args.ir_dir, rgb_dir=args.rgb_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        tile_size=128, val_split=0.0,
    )
    loader_256, val_loader = get_dataloader(
        ir_dir=args.ir_dir, rgb_dir=args.rgb_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        tile_size=256, val_split=0.10,
    )

    sample_ir, sample_rgb = next(iter(loader_256))
    in_channels = sample_ir.shape[1]
    print(f"  IR bands: {in_channels}  |  Dataset: {len(loader_256.dataset)} images")
    print(f"  Phase 1 (128px): epochs 1–{phase1_epochs}")
    print(f"  Phase 2 (256px): epochs {phase2_start}–{total_epochs}\n")

    # ── Model (same U-Net as DDPM version) ────────────────────────────────────
    model = ConditionalDiffusionModel(
        ir_channels=in_channels, rgb_channels=3, image_size=256
    ).to(device)

    try:
        import xformers
        model.unet.enable_xformers_memory_efficient_attention()
        print("  [SPEED] xFormers enabled")
    except ImportError:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    ema = EMAModel(model, decay=0.9999)

    # ── Flow Matching Scheduler ───────────────────────────────────────────────
    flow_scheduler = get_flow_train_scheduler(num_train_timesteps=1000)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    ckpt_path   = os.path.join(args.checkpoint_dir, "flow_latest.pth")
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)
            saved = sorted([f for f in os.listdir(args.checkpoint_dir) if f.startswith("flow_epoch_")])
            if saved:
                last = int(saved[-1].split("_")[-1].replace(".pth", ""))
                start_epoch = last + 1
            print(f"  [RESUME] Continuing from Epoch {start_epoch}")
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint: {e}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print("  [SPEED] 8-bit AdamW enabled")
    except ImportError:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    from diffusers.optimization import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=5, num_training_steps=total_epochs
    )

    model, optimizer, loader_128, loader_256, lr_scheduler = accelerator.prepare(
        model, optimizer, loader_128, loader_256, lr_scheduler
    )
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    edge_criterion = EdgeLoss().to(device)
    perc_criterion = PerceptualLoss().to(device)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0

        loader = loader_128 if epoch < phase2_start else loader_256
        if epoch == phase2_start:
            print(f"\n  [PROG-RES] Switching to 256×256 at Epoch {epoch}\n")

        optimizer.zero_grad()

        expected = len(loader)
        if args.limit_batches and args.limit_batches < expected:
            expected = args.limit_batches

        pbar = tqdm(loader, total=expected, desc=f"Epoch {epoch:03d}/{total_epochs}", leave=False)
        steps = 0

        for step, (ir_batch, rgb_batch) in enumerate(pbar):
            if args.limit_batches and step >= args.limit_batches:
                break
            steps += 1

            ir_batch  = ir_batch.to(device)
            rgb_batch = rgb_batch.to(device)
            B         = rgb_batch.shape[0]

            # Augmentations
            ir_batch = spectral_augment(ir_batch)
            ir_batch, rgb_batch = geospatial_augment(ir_batch, rgb_batch)

            # CFG dropout (10% unconditional)
            if torch.rand(1).item() < 0.10:
                ir_cond = torch.zeros_like(ir_batch)
            else:
                ir_cond = ir_batch

            # ── Flow Matching Forward Process ─────────────────────────────────
            # Linear interpolation: x_t = (1-t)*x0 + t*noise
            # Target velocity: v = noise - x0  (constant — no SNR weighting!)
            noise = torch.randn_like(rgb_batch)

            # Sample continuous t in [0, 1], then scale to scheduler timesteps
            t_cont = torch.rand(B, device=device)                           # (B,) in [0,1]
            t_int  = (t_cont * flow_scheduler.config.num_train_timesteps    # scale to [0,1000)
                      ).long().clamp(0, flow_scheduler.config.num_train_timesteps - 1)

            # Linear interpolation (the "rectified" straight path)
            t_view = t_cont.view(-1, 1, 1, 1)
            x_t    = (1.0 - t_view) * rgb_batch + t_view * noise

            # Target: the velocity pointing from x0 to noise
            v_target = noise - rgb_batch

            with accelerator.accumulate(model):
                # Predict velocity field
                v_pred = model(x_t, ir_cond, t_int)

                # ── Loss: MSE on velocity + edge + perceptual on x0 estimate ──
                # Flow loss (main)
                flow_loss = F.mse_loss(v_pred, v_target)

                # Recover x0 estimate: x0_hat = x_t - t * v_pred
                x0_hat = x_t - t_view * v_pred
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

                edge_loss = edge_criterion(x0_hat, rgb_batch)

                perc_w = 0.0002 if epoch < phase2_start else 0.00005
                perc_loss = perc_criterion(x0_hat, rgb_batch)

                loss = flow_loss + 0.1 * edge_loss + perc_w * perc_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / max(1, steps)
        lr_scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0:
            save_checkpoint(epoch, accelerator.unwrap_model(model), ema, args.checkpoint_dir)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_ir, val_rgb in val_loader:
                        val_ir, val_rgb = val_ir.to(device), val_rgb.to(device)
                        B2 = val_rgb.shape[0]
                        t2 = torch.rand(B2, device=device)
                        t_int2 = (t2 * 1000).long().clamp(0, 999)
                        t2v = t2.view(-1, 1, 1, 1)
                        n2  = torch.randn_like(val_rgb)
                        xt2 = (1 - t2v) * val_rgb + t2v * n2
                        vp2 = model(xt2, val_ir, t_int2)
                        val_loss += F.mse_loss(vp2, n2 - val_rgb).item()
                val_loss /= max(len(val_loader), 1)
                print(f"  [VAL] Epoch {epoch:03d} | Val Loss: {val_loss:.4f}")
                model.train()

        if epoch % args.sample_every == 0:
            s_ir, s_rgb = next(iter(loader_256))
            save_sample_images(
                accelerator.unwrap_model(model), s_ir, s_rgb,
                epoch, args.sample_dir, device,
                guidance_scale=args.guidance_scale,
                num_steps=args.flow_steps,
            )

    save_checkpoint(total_epochs, accelerator.unwrap_model(model), ema, args.checkpoint_dir)
    print(f"\n  Flow Matching training complete!")
    print(f"  EMA weights: {args.checkpoint_dir}/flow_ema_latest.pth")
    print(f"  Inference: only {args.flow_steps} steps needed!\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiIR Flow Matching Training")
    parser.add_argument("--ir-dir",         type=str,   required=True)
    parser.add_argument("--rgb-dir",        type=str,   required=True)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--sample-dir",     type=str,   default="samples_flow")
    parser.add_argument("--num-epochs",     type=int,   default=80)
    parser.add_argument("--limit-batches",  type=int,   default=None)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--grad-accum",     type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--save-every",     type=int,   default=5)
    parser.add_argument("--sample-every",   type=int,   default=5)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-steps",     type=int,   default=4,
                        help="Inference steps at sample time (4 is usually enough!)")
    args = parser.parse_args()
    train(args)
