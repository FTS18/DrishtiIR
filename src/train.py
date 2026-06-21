"""
train.py
--------
Fully optimized Pix2Pix GAN training for IR→RGB satellite colorization.
Tuned for NVIDIA L40S (48GB VRAM, BF16 tensor cores).

Loss Strategy (Generator):
  G_total = Adversarial (fool D) + L1 × 100 (pixel fidelity → PSNR)
           + SSIM × 20 (structural integrity → SSIM metric)
           + VGG × 10 (perceptual realism → FID)

L40S Optimizations:
  - BF16 AMP (native L40S tensor cores, no gradient overflow risk)
  - TF32 matmul + cuDNN benchmark (free ~30% speedup)
  - Batch size 16 (48GB VRAM can hold it comfortably)
  - 8 data workers with persistent_workers (no stall between epochs)
  - EMA on Generator (smooth, stable final weights for inference)
  - GradScaler disabled (BF16 doesn't need loss scaling)

Dataset:
  - TiledLandsatDataset: 92 scenes × 9 tiles = 828 training pairs
  - Random horizontal + vertical flips per tile
  - Spectral band augmentation on IR

Evaluation:
  - Validation split (10%) with PSNR + SSIM per epoch
  - 3-panel sample images saved: [IR | Generated | Ground Truth]
  - Semantic correction applied to saved samples (water=blue, veg=green)
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model import Generator, Discriminator
from dataset import get_dataloader, denormalize

try:
    from pytorch_msssim import ssim as compute_ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("[WARN] pytorch_msssim not found. SSIM loss disabled. Install: pip install pytorch-msssim")


# ─── EMA for Generator ────────────────────────────────────────────────────────

class EMAModel:
    """
    Exponential Moving Average of Generator weights.
    Produces smoother, more stable outputs at inference time.
    decay=0.999 → very slow update, very stable weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ─── VGG Perceptual Loss ──────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 relu1_2 and relu2_2 features.
    Penalizes texture and feature differences that L1 misses.
    Forces the generator to produce natural-looking satellite textures.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.block1 = vgg[:4].eval()   # relu1_2
        self.block2 = vgg[4:9].eval()  # relu2_2
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Normalize from [-1,1] to ImageNet stats
        fake_n = ((fake + 1) / 2 - self.mean) / self.std
        real_n = ((real + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for block in [self.block1, self.block2]:
            fake_n = block(fake_n)
            real_n = block(real_n)
            loss  += F.l1_loss(fake_n, real_n)
        return loss


# ─── Loss Functions ───────────────────────────────────────────────────────────

class GeneratorLoss(nn.Module):
    """
    Combined Generator loss:
      L_G = L_adv + λ_L1 * L_L1 + λ_ssim * (1 - SSIM) + λ_vgg * L_vgg

    Tuned weights:
      λ_L1   = 100  : dominant term, forces pixel-accurate colorization (↑ PSNR)
      λ_ssim = 20   : preserves edges and structure (↑ SSIM)
      λ_vgg  = 10   : perceptual texture realism (↓ FID)
      λ_adv  = 1    : adversarial signal (prevents blurry collapse)
    """
    def __init__(self, l1_lambda: float = 100.0, ssim_lambda: float = 20.0, vgg_lambda: float = 10.0):
        super().__init__()
        self.l1_lambda   = l1_lambda
        self.ssim_lambda = ssim_lambda
        self.vgg_lambda  = vgg_lambda
        self.bce = nn.BCEWithLogitsLoss()
        self.l1  = nn.L1Loss()
        try:
            self.vgg = VGGPerceptualLoss()
            self.use_vgg = True
        except Exception:
            self.use_vgg = False

    def forward(self, disc_fake: torch.Tensor, fake: torch.Tensor, real: torch.Tensor):
        loss_adv  = self.bce(disc_fake, torch.ones_like(disc_fake))
        loss_l1   = self.l1(fake, real) * self.l1_lambda

        loss_ssim = torch.tensor(0.0, device=fake.device)
        if SSIM_AVAILABLE:
            fake_01 = (fake + 1.0) / 2.0
            real_01 = (real + 1.0) / 2.0
            loss_ssim = (1.0 - compute_ssim(fake_01, real_01, data_range=1.0)) * self.ssim_lambda

        loss_vgg = torch.tensor(0.0, device=fake.device)
        if self.use_vgg:
            loss_vgg = self.vgg(fake, real) * self.vgg_lambda

        total = loss_adv + loss_l1 + loss_ssim + loss_vgg
        return total, {
            "G_adv":   loss_adv.item(),
            "G_L1":    loss_l1.item(),
            "G_SSIM":  loss_ssim.item(),
            "G_VGG":   loss_vgg.item(),
            "G_total": total.item(),
        }


class DiscriminatorLoss(nn.Module):
    """
    Standard PatchGAN discriminator loss with label smoothing.
    Label smoothing on real labels (1.0 → 0.9) prevents D from becoming
    overconfident, which would kill the adversarial gradient to G.
    """
    def __init__(self, label_smooth: float = 0.1):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.alpha = label_smooth  # Real labels = 1 - alpha

    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor):
        real_target = torch.ones_like(real_pred)  * (1.0 - self.alpha)
        fake_target = torch.zeros_like(fake_pred)
        loss = 0.5 * (self.bce(real_pred, real_target) + self.bce(fake_pred, fake_target))
        return loss, loss.item()


# ─── Spectral Augmentation ────────────────────────────────────────────────────

def spectral_augment(ir_batch: torch.Tensor) -> torch.Tensor:
    """Randomly scale each spectral band 0.8–1.2× to simulate sensor variation."""
    if ir_batch.shape[1] < 2:
        return ir_batch
    B, C, H, W = ir_batch.shape
    scales = torch.rand(B, C, 1, 1, device=ir_batch.device) * 0.4 + 0.8
    return torch.clamp(ir_batch * scales, -1.0, 1.0)


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(epoch: int, gen: nn.Module, disc: nn.Module, ema: EMAModel, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(gen.state_dict(),  os.path.join(out_dir, f"generator_epoch_{epoch:03d}.pth"))
    torch.save(disc.state_dict(), os.path.join(out_dir, f"discriminator_epoch_{epoch:03d}.pth"))
    torch.save(gen.state_dict(),  os.path.join(out_dir, "generator_latest.pth"))
    torch.save(disc.state_dict(), os.path.join(out_dir, "discriminator_latest.pth"))
    torch.save(ema.state_dict(),  os.path.join(out_dir, "generator_ema_latest.pth"))
    print(f"  [CKPT] Saved epoch {epoch:03d} (gen + disc + EMA) → {out_dir}/")


def load_checkpoint(gen: nn.Module, disc: nn.Module, out_dir: str, device: str) -> int:
    """Load latest checkpoint if it exists. Returns last saved epoch."""
    g_path = os.path.join(out_dir, "generator_latest.pth")
    d_path = os.path.join(out_dir, "discriminator_latest.pth")
    if os.path.exists(g_path) and os.path.exists(d_path):
        gen.load_state_dict(torch.load(g_path,  map_location=device))
        disc.load_state_dict(torch.load(d_path, map_location=device))
        # Find last epoch from numbered files
        saved = sorted(f for f in os.listdir(out_dir) if f.startswith("generator_epoch_"))
        if saved:
            last = int(saved[-1].split("_")[-1].replace(".pth", ""))
            print(f"  [RESUME] Loaded checkpoint → continuing from epoch {last + 1}")
            return last
    return 0


# ─── Sample Image Saver ───────────────────────────────────────────────────────

def save_sample_images(gen: nn.Module, ir_batch: torch.Tensor, rgb_batch: torch.Tensor,
                       epoch: int, out_dir: str, device: str, use_ema_gen=None):
    import cv2
    os.makedirs(out_dir, exist_ok=True)

    model = use_ema_gen if use_ema_gen is not None else gen
    model.eval()
    with torch.no_grad():
        ir_s  = ir_batch[:1].to(device)
        rgb_s = rgb_batch[:1].to(device)
        with autocast(dtype=torch.bfloat16):
            fake  = model(ir_s)

    ir_disp   = denormalize(ir_s[0, 0].float().cpu().numpy())
    fake_disp = denormalize(fake[0].float().permute(1,2,0).cpu().numpy())
    real_disp = denormalize(rgb_s[0].float().permute(1,2,0).cpu().numpy())

    # Apply semantic correction to generated image
    try:
        from semantic_mask import classify_landcover, apply_semantic_correction
        ir_np  = ir_s[0].float().cpu().numpy()  # (C, H, W) in [-1, 1]
        ir_01  = (ir_np + 1.0) / 2.0
        mask   = classify_landcover(ir_01)
        fake_disp = apply_semantic_correction(fake_disp, mask, strength=0.3)
    except Exception:
        pass

    ir_bgr   = cv2.cvtColor(ir_disp,   cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_disp, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.cvtColor(real_disp, cv2.COLOR_RGB2BGR)

    panel = np.concatenate([ir_bgr, fake_bgr, real_bgr], axis=1)
    path  = os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png")
    cv2.imwrite(path, panel)
    gen.train()
    if use_ema_gen is not None:
        use_ema_gen.eval()


# ─── Validation Pass ──────────────────────────────────────────────────────────

def validate(gen: nn.Module, val_loader, device: str) -> dict:
    """Compute mean PSNR and SSIM on the validation split."""
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity  as ssim_fn

    gen.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for val_ir, val_rgb in val_loader:
            val_ir  = val_ir.to(device)
            val_rgb = val_rgb.to(device)
            with autocast(dtype=torch.bfloat16):
                fake = gen(val_ir)
            for i in range(fake.shape[0]):
                f = denormalize(fake[i].float().permute(1,2,0).cpu().numpy())
                r = denormalize(val_rgb[i].float().permute(1,2,0).cpu().numpy())
                psnrs.append(psnr_fn(r, f, data_range=255))
                ssims.append(ssim_fn(r, f, data_range=255, channel_axis=2, win_size=7))
    gen.train()
    return {"PSNR": round(float(np.mean(psnrs)), 2), "SSIM": round(float(np.mean(ssims)), 4)}


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args):
    # ── L40S Hardware Setup ───────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f"  DrishtiIR Pix2Pix Training  |  Device: {device.upper()}")
    print(f"  Precision: BF16 | TF32: ON | cuDNN Benchmark: ON")
    print(f"{'='*65}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloader(
        ir_dir=args.ir_dir,
        rgb_dir=args.rgb_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tile_size=256,
        val_split=0.10,       # 10% for validation
        tiled=True,           # Overlapping tiles: 92 × 9 = 828 pairs
        stride=128,           # 50% overlap
        augment=True,
    )
    sample_ir, _ = next(iter(train_loader))
    in_channels  = sample_ir.shape[1]

    print(f"  Training samples  : {len(train_loader.dataset)}")
    print(f"  Val samples       : {len(val_loader.dataset)}")
    print(f"  IR channels       : {in_channels}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Steps/epoch       : {len(train_loader)}")
    print(f"  Total epochs      : {args.num_epochs}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    gen  = Generator(in_channels=in_channels).to(device)
    disc = Discriminator(in_channels=in_channels).to(device)

    # EMA on generator
    ema = EMAModel(gen, decay=0.999)

    # Resume from checkpoint if available
    start_epoch = load_checkpoint(gen, disc, args.checkpoint_dir, device)

    # ── Optimizers ────────────────────────────────────────────────────────────
    # G uses higher LR to keep up with D
    opt_gen  = Adam(gen.parameters(),  lr=args.lr_g,  betas=(0.5, 0.999))
    opt_disc = Adam(disc.parameters(), lr=args.lr_d,  betas=(0.5, 0.999))

    # Cosine decay: smoothly reduce LR from lr → lr/100 over training
    sched_gen  = CosineAnnealingLR(opt_gen,  T_max=args.num_epochs, eta_min=args.lr_g  / 100)
    sched_disc = CosineAnnealingLR(opt_disc, T_max=args.num_epochs, eta_min=args.lr_d / 100)

    # ── Loss Functions ────────────────────────────────────────────────────────
    gen_loss_fn  = GeneratorLoss(
        l1_lambda=args.l1_lambda,
        ssim_lambda=args.ssim_lambda,
        vgg_lambda=args.vgg_lambda,
    ).to(device)
    disc_loss_fn = DiscriminatorLoss(label_smooth=0.1)

    # BF16 GradScaler — actually not needed for BF16 (no overflow), but keep for safety
    # We'll use torch.autocast directly instead
    print(f"  SSIM loss : {'ON' if SSIM_AVAILABLE else 'OFF (install pytorch-msssim)'}")
    print(f"  VGG loss  : {'ON' if gen_loss_fn.use_vgg else 'OFF'}")
    print()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        t0 = time.time()
        metrics = {"D": 0.0, "G_adv": 0.0, "G_L1": 0.0, "G_SSIM": 0.0, "G_VGG": 0.0, "G_total": 0.0}

        gen.train()
        disc.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.num_epochs}", leave=False)
        for ir_batch, rgb_batch in pbar:
            ir_batch  = ir_batch.to(device,  non_blocking=True)
            rgb_batch = rgb_batch.to(device, non_blocking=True)

            # Spectral augmentation on IR bands
            # Disabled to prevent mode collapse on highly diverse global dataset
            # ir_batch = spectral_augment(ir_batch)

            # ── Discriminator Step ────────────────────────────────────────
            with autocast(dtype=torch.bfloat16):
                fake_rgb    = gen(ir_batch).detach()  # No grad through G
                real_pred   = disc(ir_batch, rgb_batch)
                fake_pred   = disc(ir_batch, fake_rgb)
                loss_D, d_val = disc_loss_fn(real_pred, fake_pred)

            opt_disc.zero_grad(set_to_none=True)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_disc.step()

            # ── Generator Step ────────────────────────────────────────────
            with autocast(dtype=torch.bfloat16):
                fake_rgb    = gen(ir_batch)
                fake_pred_g = disc(ir_batch, fake_rgb)
                loss_G, breakdown = gen_loss_fn(fake_pred_g, fake_rgb, rgb_batch)

            opt_gen.zero_grad(set_to_none=True)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_gen.step()

            # Update EMA
            ema.update(gen)

            # Accumulate
            metrics["D"]       += d_val
            metrics["G_adv"]   += breakdown["G_adv"]
            metrics["G_L1"]    += breakdown["G_L1"]
            metrics["G_SSIM"]  += breakdown["G_SSIM"]
            metrics["G_VGG"]   += breakdown["G_VGG"]
            metrics["G_total"] += breakdown["G_total"]
            pbar.set_postfix(D=f"{d_val:.3f}", G=f"{breakdown['G_total']:.3f}")

        n = len(train_loader)
        elapsed = time.time() - t0
        sched_gen.step()
        sched_disc.step()

        print(
            f"Epoch {epoch:03d} | "
            f"D={metrics['D']/n:.3f} | "
            f"G_adv={metrics['G_adv']/n:.3f} | "
            f"G_L1={metrics['G_L1']/n:.3f} | "
            f"G_SSIM={metrics['G_SSIM']/n:.3f} | "
            f"G_VGG={metrics['G_VGG']/n:.3f} | "
            f"G={metrics['G_total']/n:.3f} | "
            f"LR={sched_gen.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        # ── Validation ──────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, gen, disc, ema, args.checkpoint_dir)
            try:
                val_metrics = validate(ema.shadow, val_loader, device)
                print(f"  [VAL] PSNR={val_metrics['PSNR']:.2f} dB | SSIM={val_metrics['SSIM']:.4f}")
            except Exception as e:
                print(f"  [VAL] Skipped: {e}")

        # ── Sample Images ────────────────────────────────────────────────
        if epoch % args.sample_every == 0:
            s_ir, s_rgb = next(iter(train_loader))
            save_sample_images(
                gen, s_ir, s_rgb, epoch,
                args.sample_dir, device,
                use_ema_gen=ema.shadow,
            )
            print(f"  [SAMPLE] → {args.sample_dir}/sample_epoch_{epoch:03d}.png")

    # Final save
    save_checkpoint(args.num_epochs, gen, disc, ema, args.checkpoint_dir)
    print(f"\n  Training complete! EMA weights → {args.checkpoint_dir}/generator_ema_latest.pth")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiIR Pix2Pix — optimized for L40S (48GB BF16)")
    parser.add_argument("--ir-dir",         type=str,   required=True)
    parser.add_argument("--rgb-dir",        type=str,   required=True)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--sample-dir",     type=str,   default="samples")
    parser.add_argument("--num-epochs",     type=int,   default=200)
    # L40S: 48GB VRAM → batch 16 fits at 256×256 easily
    parser.add_argument("--batch-size",     type=int,   default=16)
    # Separate LRs for G and D (standard practice for stable GAN training)
    parser.add_argument("--lr-g",           type=float, default=2e-4,  help="Generator LR")
    parser.add_argument("--lr-d",           type=float, default=1e-4,  help="Discriminator LR (lower = D trains slower = more stable)")
    parser.add_argument("--l1-lambda",      type=float, default=300.0)
    parser.add_argument("--ssim-lambda",    type=float, default=20.0)
    parser.add_argument("--vgg-lambda",     type=float, default=10.0)
    parser.add_argument("--save-every",     type=int,   default=10)
    parser.add_argument("--sample-every",   type=int,   default=5)
    # L40S has 16+ CPU cores
    parser.add_argument("--num-workers",    type=int,   default=8)
    args = parser.parse_args()
    train(args)
