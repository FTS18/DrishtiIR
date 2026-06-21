"""
train.py
--------
Full training engine for the IR-to-RGB Pix2Pix GAN.

Loss Strategy:
  Generator total loss = Adversarial Loss + L1_LAMBDA * L1 Loss + SSIM_LAMBDA * (1 - SSIM)
  Discriminator loss   = 0.5 * (Real Loss + Fake Loss)

Features:
  - Automatic fallback to SyntheticIRDataset when no data directory is provided
  - Periodic checkpoint saving
  - Training metrics logging to console
  - Sample output visualization saved to disk every N epochs
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Resolve local src imports regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from model import Generator, Discriminator
from dataset import get_dataloader, denormalize

# Optional SSIM loss
try:
    from pytorch_msssim import ssim as compute_ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


# ─── Hyperparameters ──────────────────────────────────────────────────────────

DEFAULTS = {
    "lr":           2e-4,
    "beta1":        0.5,
    "beta2":        0.999,
    "l1_lambda":    100.0,   # Weight for pixel-level L1 reconstruction fidelity
    "ssim_lambda":  20.0,    # Weight for structural similarity loss
    "num_epochs":   100,
    "batch_size":   8,
    "save_every":   10,      # Save checkpoint every N epochs
    "sample_every": 5,       # Write sample images every N epochs
    "lr_decay_epoch": 50,    # Start halving LR after this epoch
    "num_workers":  0,       # Set >0 only on Linux/Mac; Windows multiprocessing caveat
}


# ─── Loss Functions ───────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        import torchvision.models as models
        blocks = []
        # Load pre-trained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # We extract features up to relu2_2 (layer index 9)
        blocks.append(vgg[:4].eval())
        blocks.append(vgg[4:9].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # Assuming input/target are in [-1, 1], normalize to [0, 1] then subtract ImageNet mean
        input = (input + 1) / 2
        target = (target + 1) / 2
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.l1_loss(x, y)
        return loss

class GeneratorLoss(nn.Module):
    """
    Combined generator objective:
      Adversarial (fool discriminator) + L1 (pixel fidelity) + SSIM (structural) + VGG (Perceptual)
    """
    def __init__(self, l1_lambda: float = 100.0, ssim_lambda: float = 20.0, vgg_lambda: float = 10.0):
        super().__init__()
        self.l1_lambda   = l1_lambda
        self.ssim_lambda = ssim_lambda
        self.vgg_lambda  = vgg_lambda
        self.bce  = nn.BCEWithLogitsLoss()
        self.l1   = nn.L1Loss()
        self.vgg  = VGGPerceptualLoss()

    def forward(
        self,
        disc_fake_pred: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_adv = self.bce(disc_fake_pred, torch.ones_like(disc_fake_pred))
        loss_l1  = self.l1(fake_rgb, real_rgb) * self.l1_lambda
        
        loss_vgg = self.vgg(fake_rgb, real_rgb) * self.vgg_lambda

        loss_ssim = torch.tensor(0.0, device=fake_rgb.device)
        if SSIM_AVAILABLE:
            # SSIM requires [0, 1] range
            fake_01 = (fake_rgb + 1.0) / 2.0
            real_01 = (real_rgb + 1.0) / 2.0
            loss_ssim = (1.0 - compute_ssim(fake_01, real_01, data_range=1.0)) * self.ssim_lambda

        total = loss_adv + loss_l1 + loss_ssim + loss_vgg
        breakdown = {
            "G_adv":  loss_adv.item(),
            "G_L1":   loss_l1.item(),
            "G_SSIM": loss_ssim.item(),
            "G_VGG":  loss_vgg.item(),
            "G_total": total.item(),
        }
        return total, breakdown


class DiscriminatorLoss(nn.Module):
    """Standard LSGAN-flavoured discriminator loss (BCEWithLogits)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        loss_real = self.bce(real_pred, torch.ones_like(real_pred))
        loss_fake = self.bce(fake_pred, torch.zeros_like(fake_pred))
        total = (loss_real + loss_fake) * 0.5
        return total, total.item()


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(epoch: int, gen: nn.Module, disc: nn.Module, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(gen.state_dict(),  os.path.join(out_dir, f"generator_epoch_{epoch:03d}.pth"))
    torch.save(disc.state_dict(), os.path.join(out_dir, f"discriminator_epoch_{epoch:03d}.pth"))
    # Always keep a "latest" copy for fast resumption
    torch.save(gen.state_dict(),  os.path.join(out_dir, "generator_latest.pth"))
    torch.save(disc.state_dict(), os.path.join(out_dir, "discriminator_latest.pth"))


def load_latest_checkpoint(gen: nn.Module, disc: nn.Module, out_dir: str, device: str) -> int:
    """Resumes from the latest checkpoint if available. Returns the start epoch."""
    gen_path  = os.path.join(out_dir, "generator_latest.pth")
    disc_path = os.path.join(out_dir, "discriminator_latest.pth")
    if os.path.exists(gen_path) and os.path.exists(disc_path):
        gen.load_state_dict(torch.load(gen_path,  map_location=device))
        disc.load_state_dict(torch.load(disc_path, map_location=device))
        print(f"[RESUME] Loaded checkpoints from: {out_dir}")
        return 0   # epoch tracking is simplified; add epoch metadata file for precision
    return 0


# ─── Sample Image Writer ──────────────────────────────────────────────────────

def save_sample_images(
    gen: nn.Module,
    ir_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    epoch: int,
    out_dir: str,
    device: str,
) -> None:
    """Saves a 3-panel grid (IR | Generated RGB | Ground Truth RGB) as PNG."""
    try:
        import cv2
    except ImportError:
        return  # Skip if OpenCV not available

    os.makedirs(out_dir, exist_ok=True)
    gen.eval()
    with torch.no_grad():
        ir_sample  = ir_batch[:1].to(device)
        rgb_sample = rgb_batch[:1].to(device)
        fake_rgb   = gen(ir_sample)

    # Denormalize all images from [-1, 1] → [0, 255]
    
    # If IR has 3 channels, display just the first (Thermal)
    if ir_sample.shape[1] >= 3:
        ir_display = denormalize(ir_sample[0, 0].cpu().numpy())
    else:
        ir_display = denormalize(ir_sample[0, 0].cpu().numpy())
        
    fake_display = denormalize(fake_rgb[0].permute(1, 2, 0).cpu().numpy())
    real_display = denormalize(rgb_sample[0].permute(1, 2, 0).cpu().numpy())

    # Convert single-channel IR to 3-channel for side-by-side
    ir_bgr   = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_display, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.cvtColor(real_display, cv2.COLOR_RGB2BGR)

    panel = np.concatenate([ir_bgr, fake_bgr, real_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"), panel)
    gen.train()


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")
    print(f"SSIM loss available: {SSIM_AVAILABLE}")

    # Data
    use_synthetic = not (args.ir_dir and os.path.isdir(args.ir_dir))
    if use_synthetic:
        print("[WARN] No valid --ir-dir provided. Training on SYNTHETIC data.")

    loader = get_dataloader(
        ir_dir=args.ir_dir if not use_synthetic else None,
        rgb_dir=args.rgb_dir if not use_synthetic else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic=use_synthetic,
    )
    
    sample_ir, _ = next(iter(loader))
    in_channels = sample_ir.shape[1]
    print(f"Detected in_channels: {in_channels}")

    # Models
    gen  = Generator(in_channels=in_channels).to(device)
    disc = Discriminator(in_channels=in_channels).to(device)

    # Resume if checkpoint exists
    start_epoch = load_latest_checkpoint(gen, disc, args.checkpoint_dir, device)

    # Optimizers & Schedulers
    opt_gen  = Adam(gen.parameters(),  lr=args.lr, betas=(args.beta1, args.beta2))
    opt_disc = Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    sched_gen  = StepLR(opt_gen,  step_size=args.lr_decay_epoch, gamma=0.5)
    sched_disc = StepLR(opt_disc, step_size=args.lr_decay_epoch, gamma=0.5)

    # Losses
    gen_loss_fn  = GeneratorLoss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda)
    disc_loss_fn = DiscriminatorLoss()

    print(f"Dataset size : {len(loader.dataset)} samples")
    print(f"Batch size   : {args.batch_size}")
    print(f"Steps/epoch  : {len(loader)}")
    print(f"Total epochs : {args.num_epochs}")
    print("-" * 60)

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        t0 = time.time()
        epoch_metrics: dict[str, float] = {
            "D": 0.0, "G_adv": 0.0, "G_L1": 0.0, "G_SSIM": 0.0, "G_total": 0.0
        }

        gen.train()
        disc.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.num_epochs}", leave=False)
        for ir_batch, rgb_batch in pbar:
            ir_batch  = ir_batch.to(device)
            rgb_batch = rgb_batch.to(device)

            # ── Discriminator Step ──────────────────────────────────────────
            fake_rgb = gen(ir_batch).detach()  # Detach so grad doesn't flow to gen

            real_pred = disc(ir_batch, rgb_batch)
            fake_pred = disc(ir_batch, fake_rgb)
            loss_D, loss_D_val = disc_loss_fn(real_pred, fake_pred)

            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # ── Generator Step ──────────────────────────────────────────────
            fake_rgb     = gen(ir_batch)   # Fresh forward pass for grad computation
            fake_pred_g  = disc(ir_batch, fake_rgb)
            loss_G, breakdown = gen_loss_fn(fake_pred_g, fake_rgb, rgb_batch)

            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            # Accumulate metrics
            epoch_metrics["D"]       += loss_D_val
            epoch_metrics["G_adv"]   += breakdown["G_adv"]
            epoch_metrics["G_L1"]    += breakdown["G_L1"]
            epoch_metrics["G_SSIM"]  += breakdown["G_SSIM"]
            epoch_metrics["G_total"] += breakdown["G_total"]

            pbar.set_postfix(D=f"{loss_D_val:.3f}", G=f"{breakdown['G_total']:.3f}")

        # Average over batches
        n = len(loader)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"D={epoch_metrics['D']/n:.3f} | "
            f"G_adv={epoch_metrics['G_adv']/n:.3f} | "
            f"G_L1={epoch_metrics['G_L1']/n:.3f} | "
            f"G_SSIM={epoch_metrics['G_SSIM']/n:.3f} | "
            f"G_total={epoch_metrics['G_total']/n:.3f} | "
            f"Time={elapsed:.1f}s"
        )

        sched_gen.step()
        sched_disc.step()

        # Checkpoint and sample images
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, gen, disc, args.checkpoint_dir)
            print(f"  > Checkpoint saved (epoch {epoch})")

        if epoch % args.sample_every == 0:
            sample_ir, sample_rgb = next(iter(loader))
            save_sample_images(gen, sample_ir, sample_rgb, epoch, args.sample_dir, device)
            print(f"  > Sample saved -> {args.sample_dir}/sample_epoch_{epoch:03d}.png")

    # Final save
    save_checkpoint(args.num_epochs, gen, disc, args.checkpoint_dir)
    print("\nTraining complete. Final checkpoint saved.")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IR-to-RGB Colorization GAN")
    parser.add_argument("--ir-dir",         type=str, default=None,          help="Path to IR training GeoTIFFs")
    parser.add_argument("--rgb-dir",        type=str, default=None,          help="Path to RGB training GeoTIFFs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--sample-dir",     type=str, default="samples",     help="Where to save sample output images")
    parser.add_argument("--num-epochs",     type=int, default=DEFAULTS["num_epochs"])
    parser.add_argument("--batch-size",     type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    parser.add_argument("--beta1",          type=float, default=DEFAULTS["beta1"])
    parser.add_argument("--beta2",          type=float, default=DEFAULTS["beta2"])
    parser.add_argument("--l1-lambda",      type=float, default=DEFAULTS["l1_lambda"])
    parser.add_argument("--ssim-lambda",    type=float, default=DEFAULTS["ssim_lambda"])
    parser.add_argument("--save-every",     type=int,   default=DEFAULTS["save_every"])
    parser.add_argument("--sample-every",   type=int,   default=DEFAULTS["sample_every"])
    parser.add_argument("--lr-decay-epoch", type=int,   default=DEFAULTS["lr_decay_epoch"])
    parser.add_argument("--num-workers",    type=int,   default=DEFAULTS["num_workers"])

    args = parser.parse_args()
    train(args)
