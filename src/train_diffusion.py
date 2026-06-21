"""
train_diffusion.py
------------------
Training script for the Conditional DDPM.
Optimized for training on Kaggle (P100 or dual T4 GPUs).
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model_diffusion import ConditionalDiffusionModel, get_scheduler
from dataset import get_dataloader, denormalize

def save_checkpoint(epoch, model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"diffusion_epoch_{epoch:03d}.pth"))
    torch.save(model.state_dict(), os.path.join(out_dir, "diffusion_latest.pth"))

def save_sample_images(model, noise_scheduler, ir_batch, rgb_batch, epoch, out_dir, device):
    """
    Run the reverse diffusion process to generate a sample.
    """
    import cv2
    import numpy as np
    
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        ir_sample = ir_batch[:1].to(device)
        real_rgb = rgb_batch[:1].to(device)
        
        # Start from pure noise
        noisy_rgb = torch.randn_like(real_rgb)
        
        # Reverse diffusion loop
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling", leave=False):
            timesteps = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = model(noisy_rgb, ir_sample, timesteps)
            noisy_rgb = noise_scheduler.step(noise_pred, t, noisy_rgb).prev_sample
            
        fake_rgb = noisy_rgb
        
    # Denormalize
    if ir_sample.shape[1] >= 3:
        ir_display = denormalize(ir_sample[0, 0].cpu().numpy())
    else:
        ir_display = denormalize(ir_sample[0, 0].cpu().numpy())
        
    fake_display = denormalize(fake_rgb[0].permute(1, 2, 0).cpu().numpy())
    real_display = denormalize(real_rgb[0].permute(1, 2, 0).cpu().numpy())

    ir_bgr = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_display, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.cvtColor(real_display, cv2.COLOR_RGB2BGR)

    panel = np.concatenate([ir_bgr, fake_bgr, real_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"diff_sample_epoch_{epoch:03d}.png"), panel)
    
    model.train()

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Diffusion Model on: {device.upper()}")

    use_synthetic = not (args.ir_dir and os.path.isdir(args.ir_dir))
    loader = get_dataloader(
        ir_dir=args.ir_dir if not use_synthetic else None,
        rgb_dir=args.rgb_dir if not use_synthetic else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic=use_synthetic,
    )
    
    sample_ir, sample_rgb = next(iter(loader))
    in_channels = sample_ir.shape[1]
    
    model = ConditionalDiffusionModel(ir_channels=in_channels, rgb_channels=3, image_size=256).to(device)
    noise_scheduler = get_scheduler()
    
    # Checkpoint resuming
    ckpt_path = os.path.join(args.checkpoint_dir, "diffusion_latest.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[RESUME] Loaded latest diffusion checkpoint.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.num_epochs}", leave=False)
        for ir_batch, rgb_batch in pbar:
            ir_batch = ir_batch.to(device)
            rgb_batch = rgb_batch.to(device)
            
            # Sample random noise
            noise = torch.randn_like(rgb_batch)
            bsz = rgb_batch.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # Add noise to the clean RGB images
            noisy_rgb = noise_scheduler.add_noise(rgb_batch, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model(noisy_rgb, ir_batch, timesteps)
            
            # Loss is MSE between predicted noise and actual noise
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | Diffusion MSE Loss: {avg_loss:.4f}")
        
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, model, args.checkpoint_dir)
            
        if epoch % args.sample_every == 0:
            sample_ir, sample_rgb = next(iter(loader))
            save_sample_images(model, noise_scheduler, sample_ir, sample_rgb, epoch, args.sample_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", type=str, default=None)
    parser.add_argument("--rgb-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--sample-dir", type=str, default="samples_diffusion")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    
    args = parser.parse_args()
    train(args)
