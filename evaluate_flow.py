"""
evaluate_flow.py
----------------
Run this on Lightning AI to generate final PSNR / SSIM / FID metrics
for the DrishtiIR Flow Matching model on the validation set.

Usage:
    python evaluate_flow.py --ir-dir data/train_massive/ir_multiband \
                            --rgb-dir data/train_massive/rgb \
                            --ckpt checkpoints/flow_ema_latest.pth \
                            --n-samples 50 \
                            --flow-steps 4

Results are saved to: results/evaluation_report_flow.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.model_diffusion import ConditionalDiffusionModel, get_flow_inference_scheduler
from src.dataset import denormalize, get_dataloader
from src.metrics import compute_psnr, compute_ssim, compute_fid


def load_flow_model(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    in_total = state.get("unet.conv_in.weight", torch.zeros(1, 4)).shape[1]
    ir_ch = max(1, in_total - 3)
    model = ConditionalDiffusionModel(ir_channels=ir_ch, rgb_channels=3, image_size=256)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  [EVAL] Flow model loaded from {ckpt_path}")
    return model


@torch.no_grad()
def run_flow(model, ir_t, scheduler, guidance_scale, device):
    uncond = torch.zeros_like(ir_t)
    # Start from pure noise
    x = torch.randn(1, 3, ir_t.shape[2], ir_t.shape[3], device=device)
    
    for t in scheduler.timesteps:
        t_val = t.item() if torch.is_tensor(t) else t
        t_b = torch.tensor([int(t_val)], device=device, dtype=torch.long)
        
        v_cond   = model(x, ir_t,   t_b)
        v_uncond = model(x, uncond, t_b)
        v_pred   = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        x = scheduler.step(v_pred, t, x).prev_sample
        
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir",      required=True)
    parser.add_argument("--rgb-dir",     required=True)
    parser.add_argument("--ckpt",        default="checkpoints/flow_ema_latest.pth")
    parser.add_argument("--n-samples",   type=int, default=50)
    parser.add_argument("--flow-steps",  type=int, default=4)
    parser.add_argument("--guidance",    type=float, default=3.0)
    parser.add_argument("--out",         default="results/evaluation_report_flow.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [EVAL] Device: {device}")

    model     = load_flow_model(args.ckpt, device)
    scheduler = get_flow_inference_scheduler(num_inference_steps=args.flow_steps)

    loader = get_dataloader(
        ir_dir=args.ir_dir, rgb_dir=args.rgb_dir,
        batch_size=1, num_workers=2,
        tile_size=256, tiled=False, augment=False,
        limit_data=args.n_samples, val_split=0.0,
    )

    psnr_list, ssim_list = [], []
    fake_list, real_list = [], []
    times = []

    for ir_t, rgb_t in tqdm(loader, total=min(args.n_samples, len(loader)), desc="Evaluating"):
        ir_t  = ir_t.to(device)
        rgb_t = rgb_t.to(device)

        # Expand 1-ch to 4-ch if needed
        if ir_t.shape[1] == 1:
            ir_t = ir_t.repeat(1, 4, 1, 1)

        t0   = time.perf_counter()
        fake = run_flow(model, ir_t, scheduler, args.guidance, device)
        times.append((time.perf_counter() - t0) * 1000.0)

        # Denormalize → uint8
        fake_np = denormalize(fake[0].permute(1, 2, 0).cpu().numpy())   # (H,W,3) uint8
        real_np = denormalize(rgb_t[0].permute(1, 2, 0).cpu().numpy())  # (H,W,3) uint8

        psnr_list.append(compute_psnr(real_np, fake_np))
        ssim_list.append(compute_ssim(real_np, fake_np))
        fake_list.append(fake_np)
        real_list.append(real_np)

        if len(times) >= args.n_samples:
            break

    # FID (needs ≥ 2 images)
    print("  [EVAL] Computing FID (this may take a moment)...")
    fid_score = compute_fid(real_list, fake_list)

    report = {
        "model":           "DrishtiIR Flow Matching EMA",
        "checkpoint":      args.ckpt,
        "n_samples":       len(times),
        "flow_steps":      args.flow_steps,
        "guidance_scale":  args.guidance,
        "device":          device,
        "metrics": {
            "PSNR_mean_dB":   round(float(np.mean(psnr_list)), 4),
            "PSNR_std_dB":    round(float(np.std(psnr_list)),  4),
            "SSIM_mean":      round(float(np.mean(ssim_list)), 4),
            "SSIM_std":       round(float(np.std(ssim_list)),  4),
            "FID":            round(float(fid_score),          4),
            "Inference_ms_per_tile_mean": round(float(np.mean(times)), 1),
            "Inference_ms_per_tile_std":  round(float(np.std(times)),  1),
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("  FINAL EVALUATION RESULTS — DrishtiIR Flow Matching Model")
    print("="*60)
    m = report["metrics"]
    print(f"  PSNR  : {m['PSNR_mean_dB']:.2f} ± {m['PSNR_std_dB']:.2f} dB")
    print(f"  SSIM  : {m['SSIM_mean']:.4f} ± {m['SSIM_std']:.4f}")
    print(f"  FID   : {m['FID']:.2f}")
    print(f"  Speed : {m['Inference_ms_per_tile_mean']:.0f} ms/tile (±{m['Inference_ms_per_tile_std']:.0f} ms)")
    print(f"\n  Report saved to: {args.out}")
    print("="*60)


if __name__ == "__main__":
    main()
