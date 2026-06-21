"""
evaluate.py
-----------
CLI Evaluation Script for DrishtiIR — ISRO PS-10 Benchmark.

Computes all three official PS-10 metrics on a test set:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - FID   (Fréchet Inception Distance)
  - Inference Time

Also runs:
  - Semantic land cover classification (water/vegetation/urban)
  - Co-registration verification (NCC)
  - Downstream object detection comparison (IR vs Colorized)

Usage:
    # Evaluate Pix2Pix GAN:
    python src/evaluate.py --ir-dir data/test/ir --rgb-dir data/test/rgb
    
    # Evaluate Diffusion Model:
    python src/evaluate.py --ir-dir data/test/ir --rgb-dir data/test/rgb --use-diffusion
    
    # Full verbose report:
    python src/evaluate.py --ir-dir data/test/ir --rgb-dir data/test/rgb --verbose
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference import load_generator, preprocess_array, run_inference
from src.metrics import compute_all_metrics, verify_coregistration
from src.semantic_mask import classify_landcover, get_semantic_color_map

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_image_pair(ir_path: str, rgb_path: str, tile_size: int = 256):
    """Load an IR/RGB pair from disk. Supports .tif and .png/.jpg."""
    ext = os.path.splitext(ir_path)[1].lower()

    if ext in (".tif", ".tiff") and RASTERIO_AVAILABLE:
        import rasterio
        with rasterio.open(ir_path) as src:
            ir_arr = src.read().astype(np.float32)
        with rasterio.open(rgb_path) as src:
            rgb_arr = src.read().astype(np.float32)
        # Convert RGB to uint8 for display/metrics
        if rgb_arr.max() > 255:
            rgb_arr = np.clip(rgb_arr, 7000, 45000)
            rgb_arr = ((rgb_arr - 7000) / (45000 - 7000) * 255).astype(np.uint8)
        else:
            rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)
        rgb_hw3 = rgb_arr.transpose(1, 2, 0)
    else:
        ir_raw  = cv2.imread(ir_path,  cv2.IMREAD_GRAYSCALE)
        rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if ir_raw is None or rgb_raw is None:
            raise FileNotFoundError(f"Could not read: {ir_path} or {rgb_path}")
        ir_arr  = ir_raw[np.newaxis, :, :]
        rgb_hw3 = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)

    # Resize to tile_size
    h, w = rgb_hw3.shape[:2]
    if h != tile_size or w != tile_size:
        rgb_hw3 = cv2.resize(rgb_hw3, (tile_size, tile_size))

    ir_preprocessed = preprocess_array(ir_arr, tile_size)
    return ir_preprocessed, rgb_hw3


def run_diffusion_inference(ckpt_path: str, ir_pre: np.ndarray, device: str, ddim_steps: int = 50):
    """Run inference using the saved Diffusion EMA model."""
    from src.model_diffusion import ConditionalDiffusionModel, get_ddim_scheduler
    from src.dataset import denormalize

    in_channels = ir_pre.shape[0]
    model = ConditionalDiffusionModel(ir_channels=in_channels, rgb_channels=3, image_size=256)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    ddim = get_ddim_scheduler(num_inference_steps=ddim_steps)
    ir_t = torch.tensor(ir_pre, dtype=torch.float32).unsqueeze(0).to(device)
    uncond = torch.zeros_like(ir_t)

    t0 = time.perf_counter()
    noisy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        for t in ddim.timesteps:
            t_b = torch.tensor([t], device=device, dtype=torch.long)
            cond_pred   = model(noisy, ir_t, t_b, is_unconditional=False)
            uncond_pred = model(noisy, uncond, t_b, is_unconditional=True)
            noise_pred  = uncond_pred + 3.0 * (cond_pred - uncond_pred)
            noisy = ddim.step(noise_pred, t, noisy).prev_sample
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ir_disp  = denormalize(ir_pre[0])
    rgb_disp = denormalize(noisy[0].permute(1, 2, 0).cpu().numpy())
    return ir_disp, rgb_disp, elapsed_ms


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print(f"  DrishtiIR Evaluation Suite — ISRO PS-10 Benchmark")
    print(f"  Device: {device.upper()} | Model: {'Diffusion (EMA)' if args.use_diffusion else 'Pix2Pix GAN'}")
    print(f"{'='*65}\n")

    # Gather file pairs
    ir_files  = sorted([f for f in os.listdir(args.ir_dir)  if f.endswith((".tif", ".png", ".jpg"))])
    rgb_files = sorted([f for f in os.listdir(args.rgb_dir) if f.endswith((".tif", ".png", ".jpg"))])

    if not ir_files:
        print(f"[ERROR] No images found in {args.ir_dir}")
        return

    # Load GAN model if needed
    if not args.use_diffusion:
        gen = load_generator(args.checkpoint, device)

    fake_rgbs, real_rgbs = [], []
    times_ms = []
    sem_stats = {"water": [], "vegetation": [], "urban": []}
    coreg_scores = []
    detection_results = []

    for i, (ir_f, rgb_f) in enumerate(zip(ir_files, rgb_files)):
        ir_path  = os.path.join(args.ir_dir,  ir_f)
        rgb_path = os.path.join(args.rgb_dir, rgb_f)

        try:
            ir_pre, real_rgb = load_image_pair(ir_path, rgb_path, args.tile_size)
        except Exception as e:
            print(f"  [SKIP] {ir_f}: {e}")
            continue

        # Run inference
        if args.use_diffusion:
            ckpt = os.path.join(args.checkpoint_dir, "diffusion_ema_latest.pth")
            ir_disp, fake_rgb, elapsed = run_diffusion_inference(ckpt, ir_pre, device, args.ddim_steps)
        else:
            ir_disp, fake_rgb, elapsed = run_inference(gen, ir_pre, device)

        # Apply semantic correction
        from src.semantic_mask import apply_semantic_correction
        land_mask = classify_landcover(ir_pre)
        fake_rgb_corrected = apply_semantic_correction(fake_rgb, land_mask, strength=0.25)

        fake_rgbs.append(fake_rgb_corrected)
        real_rgbs.append(real_rgb)
        times_ms.append(elapsed)

        # Semantic statistics
        total = land_mask.size
        sem_stats["water"].append(100 * (land_mask == 1).sum() / total)
        sem_stats["vegetation"].append(100 * (land_mask == 2).sum() / total)
        sem_stats["urban"].append(100 * (land_mask == 3).sum() / total)

        # Co-registration
        real_gray = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
        coreg = verify_coregistration(ir_disp, real_gray)
        coreg_scores.append(coreg["ncc"])

        # Detection benchmark
        if args.run_detection:
            from src.detection import compare_detection
            det = compare_detection(ir_disp, fake_rgb_corrected, device)
            detection_results.append(det)

        print(f"  [{i+1:03d}/{len(ir_files)}] {ir_f} | {elapsed:.0f} ms | Water={sem_stats['water'][-1]:.1f}% | NCC={coreg['ncc']:.3f}")

    # ── Aggregate Metrics ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Computing FID (loading InceptionV3)...")
    metrics = compute_all_metrics(real_rgbs, fake_rgbs, device)

    print(f"\n{'='*65}")
    print(f"  FINAL EVALUATION RESULTS — {metrics['n']} image(s)")
    print(f"{'='*65}")
    print(f"  PSNR              : {metrics['PSNR']:.2f} dB       (target > 28 dB)")
    print(f"  SSIM              : {metrics['SSIM']:.4f}         (target > 0.85)")
    print(f"  FID               : {metrics['FID']:.2f}           (target < 50)")
    print(f"  Avg Inference     : {np.mean(times_ms):.1f} ms    (per 256×256 tile)")
    print(f"  Avg Co-Reg NCC    : {np.mean(coreg_scores):.4f}   (USGS pre-aligned)")
    print(f"{'─'*65}")
    print(f"  Semantic Breakdown:")
    print(f"    Water           : {np.mean(sem_stats['water']):.1f}%")
    print(f"    Vegetation      : {np.mean(sem_stats['vegetation']):.1f}%")
    print(f"    Urban/Bare Soil : {np.mean(sem_stats['urban']):.1f}%")

    if detection_results:
        avg_ir_count  = np.mean([d["ir"]["count"]  for d in detection_results])
        avg_rgb_count = np.mean([d["rgb"]["count"] for d in detection_results])
        avg_delta     = np.mean([d["delta_count"]  for d in detection_results])
        print(f"{'─'*65}")
        print(f"  Downstream Detection (Faster-RCNN, threshold=0.3):")
        print(f"    IR detections    : {avg_ir_count:.1f}")
        print(f"    RGB detections   : {avg_rgb_count:.1f}")
        print(f"    Delta            : +{avg_delta:.1f} objects (+{100*avg_delta/max(avg_ir_count,1):.0f}% improvement)")

    print(f"{'='*65}\n")

    # Save report
    if args.output_report:
        report = {
            "model":       "diffusion_ema" if args.use_diffusion else "pix2pix_gan",
            "n_images":    metrics["n"],
            "psnr":        metrics["PSNR"],
            "ssim":        metrics["SSIM"],
            "fid":         metrics["FID"],
            "avg_ms":      float(np.mean(times_ms)),
            "avg_ncc":     float(np.mean(coreg_scores)),
        }
        import json
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {args.output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiIR Evaluation Suite — ISRO PS-10")
    parser.add_argument("--ir-dir",         required=True,                                        help="Directory of IR test images")
    parser.add_argument("--rgb-dir",        required=True,                                        help="Directory of real RGB ground truth images")
    parser.add_argument("--checkpoint",     default="checkpoints/generator_latest.pth",           help="GAN checkpoint path")
    parser.add_argument("--checkpoint-dir", default="checkpoints",                                help="Diffusion checkpoint directory")
    parser.add_argument("--use-diffusion",  action="store_true",                                  help="Use Diffusion EMA model instead of GAN")
    parser.add_argument("--tile-size",      type=int,   default=256)
    parser.add_argument("--ddim-steps",     type=int,   default=50)
    parser.add_argument("--run-detection",  action="store_true",                                  help="Run Faster-RCNN downstream detection benchmark")
    parser.add_argument("--output-report",  type=str,   default="evaluation_report.json",        help="Save JSON report to this path")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()
    evaluate(args)
