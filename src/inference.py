"""
inference.py
------------
Inference utilities: load a trained Generator checkpoint and run single-image
or batch colorization with metric reporting.

Can be used as:
  - A Python module imported by app.py (Streamlit dashboard)
  - A CLI tool: python inference.py --input image.tif --output result.png
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from model import Generator
from dataset import denormalize, normalize, IR_DN_MIN, IR_DN_MAX, TILE_SIZE

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_generator(checkpoint_path: str, device: str = "cpu") -> Generator:
    """Load a trained Generator from a .pth checkpoint file."""
    if not os.path.exists(checkpoint_path):
        print(f"[Inference] WARNING: Checkpoint not found at '{checkpoint_path}'.")
        print("[Inference] Using randomly initialized weights for demo purposes.")
        gen = Generator()
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # The first conv layer weight shape is (out_channels, in_channels, k, k)
        in_channels = state_dict['enc1.block.0.weight'].shape[1]
        gen = Generator(in_channels=in_channels)
        gen.load_state_dict(state_dict)
        print(f"[Inference] Loaded checkpoint: {checkpoint_path}")
    gen.to(device)
    gen.eval()
    return gen


# ─── Image Pre-processing ─────────────────────────────────────────────────────

def preprocess_tiff(path: str, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Read a single-channel GeoTIFF (thermal/IR band) and return a
    normalized float32 array in [-1, 1] shaped (1, tile_size, tile_size).
    """
    assert RASTERIO_AVAILABLE, "rasterio required for .tif input. Run: pip install rasterio"
    with rasterio.open(path) as src:
        band = src.read(
            1,
            out_shape=(tile_size, tile_size),
            resampling=rasterio.enums.Resampling.bilinear,
        )
    return normalize(band, IR_DN_MIN, IR_DN_MAX)[np.newaxis, :, :]  # (1, H, W)


def preprocess_png(path: str, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Read a PNG/JPEG grayscale image, resize, and normalize to [-1, 1].
    For demo use when GeoTIFFs are not available.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, (tile_size, tile_size)).astype(np.float32) / 255.0
    img = img * 2.0 - 1.0   # Normalize to [-1, 1]
    return img[np.newaxis, :, :]   # (1, H, W)


def preprocess_array(arr: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Accepts a raw 2D numpy array (H, W) or 3D numpy array (C, H, W).
    """
    # If passed as (H, W), expand to (1, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
        
    c, h, w = arr.shape
    if h > tile_size and w > tile_size:
        # It's a huge raw scene. Take a center crop to preserve scale.
        cy, cx = h // 2, w // 2
        half = tile_size // 2
        arr = arr[:, cy - half : cy + half, cx - half : cx + half].astype(np.float32)
    else:
        # Resize each channel
        resized = []
        for i in range(c):
            resized.append(cv2.resize(arr[i], (tile_size, tile_size)))
        arr = np.stack(resized).astype(np.float32)
    
    # Check if this is a 16-bit Landsat thermal image
    if arr.max() > 255.0:
        if c == 1:
            arr = np.clip(arr, IR_DN_MIN, IR_DN_MAX)
            arr = (arr - IR_DN_MIN) / (IR_DN_MAX - IR_DN_MIN)
        else:
            arr = np.clip(arr, 0.0, 65535.0)
            arr = arr / 65535.0
    elif arr.max() > 1.0:
        arr = arr / 255.0
        
    arr = arr * 2.0 - 1.0
    return arr


# ─── Core Inference ───────────────────────────────────────────────────────────

def run_inference(
    gen: Generator,
    ir_array: np.ndarray,   # shape: (1, H, W) in [-1, 1]
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run a single forward pass through the Generator.

    Returns:
      ir_display  : uint8 grayscale IR image (H, W)
      rgb_output  : uint8 RGB colorized output (H, W, 3)
      elapsed_ms  : inference time in milliseconds
    """
    x = torch.tensor(ir_array, dtype=torch.float32).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        fake_rgb = gen(x)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Denormalize
    ir_display  = denormalize(ir_array[0])                                        # (H, W)  uint8
    rgb_output  = denormalize(fake_rgb[0].permute(1, 2, 0).cpu().numpy())         # (H, W, 3) uint8

    return ir_display, rgb_output, elapsed_ms


def run_inference_tta(
    gen: Generator,
    ir_array: np.ndarray,   # shape: (C, H, W) in [-1, 1]
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Test-Time Augmentation (TTA) [Technique #15]:
    Runs the model on 8 augmented copies of the input image (4 rotations × 2 flips),
    then averages all predictions together. The averaged composite is consistently
    sharper, more color-accurate, and less artifact-prone than a single forward pass.

    Returns:
      ir_display  : uint8 grayscale IR image (H, W)
      rgb_output  : uint8 TTA-averaged colorized output (H, W, 3)
      elapsed_ms  : total inference time in milliseconds
    """
    x = torch.tensor(ir_array, dtype=torch.float32).unsqueeze(0).to(device)

    t0 = time.perf_counter()

    # 8 augmentations: 4 rotations (0°, 90°, 180°, 270°) × 2 flips (normal, horizontally flipped)
    augmented_inputs = []
    for k in range(4):
        rotated = torch.rot90(x, k, dims=[2, 3])
        augmented_inputs.append(rotated)
        augmented_inputs.append(torch.flip(rotated, dims=[3]))  # Horizontal flip

    # Run all 8 forward passes and de-augment the outputs
    predictions = []
    with torch.no_grad():
        for i, aug_x in enumerate(augmented_inputs):
            pred = gen(aug_x)
            k = i // 2
            flipped = (i % 2 == 1)
            # Reverse the flip
            if flipped:
                pred = torch.flip(pred, dims=[3])
            # Reverse the rotation
            pred = torch.rot90(pred, -k, dims=[2, 3])
            predictions.append(pred)

    # Average all 8 de-augmented predictions
    avg_pred = torch.stack(predictions, dim=0).mean(dim=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Denormalize
    ir_display  = denormalize(ir_array[0])
    rgb_output  = denormalize(avg_pred[0].permute(1, 2, 0).cpu().numpy())

    return ir_display, rgb_output, elapsed_ms


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    fake_rgb: np.ndarray,     # (H, W, 3) uint8
    real_rgb: np.ndarray,     # (H, W, 3) uint8
) -> dict[str, float]:
    """
    Compute PSNR and SSIM between generated and ground-truth RGB images.
    Returns a dict with 'PSNR' and 'SSIM' keys. Returns empty dict if skimage is unavailable.
    """
    if not SKIMAGE_AVAILABLE:
        return {}

    psnr_val = psnr_fn(real_rgb, fake_rgb, data_range=255)
    ssim_val = ssim_fn(
        real_rgb, fake_rgb,
        data_range=255,
        channel_axis=2,        # Color axis for multi-channel SSIM
        win_size=7,
    )
    return {"PSNR": round(float(psnr_val), 2), "SSIM": round(float(ssim_val), 4)}


# ─── CLI Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR-to-RGB Colorization Inference")
    parser.add_argument("--input",      required=True,                   help="Path to input IR image (.tif / .png / .jpg)")
    parser.add_argument("--output",     default="colorized_output.png",  help="Path to save colorized output")
    parser.add_argument("--checkpoint", default="checkpoints/generator_latest.pth", help="Generator checkpoint path")
    parser.add_argument("--tile-size",  type=int, default=TILE_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_generator(args.checkpoint, device)

    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".tif", ".tiff"):
        ir_arr = preprocess_tiff(args.input, args.tile_size)
    else:
        ir_arr = preprocess_png(args.input, args.tile_size)

    ir_disp, rgb_out, ms = run_inference(gen, ir_arr, device)

    print(f"Inference time : {ms:.1f} ms")

    # Save side-by-side comparison panel
    ir_bgr  = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)
    rgb_bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    panel   = np.concatenate([ir_bgr, rgb_bgr], axis=1)
    cv2.imwrite(args.output, panel)
    print(f"Saved output   : {args.output}")
