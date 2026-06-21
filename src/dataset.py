"""
dataset.py
----------
Geospatial data pipeline for paired Landsat 8/9 IR and RGB tiles.

Handles:
- .npy reading (from fetch_massive_dataset.py)
- TIFF reading via rasterio
- Resize/crop to requested tile_size (128 or 256 for progressive training)
- Normalization to [-1, 1] for diffusion training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

# Rasterio is only required for real GeoTIFF workflows
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

# Standard Landsat Collection-2 Level-2 surface reflectance/temperature scale limits
# Thermal (ST_B10): ~35,000 to ~55,000 DN (approx 268K to 337K)
IR_DN_MIN = 35000
IR_DN_MAX = 55000
# RGB (SR_B2,3,4): ~7000 (0.0 reflectance) to ~45000 (1.0 reflectance)
RGB_DN_MIN = 7000
RGB_DN_MAX = 45000

TILE_SIZE = 256


# ─── Normalization Helpers ────────────────────────────────────────────────────

def normalize(band: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Map a raw DN band into [-1, 1] range for GAN training."""
    band = np.clip(band, min_val, max_val).astype(np.float32)
    return (band - min_val) / (max_val - min_val) * 2.0 - 1.0


def denormalize(tensor: np.ndarray) -> np.ndarray:
    """Convert a [-1, 1] model output back to [0, 255] uint8 for display."""
    img = (tensor + 1.0) / 2.0 * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── Resize Helper ────────────────────────────────────────────────────────────

def resize_array(arr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize a (C, H, W) array to (C, target_size, target_size).
    Uses center crop if the image is larger, or bilinear resize otherwise.
    """
    C, H, W = arr.shape
    if H == target_size and W == target_size:
        return arr

    # If image is larger than target, take a random crop during training
    if H > target_size and W > target_size:
        # Random crop for data augmentation
        top  = np.random.randint(0, H - target_size)
        left = np.random.randint(0, W - target_size)
        return arr[:, top:top + target_size, left:left + target_size]

    # Otherwise resize each channel
    resized = []
    for i in range(C):
        resized.append(cv2.resize(arr[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR))
    return np.stack(resized)


# ─── Real GeoTIFF / NPY Dataset ──────────────────────────────────────────────

class LandsatIRDataset(Dataset):
    """
    Loads paired IR and RGB tiles from .npy or .tif files.

    Directory layout expected:
        data/train_massive/ir_multiband/  → .npy files with shape (C, H, W)
        data/train_massive/rgb/           → .npy files with shape (3, H, W)
    """

    def __init__(self, ir_dir: str, rgb_dir: str, tile_size: int = TILE_SIZE):
        valid_ext = (".tif", ".jpg", ".png", ".jpeg", ".npy")
        self.ir_files = sorted(
            [os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith(valid_ext)]
        )
        self.rgb_files = sorted(
            [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(valid_ext)]
        )
        assert len(self.ir_files) == len(self.rgb_files), (
            f"IR and RGB file counts do not match: {len(self.ir_files)} vs {len(self.rgb_files)}"
        )
        assert len(self.ir_files) > 0, (
            f"No files found in {ir_dir} or {rgb_dir}. "
            "Run fetch_massive_dataset.py first."
        )
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.ir_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ir_path = self.ir_files[idx]
        rgb_path = self.rgb_files[idx]

        # ── Load arrays ──────────────────────────────────────────────────
        if ir_path.endswith(".npy"):
            ir_arr = np.load(ir_path).astype(np.float32)
        else:
            with rasterio.open(ir_path) as src:
                ir_arr = src.read().astype(np.float32)

        if rgb_path.endswith(".npy"):
            rgb_arr = np.load(rgb_path).astype(np.float32)
        else:
            with rasterio.open(rgb_path) as src:
                rgb_arr = src.read().astype(np.float32)

        # ── Use the same random crop seed for IR and RGB ─────────────────
        state = np.random.get_state()
        ir_arr  = resize_array(ir_arr, self.tile_size)
        np.random.set_state(state)
        rgb_arr = resize_array(rgb_arr, self.tile_size)

        # ── Normalize IR to [-1, 1] ─────────────────────────────────────
        if ir_arr.max() > 255.0:
            # 16-bit Landsat DN values
            for ch in range(ir_arr.shape[0]):
                ch_min, ch_max = ir_arr[ch].min(), ir_arr[ch].max()
                if ch_max > ch_min:
                    ir_arr[ch] = (ir_arr[ch] - ch_min) / (ch_max - ch_min) * 2.0 - 1.0
                else:
                    ir_arr[ch] = 0.0
        elif ir_arr.max() > 1.0:
            ir_arr = ir_arr / 255.0 * 2.0 - 1.0
        # else: already in [0,1] or [-1,1] range

        # ── Normalize RGB to [-1, 1] ────────────────────────────────────
        if rgb_arr.max() > 255.0:
            # 16-bit Landsat DN values — per-channel normalization
            for ch in range(rgb_arr.shape[0]):
                ch_min, ch_max = rgb_arr[ch].min(), rgb_arr[ch].max()
                if ch_max > ch_min:
                    rgb_arr[ch] = (rgb_arr[ch] - ch_min) / (ch_max - ch_min) * 2.0 - 1.0
                else:
                    rgb_arr[ch] = 0.0
        elif rgb_arr.max() > 1.0:
            rgb_arr = rgb_arr / 255.0 * 2.0 - 1.0

        ir_tensor  = torch.from_numpy(ir_arr)
        rgb_tensor = torch.from_numpy(rgb_arr)

        return ir_tensor, rgb_tensor


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_dataloader(
    ir_dir: str,
    rgb_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    tile_size: int = TILE_SIZE,
    val_split: float = 0.0,
) -> "DataLoader | tuple[DataLoader, DataLoader | None]":
    """
    Returns a DataLoader for real GeoTIFF/NPY data.

    tile_size  : controls resolution (128 for Phase 1, 256 for Phase 2 progressive training).
    val_split  : fraction of data to hold out for validation (0.0 = no split, returns single loader).
                 When > 0, returns (train_loader, val_loader) tuple.
    """
    from torch.utils.data import random_split

    if not ir_dir or not os.path.isdir(ir_dir):
        raise ValueError(f"IR directory not found: {ir_dir}")
    if not rgb_dir or not os.path.isdir(rgb_dir):
        raise ValueError(f"RGB directory not found: {rgb_dir}")

    full_dataset = LandsatIRDataset(ir_dir=ir_dir, rgb_dir=rgb_dir, tile_size=tile_size)

    if val_split > 0.0:
        n_val   = max(1, int(len(full_dataset) * val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
        )
        return train_loader, val_loader

    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
