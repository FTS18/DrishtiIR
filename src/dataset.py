"""
dataset.py
----------
Geospatial data pipeline for paired Landsat 8/9 IR and RGB tiles.

Handles:
- TIFF reading via rasterio
- Bilinear upsampling of 100m thermal band to 30m grid
- Patch/tile generation with overlap
- Normalization to [-1, 1] for GAN training
- Fallback synthetic dataset for demo/testing
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


# ─── Real GeoTIFF Dataset ─────────────────────────────────────────────────────

class LandsatIRDataset(Dataset):
    """
    Loads paired IR (Landsat Band 10/11) and RGB (Bands 4,3,2) GeoTIFF tiles.

    Directory layout expected:
        data/train/ir/  → single-channel .tif files (thermal band)
        data/train/rgb/ → three-channel .tif files (R, G, B bands stacked)
    """

    def __init__(self, ir_dir: str, rgb_dir: str, tile_size: int = TILE_SIZE):
        assert RASTERIO_AVAILABLE, (
            "rasterio is required for GeoTIFF dataset loading. "
            "Install it with: pip install rasterio"
        )
        self.ir_files = sorted(
            [os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.endswith((".tif", ".jpg", ".png", ".jpeg"))]
        )
        self.rgb_files = sorted(
            [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith((".tif", ".jpg", ".png", ".jpeg"))]
        )
        assert len(self.ir_files) == len(self.rgb_files), (
            f"IR and RGB file counts do not match: {len(self.ir_files)} vs {len(self.rgb_files)}"
        )
        self.tile_size = tile_size
        self.transform = None

    def __len__(self) -> int:
        return len(self.ir_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ir_path = self.ir_files[idx]
        rgb_path = self.rgb_files[idx]

        with rasterio.open(ir_path) as src:
            ir_arr = src.read().astype(np.float32) # Shape: (C, H, W)
        with rasterio.open(rgb_path) as src:
            rgb_arr = src.read().astype(np.float32) # Shape: (3, H, W)

        # Handle IR arrays (either 1 channel or 3 channel)
        if ir_arr.shape[0] == 1:
            if ir_arr.max() <= 255.0:
                ir_arr = ir_arr / 255.0 * 2.0 - 1.0
            else:
                ir_arr = np.clip(ir_arr, IR_DN_MIN, IR_DN_MAX)
                ir_arr = (ir_arr - IR_DN_MIN) / (IR_DN_MAX - IR_DN_MIN)
                ir_arr = ir_arr * 2.0 - 1.0
        elif ir_arr.shape[0] >= 3:
            # Multi-band: B10, B6, B5
            if ir_arr.max() <= 255.0:
                ir_arr = ir_arr / 255.0 * 2.0 - 1.0
            else:
                ir_arr = np.clip(ir_arr, 0.0, 65535.0)
                ir_arr = (ir_arr / 65535.0) * 2.0 - 1.0

        # Handle RGB arrays
        if rgb_arr.max() > 255.0:
            rgb_arr = np.clip(rgb_arr, RGB_DN_MIN, RGB_DN_MAX)
            rgb_arr = (rgb_arr - RGB_DN_MIN) / (RGB_DN_MAX - RGB_DN_MIN)
        elif rgb_arr.max() > 1.0:
            rgb_arr = rgb_arr / 255.0

        rgb_arr = rgb_arr * 2.0 - 1.0

        ir_tensor = torch.from_numpy(ir_arr)
        rgb_tensor = torch.from_numpy(rgb_arr)

        if self.transform:
            stacked = torch.cat([ir_tensor, rgb_tensor], dim=0)
            stacked = self.transform(stacked)
            ir_ch = ir_arr.shape[0]
            ir_tensor, rgb_tensor = stacked[:ir_ch, ...], stacked[ir_ch:, ...]

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
    Returns a DataLoader for real GeoTIFF data.

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



