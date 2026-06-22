"""
dataset.py
----------
Geospatial data pipeline for paired Landsat 8/9 IR and RGB tiles.

Two Dataset classes:
  - LandsatIRDataset    : one sample per file (fast, simple)
  - TiledLandsatDataset : overlapping 256×256 patches from each 512×512 file
                          92 files × 9 tiles = 828 training pairs (9x more data)

Normalization: per-channel min-max → [-1, 1] (adapts to actual data range).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

IR_DN_MIN  = 35000
IR_DN_MAX  = 55000
RGB_DN_MIN = 7000
RGB_DN_MAX = 45000
TILE_SIZE  = 256


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize(band: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Map a raw DN band into [-1, 1] range."""
    band = np.clip(band, min_val, max_val).astype(np.float32)
    return (band - min_val) / (max_val - min_val) * 2.0 - 1.0


def denormalize(tensor: np.ndarray) -> np.ndarray:
    """Convert a [-1, 1] model output back to [0, 255] uint8."""
    img = (tensor + 1.0) / 2.0 * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalization to [-1, 1]. Handles 8-bit and 16-bit."""
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        for ch in range(arr.shape[0]):
            lo, hi = arr[ch].min(), arr[ch].max()
            if hi > lo:
                arr[ch] = (arr[ch] - lo) / (hi - lo) * 2.0 - 1.0
            else:
                arr[ch] = 0.0
    # else: already in [-1, 1] or [0, 1] — leave as is
    return arr


def _load_array(path: str) -> np.ndarray:
    """Load .npy or GeoTIFF, return (C, H, W) float32."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    assert RASTERIO_AVAILABLE, "rasterio required for .tif files"
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _list_files(directory: str) -> list[str]:
    valid = (".tif", ".jpg", ".png", ".npy")
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(valid)
    )


# ─── Simple One-Sample-Per-File Dataset ──────────────────────────────────────

class LandsatIRDataset(Dataset):
    """One random 256×256 crop per file per epoch."""

    def __init__(self, ir_dir: str, rgb_dir: str, tile_size: int = TILE_SIZE):
        self.ir_files  = _list_files(ir_dir)
        self.rgb_files = _list_files(rgb_dir)
        assert len(self.ir_files) == len(self.rgb_files), (
            f"IR/RGB count mismatch: {len(self.ir_files)} vs {len(self.rgb_files)}"
        )
        assert len(self.ir_files) > 0, f"No files found in {ir_dir}"
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.ir_files)

    def __getitem__(self, idx: int):
        ir_arr  = _normalize_array(_load_array(self.ir_files[idx]))
        rgb_arr = _normalize_array(_load_array(self.rgb_files[idx]))

        C, H, W = ir_arr.shape
        T = self.tile_size

        if H >= T and W >= T:
            top  = np.random.randint(0, H - T + 1)
            left = np.random.randint(0, W - T + 1)
            ir_arr  = ir_arr[:,  top:top+T, left:left+T]
            rgb_arr = rgb_arr[:, top:top+T, left:left+T]
        else:
            # Resize if smaller than tile_size
            ir_arr  = _resize_chw(ir_arr,  T)
            rgb_arr = _resize_chw(rgb_arr, T)

        return torch.from_numpy(ir_arr), torch.from_numpy(rgb_arr)


# ─── Tiled Dataset (9x more data from overlapping patches) ────────────────────

class TiledLandsatDataset(Dataset):
    """
    Extracts ALL overlapping 256×256 patches from each 512×512 .npy file.

    With stride=128 on a 512×512 image:
      x positions: 0, 128, 256  (3 columns)
      y positions: 0, 128, 256  (3 rows)
      → 9 tiles per image
      → 92 images × 9 = 828 training pairs

    Each tile has 50% overlap with its neighbours, which acts as a strong
    form of data augmentation and exposes the model to more edge/boundary
    context than non-overlapping tiles.
    """

    def __init__(
        self,
        ir_dir: str,
        rgb_dir: str,
        tile_size: int = 256,
        stride: int = None,        # Default: tile_size // 2 (50% overlap)
        augment: bool = True,      # Random flips for extra variety
    ):
        self.ir_files  = _list_files(ir_dir)
        self.rgb_files = _list_files(rgb_dir)
        assert len(self.ir_files) == len(self.rgb_files), (
            f"IR/RGB count mismatch: {len(self.ir_files)} vs {len(self.rgb_files)}"
        )
        assert len(self.ir_files) > 0, f"No files found in {ir_dir}"

        self.tile_size = tile_size
        self.stride    = stride if stride is not None else tile_size // 2
        self.augment   = augment

        # Pre-compute all (file_idx, top, left) tile positions
        self.tiles = []
        for file_idx in range(len(self.ir_files)):
            arr = _load_array(self.ir_files[file_idx])
            _, H, W = arr.shape
            T, S = tile_size, self.stride

            if H < T or W < T:
                # File too small: just use the whole thing resized
                self.tiles.append((file_idx, -1, -1))  # -1 = resize mode
                continue

            # Enumerate all valid top-left positions with the given stride
            for top in range(0, H - T + 1, S):
                for left in range(0, W - T + 1, S):
                    self.tiles.append((file_idx, top, left))

        print(f"  [TiledDataset] {len(self.ir_files)} files → {len(self.tiles)} tiles "
              f"(tile={tile_size}px, stride={self.stride}px)")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int):
        file_idx, top, left = self.tiles[idx]
        T = self.tile_size

        ir_arr  = _normalize_array(_load_array(self.ir_files[file_idx]))
        rgb_arr = _normalize_array(_load_array(self.rgb_files[file_idx]))

        if top == -1:
            # Resize mode for small files
            ir_arr  = _resize_chw(ir_arr,  T)
            rgb_arr = _resize_chw(rgb_arr, T)
        else:
            ir_arr  = ir_arr[:,  top:top+T, left:left+T]
            rgb_arr = rgb_arr[:, top:top+T, left:left+T]

        # Augmentation: random flips (applied identically to IR and RGB)
        if self.augment:
            if np.random.rand() > 0.5:
                ir_arr  = np.flip(ir_arr,  axis=2).copy()
                rgb_arr = np.flip(rgb_arr, axis=2).copy()
            if np.random.rand() > 0.5:
                ir_arr  = np.flip(ir_arr,  axis=1).copy()
                rgb_arr = np.flip(rgb_arr, axis=1).copy()

        return torch.from_numpy(ir_arr), torch.from_numpy(rgb_arr)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resize_chw(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize each channel of (C, H, W) to (C, size, size)."""
    out = []
    for ch in range(arr.shape[0]):
        out.append(cv2.resize(arr[ch], (size, size), interpolation=cv2.INTER_LINEAR))
    return np.stack(out)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_dataloader(
    ir_dir: str,
    rgb_dir: str,
    batch_size: int = 16,
    num_workers: int = 8,
    tile_size: int = TILE_SIZE,
    val_split: float = 0.0,
    tiled: bool = True,           # Use overlapping tiles for max data
    stride: int = None,
    augment: bool = True,
    synthetic: bool = False,      # Kept for backward-compat, always ignored
    limit_data: int = None,       # Limit number of files loaded
) -> "DataLoader | tuple[DataLoader, DataLoader | None]":
    """
    Returns a DataLoader (or train/val pair) for real Landsat .npy data.

    tiled=True  (default): uses TiledLandsatDataset — 9x more samples
    tiled=False           : uses LandsatIRDataset — one random crop per file
    """
    from torch.utils.data import random_split

    if not ir_dir or not os.path.isdir(ir_dir):
        raise ValueError(f"IR directory not found: '{ir_dir}'")
    if not rgb_dir or not os.path.isdir(rgb_dir):
        raise ValueError(f"RGB directory not found: '{rgb_dir}'")

    if tiled:
        full_dataset = TiledLandsatDataset(
            ir_dir=ir_dir, rgb_dir=rgb_dir,
            tile_size=tile_size, stride=stride, augment=augment,
        )
    else:
        full_dataset = LandsatIRDataset(
            ir_dir=ir_dir, rgb_dir=rgb_dir, tile_size=tile_size,
        )

    if limit_data is not None and limit_data < len(full_dataset):
        full_dataset = torch.utils.data.Subset(full_dataset, range(limit_data))

    if val_split > 0.0:
        n_val   = max(1, int(len(full_dataset) * val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # Disable augmentation on val set
        if hasattr(val_ds.dataset, "augment"):
            val_ds.dataset.augment = False

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=True, persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False, persistent_workers=(num_workers > 0),
        )
        return train_loader, val_loader

    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
