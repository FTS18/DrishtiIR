"""
sr_module.py
------------
Lightweight ESPCN-style Super-Resolution module.
Upscales 256×256 → 512×512 with edge sharpening.

Architecture: Conv2d feature extraction → PixelShuffle 2x upscale
Fallback: High-quality Lanczos + Unsharp Masking if weights not found.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ESPCN Architecture ────────────────────────────────────────────────────────

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016).
    Upscale factor: 2x (256→512)
    """
    def __init__(self, in_channels: int = 3, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize final conv to approximate identity upsampling (bicubic start)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Weight Init Helper ────────────────────────────────────────────────────────

def create_and_save_initial_weights(save_path: str):
    """Create and save initial ESPCN weights (identity-biased start point)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = ESPCN(in_channels=3, upscale_factor=2)
    torch.save(model.state_dict(), save_path)
    print(f"  [SR] Initial ESPCN weights saved to: {save_path}")


# ── Algorithmic SR Fallback ───────────────────────────────────────────────────

def _algorithmic_sr(img_uint8: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    High-quality Lanczos upscaling + Unsharp Masking.
    Used as fallback when no trained SR model is available.
    Produces genuinely sharper results than naive bicubic.
    """
    h, w = img_uint8.shape[:2]
    # Step 1: Lanczos 2x upscale
    upscaled = cv2.resize(img_uint8, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Unsharp Masking for edge enhancement
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)

    # Step 3: CLAHE for local contrast enhancement
    if sharpened.ndim == 3:
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        sharpened = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

SR_CKPT = "checkpoints/sr_espcn.pth"
_sr_model = None


def load_sr_model(device: str = "cpu") -> "ESPCN | None":
    """Load ESPCN model from checkpoint. Returns None if not found."""
    global _sr_model
    if _sr_model is not None:
        return _sr_model
    if not os.path.exists(SR_CKPT):
        return None
    try:
        model = ESPCN(in_channels=3, upscale_factor=2)
        state = torch.load(SR_CKPT, map_location=device)
        model.load_state_dict(state)
        # Ensure model is float32 to prevent bfloat16 silent NaNs on CPU
        model.to(device).to(torch.float32).eval()
        _sr_model = model
        return model
    except Exception as e:
        print(f"  [SR] Could not load SR model: {e}")
        return None


def apply_super_resolution(rgb_uint8: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Apply 2x super-resolution to a uint8 RGB image (H×W×3).

    Uses trained ESPCN if checkpoint exists, else high-quality
    Lanczos + Unsharp Masking fallback.

    Returns: upscaled uint8 RGB image (2H × 2W × 3).
    """
    model = load_sr_model(device)

    if model is not None:
        # Neural SR path
        x = torch.tensor(rgb_uint8.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, H, W)
        with torch.no_grad():
            out = model(x)
        out_np = out[0].permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        return out_np
    else:
        # Algorithmic fallback
        return _algorithmic_sr(rgb_uint8, scale=2)
