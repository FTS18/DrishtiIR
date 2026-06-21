"""
metrics.py
----------
Comprehensive evaluation metrics for IR→RGB colorization quality assessment.

Implements all three metrics from the ISRO PS-10 evaluation criteria:
  - PSNR  : Peak Signal-to-Noise Ratio (pixel reconstruction quality)
  - SSIM  : Structural Similarity Index (structural integrity)
  - FID   : Fréchet Inception Distance (perceptual realism of generated images)

Also provides:
  - Co-registration verification via normalized cross-correlation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from scipy import linalg

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ─── InceptionV3 Feature Extractor (for FID) ─────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """
    Extracts 2048-dim feature vectors from images using a pretrained InceptionV3.
    Standard approach for computing FID scores.
    """
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        # Keep everything up to the penultimate layer (before classification head)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).squeeze(-1).squeeze(-1)  # (B, 2048)


_inception_model = None
_inception_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_inception(device: str = "cpu") -> InceptionFeatureExtractor:
    global _inception_model
    if _inception_model is None:
        _inception_model = InceptionFeatureExtractor().to(device)
    return _inception_model


def _extract_features(images_uint8: list[np.ndarray], device: str = "cpu") -> np.ndarray:
    """
    Given a list of (H, W, 3) uint8 RGB images, extract InceptionV3 features.
    Returns array of shape (N, 2048).
    """
    model = _get_inception(device)
    feats = []
    for img in images_uint8:
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        t = _inception_transform(t).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(t)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)  # (N, 2048)


def compute_fid(
    real_images: list[np.ndarray],
    fake_images: list[np.ndarray],
    device: str = "cpu",
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between real and generated images.

    FID measures the distance between the feature distributions of real and fake images
    using the multivariate Gaussian assumption on InceptionV3 features.
    
    Lower FID = more photorealistic output.
    - FID < 10  : Photorealistic (indistinguishable from real)
    - FID < 50  : Good quality
    - FID < 150 : Acceptable
    - FID > 200 : Clearly artificial

    Args:
        real_images : list of (H, W, 3) uint8 arrays (real RGB ground truth)
        fake_images : list of (H, W, 3) uint8 arrays (generated RGB output)
        device      : "cuda" or "cpu"

    Returns:
        fid_score : float
    """
    if len(real_images) < 2 or len(fake_images) < 2:
        return float("nan")

    real_feats = _extract_features(real_images, device)
    fake_feats = _extract_features(fake_images, device)

    # Compute mean and covariance of each distribution
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    # FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2 * sqrt(sigma_r @ sigma_f))
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return round(fid, 2)


# ─── PSNR & SSIM ─────────────────────────────────────────────────────────────

def compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
    """PSNR between two (H, W, 3) uint8 images. Higher is better."""
    if not SKIMAGE_AVAILABLE:
        return float("nan")
    return round(float(psnr_fn(real, fake, data_range=255)), 2)


def compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    """SSIM between two (H, W, 3) uint8 images. Range [0, 1], higher is better."""
    if not SKIMAGE_AVAILABLE:
        return float("nan")
    return round(float(ssim_fn(real, fake, data_range=255, channel_axis=2, win_size=7)), 4)


def compute_all_metrics(
    real_images: list[np.ndarray],
    fake_images: list[np.ndarray],
    device: str = "cpu",
) -> dict:
    """
    Compute PSNR, SSIM, and FID for a set of image pairs.
    Returns aggregated mean metrics as a dict.
    """
    psnrs, ssims = [], []
    for real, fake in zip(real_images, fake_images):
        psnrs.append(compute_psnr(real, fake))
        ssims.append(compute_ssim(real, fake))

    fid = compute_fid(real_images, fake_images, device)

    return {
        "PSNR":  round(float(np.nanmean(psnrs)), 2),
        "SSIM":  round(float(np.nanmean(ssims)), 4),
        "FID":   fid,
        "n":     len(real_images),
    }


# ─── Co-Registration Verification ────────────────────────────────────────────

def verify_coregistration(ir_gray: np.ndarray, rgb_gray: np.ndarray) -> dict:
    """
    Verifies pixel-level alignment between IR and RGB images using
    Normalized Cross-Correlation (NCC).

    The Landsat 8/9 Level-2 STAC pipeline delivers pre-co-registered tiles,
    but this function provides a quantitative proof of alignment quality.

    Args:
        ir_gray  : (H, W) uint8 grayscale IR image
        rgb_gray : (H, W) uint8 grayscale version of RGB image

    Returns:
        dict with 'ncc' score (1.0 = perfect alignment) and 'aligned' boolean
    """
    ir_norm  = (ir_gray.astype(np.float32)  - ir_gray.mean())  / (ir_gray.std()  + 1e-8)
    rgb_norm = (rgb_gray.astype(np.float32) - rgb_gray.mean()) / (rgb_gray.std() + 1e-8)

    # Compute NCC via correlation
    ncc = float(np.mean(ir_norm * rgb_norm))
    ncc_normalized = (ncc + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

    return {
        "ncc":     round(ncc_normalized, 4),
        "aligned": ncc_normalized > 0.4,  # Threshold for good alignment
        "note":    "Landsat C2-L2 STAC data is pre-co-registered by USGS to ±0.3 pixel accuracy."
    }
