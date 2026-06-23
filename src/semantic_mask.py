"""
semantic_mask.py
----------------
Spectral-Index-Based Semantic Land Cover Classifier for IR→RGB Constraint.

Implements Gap #2 from ISRO PS-10:
  "Incorporate a semantic mask or pre-trained land-cover classifier to ensure
   specific IR signatures (e.g., water) are consistently mapped to the correct
   colors (e.g., blue)."

Approach:
  We use the multi-band input (B10=Thermal, B6=SWIR, B5=NIR) to compute
  spectral indices that are physically proven to identify land cover classes:
  
  - NDWI-like  : High Thermal + Low SWIR → Water bodies (rivers, lakes)
  - NDVI-like  : High NIR + Low Thermal  → Vegetation (forests, crops)
  - Bare Soil  : High SWIR + Mid NIR     → Dry land, deserts, roads
  - Urban/Heat : Very High Thermal       → Cities, buildings

Then, a soft "semantic correction" blend is applied post-inference to nudge
any pixels the model mislabeled back toward their physically correct color:
  - Water pixels → bias toward blue
  - Vegetation pixels → bias toward green
  - Urban pixels → bias toward gray/brown
"""

import numpy as np
import cv2


# ─── Spectral Index Constants ─────────────────────────────────────────────────
# For Landsat C2-L2 DN values (normalized 0→1):
# B10 (Thermal) : higher = warmer surface temperature
# B6  (SWIR)    : higher = dry soil, urban, bare land; lower = water/ice
# B5  (NIR)     : higher = dense vegetation; lower = water, bare soil

WATER_THERMAL_MAX  = 0.45  # Water is cool in thermal
WATER_SWIR_MAX     = 0.30  # Water absorbs SWIR strongly
VEGE_NIR_MIN       = 0.55  # Dense vegetation is bright in NIR
VEGE_THERMAL_MAX   = 0.60  # Vegetation keeps cool
URBAN_THERMAL_MIN  = 0.65  # Cities retain heat
BARE_SWIR_MIN      = 0.50  # Dry soil reflects SWIR


# ─── Core Classifier ──────────────────────────────────────────────────────────

def classify_landcover(ir_band: np.ndarray) -> np.ndarray:
    """
    Classify each pixel into 4 land cover classes from multi-band IR input.

    Args:
        ir_band: (C, H, W) normalized float32 array in [0, 1].
                 C=1: thermal only → falls back to thermal-only classification
                 C=3: [B10, B6, B5] (Thermal, SWIR, NIR)

    Returns:
        mask: (H, W) uint8 array with values:
              0 = Unknown / Mixed
              1 = Water (→ Blue)
              2 = Vegetation (→ Green)
              3 = Urban / Bare Soil (→ Gray/Brown)
    """
    C, H, W = ir_band.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    if C == 1:
        # Single-band thermal: only detect urban heat islands
        thermal = ir_band[0]
        mask[thermal > URBAN_THERMAL_MIN] = 3
        mask[thermal < WATER_THERMAL_MAX] = 1  # Cool = possibly water
        return mask

    # Multi-band: [B10=Thermal, B6=SWIR, B5=NIR]
    thermal = ir_band[0]
    swir    = ir_band[1]
    nir     = ir_band[2]

    # Water: cool thermal + low SWIR (water absorbs SWIR)
    water_mask = (thermal < WATER_THERMAL_MAX) & (swir < WATER_SWIR_MAX)
    mask[water_mask] = 1

    # Vegetation: high NIR + moderate thermal
    vege_mask = (nir > VEGE_NIR_MIN) & (thermal < VEGE_THERMAL_MAX) & (~water_mask)
    mask[vege_mask] = 2

    # Urban/Bare: high thermal + moderate SWIR
    urban_mask = (thermal > URBAN_THERMAL_MIN) | (swir > BARE_SWIR_MIN)
    urban_mask = urban_mask & (~water_mask) & (~vege_mask)
    mask[urban_mask] = 3

    # Morphological cleanup: remove tiny noise patches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cls in [1, 2, 3]:
        binary = (mask == cls).astype(np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        mask[mask == cls] = 0
        mask[cleaned == 1] = cls

    return mask


def get_semantic_color_map(mask: np.ndarray) -> np.ndarray:
    """
    Converts a land cover class mask to a color visualization.

    Returns:
        color_map: (H, W, 3) uint8 RGB image
                   Water → #2196F3 (blue)
                   Vegetation → #4CAF50 (green)
                   Urban → #9E9E9E (gray)
                   Unknown → #212121 (dark)
    """
    H, W = mask.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    color_map[mask == 0] = [33,  33,  33]   # Unknown: dark
    color_map[mask == 1] = [33, 150, 243]   # Water: blue
    color_map[mask == 2] = [76, 175,  80]   # Vegetation: green
    color_map[mask == 3] = [158, 158, 158]  # Urban/Bare: gray

    return color_map


def apply_semantic_correction(
    rgb_output: np.ndarray,   # (H, W, 3) uint8 - generated RGB
    land_mask: np.ndarray,    # (H, W) uint8 - class mask from classify_landcover
    strength: float = 0.25,   # Blend strength (0=no correction, 1=full override)
) -> np.ndarray:
    """
    Semantic Correction Post-Processing:
    
    Blends the model's generated RGB output with the "ideal" semantic colors.
    This enforces physical color constraints:
      - Water regions cannot be orange/brown → they get nudged toward blue
      - Vegetation regions cannot be completely gray → nudged toward green
      - Urban regions cannot be completely saturated green → nudged toward gray
    
    The `strength` parameter controls how strongly to apply the correction.
    0.25 is subtle enough that the model's artistic output is preserved while
    physically impossible colors are gently corrected.

    Args:
        rgb_output : (H, W, 3) uint8 generated image from the AI
        land_mask  : (H, W) uint8 class map from classify_landcover
        strength   : float in [0, 1] — correction blend weight

    Returns:
        corrected  : (H, W, 3) uint8 semantically constrained image
    """
    corrected = rgb_output.astype(np.float32).copy()

    # Water correction: bias toward blue channel
    water_px = (land_mask == 1)
    if water_px.any():
        corrected[water_px, 0] *= (1 - strength * 0.5)   # Reduce Red
        corrected[water_px, 1] *= (1 - strength * 0.3)   # Reduce Green
        corrected[water_px, 2] = corrected[water_px, 2] * (1 - strength) + 200 * strength  # Boost Blue

    # Vegetation correction: bias toward green channel
    vege_px = (land_mask == 2)
    if vege_px.any():
        corrected[vege_px, 0] *= (1 - strength * 0.3)   # Reduce Red
        corrected[vege_px, 1] = corrected[vege_px, 1] * (1 - strength) + 140 * strength  # Boost Green
        corrected[vege_px, 2] *= (1 - strength * 0.3)   # Reduce Blue

    # Urban correction: desaturate toward neutral gray/brown
    urban_px = (land_mask == 3)
    if urban_px.any():
        gray_val = (
            0.299 * corrected[urban_px, 0]
            + 0.587 * corrected[urban_px, 1]
            + 0.114 * corrected[urban_px, 2]
        )
        for ch in range(3):
            corrected[urban_px, ch] = (
                corrected[urban_px, ch] * (1 - strength * 0.4)
                + gray_val * (strength * 0.4)
            )

    return np.clip(corrected, 0, 255).astype(np.uint8)


# ─── Semantic Consistency Loss (for training) ─────────────────────────────────

def spectral_semantic_loss(
    fake_rgb: "torch.Tensor",    # (B, 3, H, W) in [-1, 1]
    ir_cond:  "torch.Tensor",    # (B, C, H, W) in [-1, 1]
    device:   str = "cuda",
) -> "torch.Tensor":
    """
    Semantic Consistency Loss for training:
    Penalizes the generator when it assigns physically wrong colors to pixels
    that are clearly identified as water or vegetation by the spectral bands.

    Usage in train.py:
        loss_sem = spectral_semantic_loss(fake_rgb, ir_batch, device)
        loss = loss_mse + 0.05 * loss_sem

    This is a differentiable soft constraint:
    - Water pixels should have high Blue channel relative to Red
    - Vegetation pixels should have high Green channel relative to Red
    """
    import torch

    B, C, H, W = ir_cond.shape
    # Denormalize from [-1, 1] to [0, 1]
    ir_01 = (ir_cond + 1.0) / 2.0
    rgb_01 = (fake_rgb + 1.0) / 2.0

    loss = torch.tensor(0.0, device=device)

    if C >= 3:
        thermal = ir_01[:, 0]  # (B, H, W)
        swir    = ir_01[:, 1]
        nir     = ir_01[:, 2]

        r, g, b = rgb_01[:, 0], rgb_01[:, 1], rgb_01[:, 2]

        # Water pixels should be more blue than red
        water_mask = ((thermal < WATER_THERMAL_MAX) & (swir < WATER_SWIR_MAX)).float()
        water_loss = torch.clamp(r - b + 0.1, min=0)  # Penalize red > blue in water
        loss = loss + (water_mask * water_loss).mean()

        # Vegetation pixels should be more green than red
        vege_mask = ((nir > VEGE_NIR_MIN) & (thermal < VEGE_THERMAL_MAX)).float()
        vege_loss = torch.clamp(r - g + 0.1, min=0)  # Penalize red > green in vegetation
        loss = loss + (vege_mask * vege_loss).mean()

    return loss
