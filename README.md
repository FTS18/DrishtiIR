#  DrishtiIR — Infrared Satellite Image Colorization & Enhancement

> **दृष्टि** *(Drishti)* = Vision in Sanskrit  
> **Bharatiya Antariksh Hackathon 2026 — Problem Statement 10**  
> Built with PyTorch Pix2Pix GAN · Landsat 8/9 Thermal Imagery · Streamlit

---

## Overview

An end-to-end deep learning framework that transforms single-channel infrared (thermal) satellite imagery into high-fidelity, colorized RGB images. The system simultaneously enhances structural detail (via super-resolution) and predicts realistic colorization using a U-Net Generator + PatchGAN Discriminator architecture.

```
IR (Band 10, 100m)  Super-Resolution  U-Net GAN  Colorized RGB (30m)
```

---

## Project Structure

```
ir-colorization/
 src/
    __init__.py        # Package marker
    dataset.py         # Landsat GeoTIFF loader + Synthetic fallback dataset
    model.py           # Generator, Discriminator, SRModule architectures
    train.py           # Training loop: L1 + Adversarial + SSIM loss
    inference.py       # Model loading, preprocessing, metrics
 app.py                 # Streamlit dashboard (4-tab UI)
 mock_weights.py        # Generates demo checkpoint (no training needed)
 requirements.txt       # Python dependencies
 README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Demo (No Training Required)
```bash
# Generate random model weights for demonstration
python mock_weights.py

# Launch the Streamlit dashboard
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Training on Real Landsat Data

### Step 1: Download Data
Download Landsat 8/9 imagery from [USGS EarthExplorer](https://earthexplorer.usgs.gov/):
- **IR input:** Band 10 (TIRS-1, 100m thermal) — save as `data/train/ir/*.tif`
- **RGB target:** Bands 4, 3, 2 stacked into a 3-channel GeoTIFF — save as `data/train/rgb/*.tif`

### Step 2: Train
```bash
python src/train.py \
  --ir-dir  data/train/ir  \
  --rgb-dir data/train/rgb \
  --num-epochs  100        \
  --batch-size  8          \
  --save-every  10
```

Checkpoints are saved to `checkpoints/` every 10 epochs. Sample output images are saved to `samples/`.

### Step 3: Resume Training
The training script automatically resumes from `checkpoints/generator_latest.pth` if it exists.

---

## CLI Inference

```bash
# Single image
python src/inference.py \
  --input      my_thermal_image.png \
  --output     colorized_result.png  \
  --checkpoint checkpoints/generator_latest.pth
```

---

## Architecture

### Generator (U-Net)
- 8-level encoder with progressive downsampling (Conv → BN → LeakyReLU)
- Bottleneck at spatial resolution 1×1 for global context
- 8-level decoder with skip connections from encoder (preserves spatial details)
- Output: Tanh activation → RGB values in [-1, 1]

### Discriminator (PatchGAN)
- Classifies overlapping 70×70 patches as real/fake
- Input: concatenated IR + RGB (real or generated)
- Encourages sharp, high-frequency textures rather than blurry outputs

### Super-Resolution Module
- Sub-pixel convolution (PixelShuffle) for 2× upscaling
- Upscales 100m thermal resolution to match 30m RGB grid

---

## Loss Functions

| Loss Component | Formula | Weight | Purpose |
|:---|:---|:---|:---|
| Adversarial | BCE(D(G(x)), 1) | 1.0 | Force realistic outputs |
| L1 Pixel | mean(|G(x) - y|) | λ=100 | Pixel-level fidelity |
| SSIM | 1 - SSIM(G(x), y) | λ=20 | Structural integrity |

---

## Evaluation Metrics

| Metric | Description | Target |
|:---|:---|:---|
| **PSNR** | Peak Signal-to-Noise Ratio | > 28 dB |
| **SSIM** | Structural Similarity Index | > 0.85 |
| **FID** | Fréchet Inception Distance | < 50 |
| **Inference Time** | Per-tile (256×256) speed | < 100 ms (CPU) |

---

## Dataset

| Band | Type | Resolution | Use |
|:---|:---|:---|:---|
| Band 10 (TIRS-1) | 10.6–11.2 μm Thermal IR | 100m → 30m | Model Input |
| Band 11 (TIRS-2) | 11.5–12.5 μm Thermal IR | 100m → 30m | Supplementary |
| Band 4 (Red) | 0.64–0.67 μm | 30m | RGB Target |
| Band 3 (Green) | 0.53–0.59 μm | 30m | RGB Target |
| Band 2 (Blue) | 0.45–0.51 μm | 30m | RGB Target |

---

## Dashboard Tabs

| Tab | Description |
|:---|:---|
|  **Single Image** | Upload any IR image and see real-time colorization |
|  **Demo Mode** | Generate synthetic thermal scenes procedurally |
|  **Batch Evaluate** | Process multiple images and aggregate metrics |
|  **About** | Architecture overview, loss functions, quick-start commands |
