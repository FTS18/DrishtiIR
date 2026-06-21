"""
model.py
--------
Neural network architectures for the IR-to-RGB colorization framework.

Components:
  - UNetBlock      : Reusable encoder/decoder block with BatchNorm and activations
  - Generator      : U-Net based IR → RGB image translator with skip connections
  - Discriminator  : PatchGAN discriminator for texture-level adversarial training
  - SRModule       : Lightweight pixel-shuffle super-resolution upscaler
  - ColorizeNet    : Full end-to-end pipeline combining SR and Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvNormAct(nn.Module):
    """Single Convolution → BatchNorm → Activation block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: str = "leaky",   # "leaky", "relu", or "none"
        transposed: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        conv_cls = nn.ConvTranspose2d if transposed else nn.Conv2d
        layers.append(
            conv_cls(in_ch, out_ch, kernel, stride, padding, bias=not norm)
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        # "none" means no activation (used at final output layers)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Super-Resolution Module ──────────────────────────────────────────────────

class SRModule(nn.Module):
    """
    Lightweight sub-pixel convolution (ESPCN-style) super-resolution.

    Upscales a single-channel IR image by factor `scale` (default: 2)
    to match the RGB spatial resolution before colorization.
    Output: single-channel, higher-resolution sharpened IR.
    """

    def __init__(self, in_channels: int = 1, scale: int = 2, features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upscale via sub-pixel shuffle
            nn.Conv2d(features * 2, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── U-Net Generator ─────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Pix2Pix-style U-Net Generator: IR (1ch) → RGB (3ch).

    Architecture:
    - Encoder: 8 progressive downsampling stages
    - Bottleneck: Dense convolution at 1×1 spatial bottleneck
    - Decoder: 8 upsampling stages with skip connections from encoder
    - Skip connections preserve spatial detail (avoids blurry outputs)

    Input:  (B, 1, 256, 256) — normalized single-channel IR
    Output: (B, 3, 256, 256) — normalized RGB in [-1, 1]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, features: int = 64):
        super().__init__()
        f = features

        # ── Encoder ──────────────────────────────────────────────────────────
        # No BatchNorm on the first layer (standard Pix2Pix convention)
        self.enc1 = ConvNormAct(in_channels, f,       norm=False, activation="leaky")  # 128
        self.enc2 = ConvNormAct(f,           f * 2,               activation="leaky")  # 64
        self.enc3 = ConvNormAct(f * 2,       f * 4,               activation="leaky")  # 32
        self.enc4 = ConvNormAct(f * 4,       f * 8,               activation="leaky")  # 16
        self.enc5 = ConvNormAct(f * 8,       f * 8,               activation="leaky")  # 8
        self.enc6 = ConvNormAct(f * 8,       f * 8,               activation="leaky")  # 4
        self.enc7 = ConvNormAct(f * 8,       f * 8,               activation="leaky")  # 2

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1),           # 1×1
            nn.ReLU(inplace=True),
        )

        # ── Decoder (mirrored encoder + skip connections) ─────────────────────
        # Skip connection doubles input channels for each decoder block
        self.dec1 = ConvNormAct(f * 8,       f * 8, activation="relu", transposed=True)  # 2   (+ skip enc7)
        self.dec2 = ConvNormAct(f * 8 * 2,   f * 8, activation="relu", transposed=True)  # 4   (+ skip enc6)
        self.dec3 = ConvNormAct(f * 8 * 2,   f * 8, activation="relu", transposed=True)  # 8   (+ skip enc5)
        self.dec4 = ConvNormAct(f * 8 * 2,   f * 8, activation="relu", transposed=True)  # 16  (+ skip enc4)
        self.dec5 = ConvNormAct(f * 8 * 2,   f * 4, activation="relu", transposed=True)  # 32  (+ skip enc3)
        self.dec6 = ConvNormAct(f * 4 * 2,   f * 2, activation="relu", transposed=True)  # 64  (+ skip enc2)
        self.dec7 = ConvNormAct(f * 2 * 2,   f,     activation="relu", transposed=True)  # 128 (+ skip enc1)

        # Final transposed conv to reconstruct spatial size and produce 3-ch RGB
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Dropout used in the first three decoder blocks (regularization)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encode ────────────────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        bn = self.bottleneck(e7)

        # ── Decode (with skip connections and dropout) ─────────────────────────
        d1 = self.dropout(self.dec1(bn))
        d2 = self.dropout(self.dec2(torch.cat([d1, e7], dim=1)))
        d3 = self.dropout(self.dec3(torch.cat([d2, e6], dim=1)))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.final(torch.cat([d7, e1], dim=1))


# ─── PatchGAN Discriminator ───────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator: classifies overlapping 70×70 image patches as real/fake.

    Better at capturing high-frequency texture than a full-image discriminator.
    Input: IR image concatenated with RGB image (real or generated) → 4 channels total.
    Output: Spatial map of real/fake patch predictions (not a single scalar).
    """

    def __init__(self, in_channels: int = 1, features: list[int] = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # First block: no BatchNorm
        layers: list[nn.Module] = [
            ConvNormAct(in_channels + 3, features[0], norm=False, activation="leaky")
        ]

        for i in range(1, len(features)):
            stride = 1 if i == len(features) - 1 else 2   # Last conv is stride 1
            layers.append(
                ConvNormAct(features[i - 1], features[i], stride=stride, activation="leaky")
            )

        # Final 1-channel prediction map (no activation; BCEWithLogitsLoss handles it)
        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for stable GAN training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Concatenate IR and RGB inputs along channel axis before discriminating."""
        return self.model(torch.cat([ir, rgb], dim=1))


# ─── Full End-to-End Pipeline ─────────────────────────────────────────────────

class ColorizeNet(nn.Module):
    """
    Full inference-only pipeline: SR upscaling + colorization in one forward pass.

    Used at deployment/inference time. During training, SRModule and Generator
    are trained separately for stability.
    """

    def __init__(self, sr_scale: int = 2):
        super().__init__()
        self.sr = SRModule(scale=sr_scale)
        self.gen = Generator()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sr_out  : super-resolved IR (for visualization)
          rgb_out : colorized RGB output
        """
        sr_out = self.sr(x)
        rgb_out = self.gen(sr_out)
        return sr_out, rgb_out


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model smoke tests on: {device}")

    dummy_ir = torch.randn(2, 1, 256, 256).to(device)
    dummy_rgb = torch.randn(2, 3, 256, 256).to(device)

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    gen_out = gen(dummy_ir)
    disc_out = disc(dummy_ir, gen_out)

    print(f"Generator output  : {gen_out.shape}")   # (2, 3, 256, 256)
    print(f"Discriminator map : {disc_out.shape}")  # (2, 1, 30, 30)

    # Parameter counts
    gen_params  = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator params     : {gen_params:,}")
    print(f"Discriminator params : {disc_params:,}")
    print("Model smoke test PASSED.")
