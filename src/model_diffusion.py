"""
model_diffusion.py
------------------
Conditional Diffusion Model for IR→RGB satellite colorization.

Architecture:
  - Simple learned Conv projection for IR conditioning (no Tanh squashing)
  - UNet2DModel from diffusers library
  - Cosine beta schedule for better texture learning
  - DDIM scheduler for fast 50-step inference
  - EMA weights for stable final model
"""

import copy
import torch
import torch.nn as nn

try:
    from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
except ImportError:
    pass  # Handled in train script


# ─── IR Feature Projector ─────────────────────────────────────────────────────

class IRProjector(nn.Module):
    """
    Lightweight convolutional projector that extracts features from multi-band IR
    input without destroying the signal.

    Unlike the old SRModule(scale=1) which applied Tanh() and squashed the
    conditioning signal, this uses LeakyReLU to preserve the full dynamic range.
    """

    def __init__(self, in_channels: int, out_channels: int = None, features: int = 32):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
            # No activation here — pass clean features to U-Net
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Conditional DDPM ─────────────────────────────────────────────────────────

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional DDPM for IR→RGB generation.

    The IR conditioning is projected through a small conv network and
    concatenated channel-wise with the noisy RGB before being fed to the U-Net.
    """

    def __init__(self, ir_channels: int = 1, rgb_channels: int = 3, image_size: int = 256):
        super().__init__()
        self.ir_channels  = ir_channels
        self.rgb_channels = rgb_channels

        # Project IR bands to a fixed number of conditioning channels
        self.ir_proj = IRProjector(in_channels=ir_channels, out_channels=ir_channels, features=32)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=ir_channels + rgb_channels,
            out_channels=rgb_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(
        self,
        noisy_rgb: torch.Tensor,    # (B, 3, H, W)
        ir_cond: torch.Tensor,      # (B, C, H, W) — can be zeros for CFG uncond
        timesteps: torch.Tensor,    # (B,)
        is_unconditional: bool = False,  # Skip IR projection for zero-input CFG
    ) -> torch.Tensor:
        """Predict the noise residual from the noisy RGB conditioned on IR."""

        if is_unconditional:
            proj_ir = ir_cond  # Already zeros, no need to project
        else:
            proj_ir = self.ir_proj(ir_cond)

        net_input = torch.cat([proj_ir, noisy_rgb], dim=1)
        return self.unet(net_input, timesteps).sample


# ─── EMA (Exponential Moving Average) ────────────────────────────────────────

class EMAModel:
    """
    Maintains an EMA copy of the model weights.
    Instead of saving the "last" weights, we save the smooth running average.
    Result: Much more stable, photorealistic generation at inference time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ─── Noise Schedulers ─────────────────────────────────────────────────────────

def get_ddpm_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """
    Cosine beta schedule (instead of linear).
    The cosine schedule adds/removes noise much more gradually at the extremes,
    forcing the model to spend more time learning subtle, photorealistic textures.
    """
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",  # Cosine schedule
        clip_sample=True,
    )


def get_ddim_scheduler(num_train_timesteps: int = 1000, num_inference_steps: int = 50) -> DDIMScheduler:
    """
    DDIM Scheduler: generates equally good images in 50 steps instead of 1000.
    This is 20x faster at inference/sampling time with no quality loss.
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    )
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


# ─── Legacy alias for backward compat ─────────────────────────────────────────

def get_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """Alias kept for backward compatibility with old train scripts."""
    return get_ddpm_scheduler(num_train_timesteps)
