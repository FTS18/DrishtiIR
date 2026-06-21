"""
model_diffusion.py
------------------
Conditional Denoising Diffusion Probabilistic Model (DDPM) architecture.
Uses HuggingFace Diffusers for state-of-the-art satellite image generation.
"""

import torch
import torch.nn as nn

try:
    from diffusers import UNet2DModel, DDPMScheduler
except ImportError:
    pass # Handled in train script

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional DDPM: 
    Input: Concatenation of IR (conditioning) + Noisy RGB
    Output: Predicted Noise for RGB
    """
    def __init__(self, ir_channels=1, rgb_channels=3, image_size=256):
        super().__init__()
        self.ir_channels = ir_channels
        self.rgb_channels = rgb_channels
        
        # We use the standard UNet2DModel from diffusers.
        # It expects `in_channels` = IR_channels + RGB_channels
        # It outputs `out_channels` = RGB_channels (the predicted noise)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=ir_channels + rgb_channels,
            out_channels=rgb_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, noisy_rgb: torch.Tensor, ir_cond: torch.Tensor, timesteps: torch.Tensor):
        """
        noisy_rgb: (B, 3, H, W)
        ir_cond: (B, C, H, W)
        timesteps: (B,)
        """
        # Concatenate condition (IR) and target (noisy RGB) along the channel dimension
        net_input = torch.cat([ir_cond, noisy_rgb], dim=1)
        
        # Predict the noise
        noise_pred = self.unet(net_input, timesteps).sample
        return noise_pred

def get_scheduler(num_train_timesteps=1000):
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear"
    )
