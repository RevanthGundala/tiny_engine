import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

class GameNGen(nn.Module):
    def __init__(self, model_id: str, timesteps: int, history_len: int):
        super().__init__()
        self.model_id = model_id
        self.history_len = history_len
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(timesteps)

        # Modify the U-Net to accept history
        original_in_channels = self.unet.config.in_channels # Should be 4
        new_in_channels = original_in_channels * (1 + self.history_len)

        original_conv_in = self.unet.conv_in

        self.unet.conv_in = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_conv_in.out_channels,
            kernel_size=original_conv_in.kernel_size,
            stride=original_conv_in.stride,
            padding=original_conv_in.padding,
        )

        # Initialize the new weights
        with torch.no_grad():
            # Copy original weights for the main noisy latent
            self.unet.conv_in.weight[:, :original_in_channels, :, :] = original_conv_in.weight
            # Zero-initialize weights for the history latents
            self.unet.conv_in.weight[:, original_in_channels:, :, :].zero_()
            # Copy bias
            self.unet.conv_in.bias = original_conv_in.bias

        # Update the model's config
        self.unet.config.in_channels = new_in_channels

        # not training so freeze
        self.vae.requires_grad_(False)

    def forward(self, noisy_latents: torch.Tensor, timesteps: int, conditioning: torch.Tensor) -> torch.Tensor:
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=conditioning
        ).sample


        return noise_pred

class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int, cross_attention_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_actions, out_features=cross_attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=cross_attention_dim, out_features=cross_attention_dim)
        )

    def forward(self, x):
        return self.encoder(x)