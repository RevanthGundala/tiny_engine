import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

class GameNGen(nn.Module):
    def __init__(self, model_id: str, timesteps: int):
        super().__init__()
        self.model_id = model_id
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(timesteps)

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

class ImageProj(nn.Module):
    def __init__(self, latent_channels: int, cross_attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_channels, cross_attention_dim)

    def forward(self, x):
        return self.proj(x)


