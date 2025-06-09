import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

class GameNGen(nn.Module):
    def __init__(self, model_id: str, timesteps: int):
        super().__init__()
        self.model_id = model_id
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
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


