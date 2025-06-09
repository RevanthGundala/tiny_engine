import tqdm
from model import GameNGen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import TrainingConfig
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
import json
import logging
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

class NextFrameDataset():
    def __init__(self, metadata_path: str, frames_dir: str, image_size: tuple):
        self.metadata = pd.read_csv(metadata_path)
        self.frames_dir = frames_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize VAE to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.metadata) - 1

    def __getitem___(self, idx: int) -> dict:
        # Get the metadata for the current and next step
        curr_row = self.metadata.iloc[idx]
        next_row = self.metadata.iloc[idx + 1]

        # Construct the full file paths correctly
        curr_frame_path = os.path.join(self.frames_dir, f"frame_{curr_row['frame_id']}.png")
        next_frame_path = os.path.join(self.frames_dir, f"frame_{next_row['frame_id']}.png")

        # Get the action string from the metadata
        action_list = json.loads(curr_row['action'])

        curr_image = self.transform(Image.open(curr_frame_path).convert("RGB"))
        next_image = self.transform(Image.open(next_frame_path).convert("RGB"))
        curr_action = torch.tensor(action_list, dtype=torch.float32)

        return {
            "current_frame": curr_image, 
            "action": curr_action, 
            "next_frame": next_image
        }

def train():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # --- Setup ---
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1
    )
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define file paths using the config
    metadata_path = os.path.join(config.data_dir, "metadata.csv")
    frames_dir = os.path.join(config.data_dir, "frames")
    engine = GameNGen(config.model_id, config.num_timesteps).to(device)
    dataset = NextFrameDataset(metadata_path, frames_dir, config.image_size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True
    )  
    optim = torch.optim.Adagrad(params=engine.unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim, num_warmup_steps=500, num_training_steps=len(dataloader) * config.num_epochs
    )
    num_actions = len(json.loads(dataset.iloc[0]["action"]))
    cross_attention_dim = engine.unet.config.cross_attention_dim 
    action_encoder = nn.Sequential(
        nn.Linear(in_features=num_actions, out_features=cross_attention_dim),
        nn.SiLU(inplace=True),
        nn.Linear(in_features=cross_attention_dim, out_features=cross_attention_dim)
    )
    engine, action_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        engine, action_encoder, optimizer, train_dataloader, lr_scheduler
    )
    logging.info("Starting training loop...")
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for batch in dataloader:
            optim.zero_grad()
            next_frames, actions, current_frames = batch["next_frame"], batch["action"], batch["current_frame"]

            # Encode into latent space
            with torch.no_grad():
                vae = accelerator.unwrap_model(engine).vae
                latent_dist = vae.encode(next_frames).latent_dist
                clean_latents = latent_dist.sample() * vae.config.scaling_factor
                image_conditioning_sample = vae.encode(current_frames).latent_dist.sample()
            
            action_conditioning = action_encoder(actions)
            # Add a "sequence" dimension to  image and action embeddings
            image_conditioning = image_conditioning_sample.view(image_conditioning_sample.size(0), -1, image_conditioning_sample.size(1))
            action_conditioning = action_conditioning.unsqueeze(1)
            conditioning_batch = torch.cat([image_conditioning, action_conditioning], dim=1)
            # create random noise
            noise = torch.randn_like(clean_latents)

            # pick random timestep. High timstep means more noise
            timesteps = torch.randint(0, engine.scheduler.config.num_train_timesteps, (clean_latents.shape[0], ), device=clean_latents.device).long() 

            noisy_latents = engine.scheduler.add_noise(clean_latents, noise, timesteps)

            # Fix autoregressive drift
            noise_strength = 0.02
            augmentation_noise = torch.randn_like(conditioning_batch) * noise_strength
            augmented_conditioning = augmentation_noise + conditioning_batch

            with accelerator.accumulate(engine):
                noise_pred = engine(noisy_latents, augmented_conditioning)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(engine.unet.parameters(), 1.0)
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        # --- Save Model Checkpoint ---
        if accelerator.is_main_process:
            unet_save_path = os.path.join(config.output_dir, f"unet_epoch_{epoch}.pth")
            action_encoder_save_path = os.path.join(config.output_dir, f"action_encoder_epoch_{epoch}.pth")
            
            accelerator.save(accelerator.unwrap_model(engine).unet.state_dict(), unet_save_path)
            accelerator.save(accelerator.unwrap_model(action_encoder).state_dict(), action_encoder_save_path)
            logging.info(f"Saved model checkpoints for epoch {epoch}")

if __name__ == "__main__":
    train()