from tqdm import tqdm
from model import GameNGen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
from huggingface_hub import hf_hub_download
import cv2

class NextFrameDataset(Dataset):
    def __init__(self, metadata_path: str, video_path: str, image_size: tuple):
        self.metadata = pd.read_csv(metadata_path)
        self.video_capture = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize VAE to [-1, 1]
        ])

    def __len__(self) -> int:
        return min(len(self.metadata) - 1, self.total_frames - 1)

    def __getitem__(self, idx: int) -> dict:
        # Set video position for desired frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, curr_frame_bgr = self.video_capture.read()
        if not ret: raise IndexError(f"Could not read {idx} from video.")
        curr_frame_rgb = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)

        # Read next frame
        ret, next_frame_bgr = self.video_capture.read() # read() automatically advances to the next frame
        if not ret:
            raise IndexError(f"Could not read frame {idx + 1} from video.")
        next_frame_rgb = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert NumPy arrays from OpenCV to PIL Images before transforming
        curr_pil_image = Image.fromarray(curr_frame_rgb)
        next_pil_image = Image.fromarray(next_frame_rgb)

        # Apply transforms
        curr_image = self.transform(curr_pil_image)
        next_image = self.transform(next_pil_image)

        # Get the action
        curr_row = self.metadata.iloc[idx]
        action_data = json.loads(str(curr_row['action']))
        if not isinstance(action_data, list):
            action_data = [action_data] # Wrap single number in a list
        curr_action = torch.tensor(action_data, dtype=torch.float32)

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
    
    # Define file paths using the config
    logging.info("Downloading data from hugging face...")
    metadata_path = hf_hub_download(repo_id=config.repo_id, filename="metadata.csv", repo_type="dataset")
    video_path = hf_hub_download(repo_id=config.repo_id, filename="dataset_video.mp4", repo_type="dataset")
    engine = GameNGen(config.model_id, config.num_timesteps)

    # --- Memory Saving Optimizations ---
    engine.unet.enable_gradient_checkpointing()
    # try:
    #     engine.unet.enable_xformers_memory_efficient_attention()
    #     logging.info("xformers memory-efficient attention enabled.")
    # except ImportError:
    #     logging.warning("xformers is not installed. For better memory efficiency, run: pip install xformers")

    dataset = NextFrameDataset(metadata_path, video_path, config.image_size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )  

    num_actions = len(dataset[0]["action"])
    cross_attention_dim = engine.unet.config.cross_attention_dim 
    action_encoder = nn.Sequential(
        nn.Linear(in_features=num_actions, out_features=cross_attention_dim),
        nn.SiLU(inplace=True),
        nn.Linear(in_features=cross_attention_dim, out_features=cross_attention_dim)
    )
    image_proj = nn.Linear(engine.vae.config.latent_channels, cross_attention_dim)

    params_to_train = list(engine.unet.parameters()) + list(action_encoder.parameters()) + list(image_proj.parameters())
    optim = torch.optim.AdamW(params=params_to_train, lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim, num_warmup_steps=500, num_training_steps=len(dataloader) * config.num_epochs
    )
    engine, action_encoder, image_proj, optim, dataloader, lr_scheduler = accelerator.prepare(
        engine, action_encoder, image_proj, optim, dataloader, lr_scheduler
    )
    logging.info("Starting training loop...")
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
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
            b, c, h, w = image_conditioning_sample.shape
            image_conditioning = image_conditioning_sample.permute(0, 2, 3, 1).reshape(b, h * w, c)
            image_conditioning = image_proj(image_conditioning)

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
                noise_pred = engine(noisy_latents, timesteps, augmented_conditioning)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(engine.unet.parameters(), 1.0)
                optim.step()
                lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        # --- Save Model Checkpoint ---
        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            unet_save_path = os.path.join(config.output_dir, f"unet_epoch_{epoch}.pth")
            action_encoder_save_path = os.path.join(config.output_dir, f"action_encoder_epoch_{epoch}.pth")
            image_proj_save_path = os.path.join(config.output_dir, f"image_proj_epoch_{epoch}.pth")
            
            accelerator.save(accelerator.unwrap_model(engine).unet.state_dict(), unet_save_path)
            accelerator.save(accelerator.unwrap_model(action_encoder).state_dict(), action_encoder_save_path)
            accelerator.save(accelerator.unwrap_model(image_proj).state_dict(), image_proj_save_path)
            logging.info(f"Saved model checkpoints for epoch {epoch}")

if __name__ == "__main__":
    train()