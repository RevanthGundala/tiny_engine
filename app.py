import os
import io
import base64
import json
from PIL import Image
import cv2
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import GameNGen, ActionEncoder
from src.config import ModelConfig, PredictionConfig
from huggingface_hub import hf_hub_download

# --- FastAPI App ---
app = FastAPI()

# --- CORS ---
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
pred_config = PredictionConfig()

# Load models
engine = GameNGen(model_config.model_id, model_config.num_timesteps, history_len=model_config.history_len).to(device)
cross_attention_dim = engine.unet.config.cross_attention_dim
action_encoder = ActionEncoder(model_config.num_actions, cross_attention_dim).to(device)

# Load weights
# NOTE: Update epoch number if you have a different checkpoint
epoch = pred_config.prediction_epoch
if model_config.use_lora:
    try:
        unet_path = os.path.join(pred_config.output_dir, f"unet_lora_epoch_{epoch}.pth")
        engine.unet.load_lora_weights(unet_path)
    except FileNotFoundError:
        print(f"Warning: LoRA weights not found at {unet_path}. Using base UNet.")
else:
    try:
        unet_path = os.path.join(pred_config.output_dir, f"unet_epoch_{epoch}.pth")
        engine.unet.load_state_dict(torch.load(unet_path))
    except FileNotFoundError:
        print(f"Warning: UNet weights not found at {unet_path}. Using base UNet.")

try:
    action_encoder_path = os.path.join(pred_config.output_dir, f"action_encoder_epoch_{epoch}.pth")
    action_encoder.load_state_dict(torch.load(action_encoder_path))
except FileNotFoundError:
    print(f"Warning: Action encoder weights not found at {action_encoder_path}. Using randomly initialized weights.")


engine.eval()
action_encoder.eval()

# --- Session State & History ---
# Using a simple in-memory dictionary for a single-user demo.
# In a real multi-user app, this would be a more robust session management system.
session_state = {
    "frame_history": None,  # A deque of latent tensors
    "action_history": None, # A deque of action tensors
}


# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize(model_config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def tensor_to_base64(tensor):
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    # Convert to PIL Image
    img = transforms.ToPILImage()(tensor.squeeze(0))
    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- Action Mapping ---
action_map = pred_config.action_map

# --- API Endpoints ---
class PredictRequest(BaseModel):
    action: str

@app.get("/api/start")
def start_game():
    """Initializes a new game session and returns the first frame."""
    global session_state
    try:
        frames_dir = pred_config.frames_dir
        if not os.path.isdir(frames_dir):
            raise HTTPException(status_code=500, detail=f"Frames directory not found at {frames_dir}. Please run data preparation script.")
        
        frame_files = sorted(os.listdir(frames_dir))
        if not frame_files:
            raise HTTPException(status_code=500, detail=f"No frames found in {frames_dir}.")

        # For starting the game, let's just use the first frame from the dataset.
        # A more advanced implementation could allow choosing a starting point.
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        pil_image = Image.open(first_frame_path).convert("RGB")

        # Initialize histories
        with torch.no_grad():
            initial_frame_tensor = transform(pil_image).unsqueeze(0).to(device)
            initial_latent = engine.vae.encode(initial_frame_tensor).latent_dist.sample()
        
        session_state["frame_history"] = deque([initial_latent] * model_config.history_len, maxlen=model_config.history_len)
        
        noop_action = torch.tensor(action_map["noop"], dtype=torch.float32, device=device).unsqueeze(0)
        session_state["action_history"] = deque([noop_action] * model_config.history_len, maxlen=model_config.history_len)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        print("Game session started, initial history created.")
        return {"frame": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@torch.inference_mode()
@app.post("/api/predict")
def predict(request: PredictRequest):
    """Predicts the next frame based on the current server-side state and received action."""
    global session_state
    
    # Check if session is initialized
    if session_state["frame_history"] is None:
        raise HTTPException(status_code=400, detail="Game session not started. Please call /api/start first.")

    # Get action tensor from request
    action_list = action_map.get(request.action)
    if action_list is None:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    action_tensor = torch.tensor(action_list, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Inference ---
    # 1. Get the history of frame latents for channel-wise concatenation
    history_latents = torch.cat(list(session_state["frame_history"]), dim=1)

    # 2. Get conditioning embedding from the action
    action_conditioning = action_encoder(action_tensor)
    conditioning_batch = action_conditioning.unsqueeze(1)

    # 3. DDIM sampling
    # The new UNet expects a different number of input channels, so we create the noise tensor accordingly
    in_channels = engine.unet.config.in_channels
    latents = torch.randn((1, in_channels, model_config.image_size[0] // 8, model_config.image_size[1] // 8), device=device)
    
    # We need to split the latents for concatenation.
    # The first 4 channels are for the noisy latent, the rest are for the history.
    noisy_latents = latents[:, :4, :, :]
    
    # Concatenate the frame history with the noisy latents
    model_input = torch.cat([noisy_latents, history_latents], dim=1)

    for t in engine.scheduler.timesteps:
        noise_pred = engine(model_input, t, conditioning_batch)
        latents = engine.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Prepare input for the next step
        noisy_latents = latents[:, :4, :, :]
        model_input = torch.cat([noisy_latents, history_latents], dim=1)


    # 4. Decode latents to image
    # We only decode the first 4 channels, which correspond to the predicted frame
    predicted_latents_only = latents[:, :4, :, :]
    predicted_latent_unscaled = predicted_latents_only / engine.vae.config.scaling_factor
    image = engine.vae.decode(predicted_latent_unscaled).sample
    
    # --- Update State ---
    session_state["frame_history"].append(predicted_latent_unscaled)
    session_state["action_history"].append(action_tensor)

    # --- Response ---
    next_frame_base64 = tensor_to_base64(image.cpu())
    return {"next_frame": next_frame_base64}


if __name__ == "__main__":
    import uvicorn
    # Note: Using a single worker is important for in-memory model setup
    uvicorn.run(app, host="0.0.0.0", port=8000) 