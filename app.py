import os
import io
import base64
import json
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import GameNGen, ActionEncoder, ImageProj
from src.config import TrainingConfig
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
config = TrainingConfig()

# Load models
engine = GameNGen(config.model_id, config.num_timesteps).to(device)
cross_attention_dim = engine.unet.config.cross_attention_dim
latent_channels = engine.vae.config.latent_channels
num_actions = 7  # Based on _vizdoom.ini

action_encoder = ActionEncoder(num_actions, cross_attention_dim).to(device)
image_proj = ImageProj(latent_channels, cross_attention_dim).to(device)

# Load weights
# NOTE: Update epoch number if you have a different checkpoint
epoch = 99 
if config.use_lora:
    try:
        unet_path = os.path.join(config.output_dir, f"unet_lora_epoch_{epoch}.pth")
        engine.unet.load_lora_weights(unet_path)
    except FileNotFoundError:
        print(f"Warning: LoRA weights not found at {unet_path}. Using base UNet.")
else:
    try:
        unet_path = os.path.join(config.output_dir, f"unet_epoch_{epoch}.pth")
        engine.unet.load_state_dict(torch.load(unet_path))
    except FileNotFoundError:
        print(f"Warning: UNet weights not found at {unet_path}. Using base UNet.")

try:
    action_encoder_path = os.path.join(config.output_dir, f"action_encoder_epoch_{epoch}.pth")
    action_encoder.load_state_dict(torch.load(action_encoder_path))
except FileNotFoundError:
    print(f"Warning: Action encoder weights not found at {action_encoder_path}. Using randomly initialized weights.")

try:
    image_proj_path = os.path.join(config.output_dir, f"image_proj_epoch_{epoch}.pth")
    image_proj.load_state_dict(torch.load(image_proj_path))
except FileNotFoundError:
    print(f"Warning: Image projection weights not found at {image_proj_path}. Using randomly initialized weights.")


engine.eval()
action_encoder.eval()
image_proj.eval()

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize(config.image_size),
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
action_map = {
    "w": [1, 0, 0, 0, 0, 0, 0],  # MOVE_FORWARD
    "s": [0, 1, 0, 0, 0, 0, 0],  # MOVE_BACKWARD
    "d": [0, 0, 1, 0, 0, 0, 0],  # MOVE_RIGHT
    "a": [0, 0, 0, 1, 0, 0, 0],  # MOVE_LEFT
    "ArrowLeft": [0, 0, 0, 0, 1, 0, 0], # TURN_LEFT
    "ArrowRight": [0, 0, 0, 0, 0, 1, 0], # TURN_RIGHT
    " ": [0, 0, 0, 0, 0, 0, 1], # ATTACK
    "noop": [0, 0, 0, 0, 0, 0, 0], # No operation
}

# --- API Endpoints ---
class PredictRequest(BaseModel):
    current_frame: str # base64 encoded image
    action: str

@app.get("/api/start")
def start_game():
    """Returns the first frame of the game."""
    try:
        video_path = hf_hub_download(repo_id=config.repo_id, filename="dataset_video.mp4", repo_type="dataset")
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        if not success:
            raise HTTPException(status_code=500, detail="Could not read first frame from video.")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {"frame": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@torch.inference_mode()
@app.post("/api/predict")
def predict(request: PredictRequest):
    """Predicts the next frame based on the current frame and action."""
    # Decode image
    try:
        img_data = base64.b64decode(request.current_frame)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Get action tensor
    action_list = action_map.get(request.action)
    if action_list is None:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    action_tensor = torch.tensor(action_list, dtype=torch.float32, device=device).unsqueeze(0)

    # Preprocess image
    current_frame_tensor = transform(img).unsqueeze(0).to(device)
    
    # --- Inference ---
    # 1. Encode current frame
    vae = engine.vae
    image_conditioning_sample = vae.encode(current_frame_tensor).latent_dist.sample()

    # 2. Get conditioning embeddings
    action_conditioning = action_encoder(action_tensor)
    
    b, c, h, w = image_conditioning_sample.shape
    image_conditioning = image_conditioning_sample.permute(0, 2, 3, 1).reshape(b, h * w, c)
    image_conditioning = image_proj(image_conditioning)
    
    action_conditioning = action_conditioning.unsqueeze(1)
    conditioning_batch = torch.cat([image_conditioning, action_conditioning], dim=1)

    # 3. DDIM sampling
    latents = torch.randn((1, engine.unet.config.in_channels, config.image_size[0] // 8, config.image_size[1] // 8), device=device)
    
    for t in engine.scheduler.timesteps:
        noise_pred = engine(latents, t, conditioning_batch)
        latents = engine.scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Decode latents to image
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    
    # --- Response ---
    next_frame_base64 = tensor_to_base64(image.cpu())
    return {"next_frame": next_frame_base64}


if __name__ == "__main__":
    import uvicorn
    # Note: Using a single worker is important for in-memory model setup
    uvicorn.run(app, host="0.0.0.0", port=8000) 