from dataclasses import dataclass

@dataclass
class TrainingConfig():
    learning_rate = 1e-4
    model_id = "CompVis/stable-diffusion-v1-4"
    num_timesteps = 100
    batch_size = 1
    image_size = (240, 320)
    num_epochs = 100
    repo_id = "RevanthGundala/vizdoom"
    output_dir = "output"