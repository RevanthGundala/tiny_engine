from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    """Parameters defining the model architecture and basic properties."""
    model_id: str = "CompVis/stable-diffusion-v1-4"
    image_size: Tuple[int, int] = (240, 320)
    num_timesteps: int = 100
    history_len: int = 4
    num_actions: int = 7
    use_lora: bool = True

@dataclass
class TrainingConfig:
    """Parameters specific to the training process."""
    repo_id: str = "RevanthGundala/vizdoom" # Dataset repository
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 100
    lora_rank: int = 4 # Only used if ModelConfig.use_lora is True
    lora_alpha: int = 4 # Only used if ModelConfig.use_lora is True

@dataclass
class PredictionConfig:
    """Parameters for the prediction server (app.py)."""
    repo_id: str = "RevanthGundala/vizdoom" # For starting frame video
    prediction_epoch: int = 99
    output_dir: str = "output" # To load weights if not using MLflow
    action_map: Dict[str, List[int]] = field(default_factory=lambda: {
        "w": [1, 0, 0, 0, 0, 0, 0],  # MOVE_FORWARD
        "s": [0, 1, 0, 0, 0, 0, 0],  # MOVE_BACKWARD
        "d": [0, 0, 1, 0, 0, 0, 0],  # MOVE_RIGHT
        "a": [0, 0, 0, 1, 0, 0, 0],  # MOVE_LEFT
        "ArrowLeft": [0, 0, 0, 0, 1, 0, 0], # TURN_LEFT
        "ArrowRight": [0, 0, 0, 0, 0, 1, 0], # TURN_RIGHT
        " ": [0, 0, 0, 0, 0, 0, 1], # ATTACK
        "noop": [0, 0, 0, 0, 0, 0, 0], # No operation
    })