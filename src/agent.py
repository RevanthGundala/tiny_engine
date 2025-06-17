from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from PIL import Image
import logging
import json
import numpy as np
import csv
import gymnasium
from vizdoom import gymnasium_wrapper # This import is needed to register the env

DATASET_DIR = "gamelogs"
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class GameNGenCallback(BaseCallback):
    def __init__(self, verbose: bool, save_path: str):
        super(GameNGenCallback, self).__init__(verbose)
        self.save_path = save_path
        self.frame_log = open(os.path.join(self.save_path, "metadata.csv"), mode="w", newline="")
        self.csv_writer = csv.writer(self.frame_log)
        # CSV Header
        self.csv_writer.writerow(["frame_id", "action"])
    
    def _on_step(self) -> bool:
        frame_id = self.n_calls
        key = f"{frame_id:09d}"

        try:
            obs_dict = self.locals["new_obs"]
            # The observation from the callback is in Channels-First format (C, H, W)
            frame_data = obs_dict['screen'][0]
            action = self.locals["actions"][0]

            # --- DEFINITIVE FIX ---
            # Check if the frame is in the expected Channels-First format (C, H, W).
            # A valid RGB image will have 3 channels in its first dimension.
            if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                # Pillow's fromarray function needs the image in Channels-Last format (H, W, C).
                # We must transpose the axes from (C, H, W) to (H, W, C).
                transposed_frame = np.transpose(frame_data, (1, 2, 0))
                image = Image.fromarray(transposed_frame)
                image.save(os.path.join(FRAMES_DIR, f"frame_{key}.png"))
                
                json_action = json.dumps(action, cls=NpEncoder)
                self.csv_writer.writerow([key, json_action])
            else:
                # This will now correctly catch the junk frames from terminal states.
                logging.warning(f"Skipping corrupted frame {key} with invalid shape: {frame_data.shape}")

        except Exception as e:
            # This will now only catch truly unexpected errors.
            logging.error(f"Could not process or save frame {key} due to an unexpected error: {e}")

        return True

    def _on_training_end(self) -> None:
        self.frame_log.close()

# --- Main script ---
logging.basicConfig(level=logging.INFO)

# Create the VizDoom environment. No wrappers are needed.
env = gymnasium.make("VizdoomHealthGatheringSupreme-v0")

callback = GameNGenCallback(verbose=True, save_path=DATASET_DIR)

model = PPO(
    "MultiInputPolicy",
    env,  
    verbose=1,
)

model.learn(total_timesteps=2_000_000, callback=callback)

env.close()
