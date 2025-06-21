import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json
import logging

def create_hdf5_from_frames(frames_dir, metadata_path, output_path, image_size=(64, 64)):
    """
    Creates an HDF5 file from a directory of frame images and a metadata CSV.

    Args:
        frames_dir (str): Path to the directory containing frame images (e.g., 'frame_000000000.png').
        metadata_path (str): Path to the metadata.csv file.
        output_path (str): Path to save the output HDF5 file.
        image_size (tuple): The target size to resize images to (height, width).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    metadata = pd.read_csv(metadata_path)
    
    # Filter out any missing frames and sort
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    frame_files.sort()

    num_frames = len(frame_files)
    if num_frames == 0:
        logging.error("No frames found in the directory.")
        return

    logging.info(f"Found {num_frames} frames. Processing...")

    # Ensure metadata and frames align. This is a safeguard.
    if num_frames != len(metadata):
        logging.warning(f"Mismatch between number of frames ({num_frames}) and metadata entries ({len(metadata)}). Truncating to the smaller count.")
        min_count = min(num_frames, len(metadata))
        metadata = metadata.iloc[:min_count]
        frame_files = frame_files[:min_count]
        num_frames = min_count

    with h5py.File(output_path, 'w') as hf:
        # Create datasets
        # Storing frames as uint8 to save space. They will be normalized during training.
        frames_dset = hf.create_dataset('frames', (num_frames, image_size[0], image_size[1], 3), dtype=np.uint8, compression="gzip")
        
        # Determine action shape from the first valid action
        first_action_data = json.loads(str(metadata.iloc[0]['action']))
        if not isinstance(first_action_data, list):
            first_action_data = [first_action_data]
        action_shape = (len(first_action_data),)

        actions_dset = hf.create_dataset('actions', (num_frames,) + action_shape, dtype=np.float32)

        logging.info(f"Resizing images to {image_size} and writing to HDF5 file...")
        for i, frame_file in enumerate(tqdm(frame_files, desc="Processing Frames")):
            img_path = os.path.join(frames_dir, frame_file)
            with Image.open(img_path) as img:
                img = img.resize((image_size[1], image_size[0])) # PIL resize uses (width, height)
                img_array = np.array(img, dtype=np.uint8) # (H, W, C)
                if img_array.ndim == 2: # Grayscale
                    img_array = np.stack([img_array]*3, axis=-1)
                frames_dset[i] = img_array

        logging.info("Writing actions to HDF5 file...")
        for i, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Actions"):
            action_data = json.loads(str(row['action']))
            if not isinstance(action_data, list):
                action_data = [action_data]
            actions_dset[i] = np.array(action_data, dtype=np.float32)

    logging.info(f"HDF5 dataset created successfully at {output_path}")
    logging.info(f"Contains {len(frames_dset)} frames and {len(actions_dset)} actions.")

if __name__ == '__main__':
    # These paths assume you run the script from the root of the 'tiny_engine' project
    FRAMES_DIR = "gamelogs/frames"
    METADATA_PATH = "gamelogs/metadata.csv"
    OUTPUT_HDF5_PATH = "gamelogs/dataset.hdf5"

    if not os.path.exists(FRAMES_DIR) or not os.path.exists(METADATA_PATH):
        print(f"Error: Frames directory ('{FRAMES_DIR}') or metadata file ('{METADATA_PATH}') not found.")
        print("Please run 'python src/agent.py' first to generate the game logs.")
    else:
        create_hdf5_from_frames(FRAMES_DIR, METADATA_PATH, OUTPUT_HDF5_PATH) 