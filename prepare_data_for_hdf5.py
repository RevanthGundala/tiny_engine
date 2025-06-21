import os
import cv2
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm
import logging

def download_and_extract_frames(repo_id, data_dir="gamelogs", frames_sub_dir="frames"):
    """
    Downloads the dataset video and metadata from Hugging Face Hub,
    then extracts frames from the video into a directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # --- Setup Paths ---
    frames_dir = os.path.join(data_dir, frames_sub_dir)
    os.makedirs(frames_dir, exist_ok=True)
    metadata_path_local = os.path.join(data_dir, "metadata.csv")

    # --- Download Files ---
    logging.info(f"Downloading data from repository: {repo_id}")
    try:
        # Download metadata
        hf_hub_download(
            repo_id=repo_id,
            filename="metadata.csv",
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False # Copy file to data_dir
        )
        logging.info(f"Metadata downloaded to {metadata_path_local}")

        # Download video
        video_path = hf_hub_download(
            repo_id=repo_id,
            filename="dataset_video.mp4",
            repo_type="dataset"
        )
        logging.info(f"Video downloaded to {video_path}")

    except Exception as e:
        logging.error(f"Failed to download files from Hugging Face Hub: {e}")
        return

    # --- Extract Frames ---
    logging.info("Extracting frames from video...")
    try:
        video_capture = cv2.VideoCapture(video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_id in tqdm(range(total_frames), desc="Extracting Frames"):
            ret, frame = video_capture.read()
            if not ret:
                logging.warning(f"Could not read frame {frame_id}. Stopping.")
                break
            
            frame_filename = os.path.join(frames_dir, f"frame_{frame_id:09d}.png")
            cv2.imwrite(frame_filename, frame)
            
        video_capture.release()
        logging.info(f"Successfully extracted {total_frames} frames to {frames_dir}")

    except Exception as e:
        logging.error(f"An error occurred during frame extraction: {e}")


if __name__ == "__main__":
    REPO_ID = "RevanthGundala/vizdoom"
    download_and_extract_frames(repo_id=REPO_ID)
    print("\nExtraction complete. Now you can run:")
    print("python create_hdf5_dataset.py") 