import os
import argparse
from huggingface_hub import hf_hub_download
from src.config.schema import SystemSettings

def run_download(config_path: str):
    """
    For Docker:
    Downloads checkpoint from Hugging Face if not present.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    settings = SystemSettings.load_from_yaml(config_path)

    repo_id = settings.model.hf_repo_id
    filename = settings.model.hf_filename
    checkpoint_path = settings.model.checkpoint_path
    output_dir = os.path.dirname(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(checkpoint_path):
        print(f"Weights already present at {checkpoint_path}, skipping download.")
        return

    print(f"Downloading checkpoint '{filename}' from Hugging Face Hub: {repo_id}")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    print(f"Weights downloaded and verified successfully: {downloaded_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Checkpoint Downloader")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    run_download(args.config)
