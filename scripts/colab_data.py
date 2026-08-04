"""
Download + stage helpers for the v3 Colab retraining notebook (edu/train_v3.ipynb).

Every source is normalized into the project's `{split}/images|masks` layout
with a per-source tag via scripts/prepare_dataset.py, so the trainer never
sees anything other than one consistent format.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import stat
import urllib.request
import zipfile


DEEPCRACK_ZIP_URL = "https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip"
UAV11K_DRIVE_ID = "1RMf0GYXn7Mq1s9STGFG5iByavTr05SjF"
UAV_KAGGLE_DATASET = "ziya07/uav-based-crack-detection-dataset"


def unzip(zip_path: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"Unzipped {zip_path} -> {dest}")


def download_url(url: str, dest: str) -> str:
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def _find_image_mask_dirs(root: str) -> tuple[str, str]:
    """Locate the images and masks folders in an unknown dataset layout."""
    image_dirs = sorted(glob.glob(os.path.join(root, "**", "image*"), recursive=True))
    mask_dirs = sorted(glob.glob(os.path.join(root, "**", "mask*"), recursive=True))
    image_dirs = [d for d in image_dirs if os.path.isdir(d) and len(os.listdir(d)) > 10]
    mask_dirs = [d for d in mask_dirs if os.path.isdir(d) and len(os.listdir(d)) > 10]
    if not image_dirs or not mask_dirs:
        raise SystemExit(f"Could not locate images/masks folders under {root}")
    return image_dirs[0], mask_dirs[0]


def _merge_splits(raw_root: str, out: str, source: str) -> None:
    """Copy an already-split source ({split}/{images,masks}) into the merged layout."""
    for split in ("train", "val", "test"):
        src_img = os.path.join(raw_root, split, "images")
        src_msk = os.path.join(raw_root, split, "masks")
        if not (os.path.isdir(src_img) and os.path.isdir(src_msk)):
            continue
        dst_img = os.path.join(out, split, "images")
        dst_msk = os.path.join(out, split, "masks")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_msk, exist_ok=True)
        for name in sorted(os.listdir(src_img)):
            stem = f"{source}_{os.path.splitext(name)[0]}"
            shutil.copy(os.path.join(src_img, name), os.path.join(dst_img, stem + ".jpg"))
        for name in sorted(os.listdir(src_msk)):
            stem = f"{source}_{os.path.splitext(name)[0]}"
            shutil.copy(os.path.join(src_msk, name), os.path.join(dst_msk, stem + ".png"))
    print(f"Merged {source} (pre-split) into {out}")


def setup_kaggle_credentials(kaggle_json_path: str) -> bool:
    """
    Install MyDrive/kaggle.json for kagglehub.

    Returns True on success. Get the file from Kaggle -> Settings -> API ->
    Create New Token, then upload it to MyDrive.
    """
    if not os.path.exists(kaggle_json_path):
        return False
    with open(kaggle_json_path) as f:
        creds = json.load(f)
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), stat.S_IRUSR | stat.S_IWUSR)
    os.environ["KAGGLE_USERNAME"] = str(creds.get("username", ""))
    os.environ["KAGGLE_KEY"] = str(creds.get("key", ""))
    return True


def download_uav(drive_zip: str | None, data_root: str, out: str) -> str:
    """
    Stage the UAV Kaggle source.

    Primary: kagglehub download with a fresh stratified 70/15/15 split
    (Kaggle credentials required — see setup_kaggle_credentials). Optional
    override: the project's dataset_split.zip on Drive, which keeps the
    original 220/47/48 split.
    """
    if drive_zip and os.path.exists(drive_zip):
        raw = os.path.join(data_root, "uav_raw")
        unzip(drive_zip, raw)
        base = next(
            (c for c in (raw, os.path.join(raw, "dataset_split"))
             if os.path.isdir(os.path.join(c, "train"))),
            raw,
        )
        _merge_splits(base, out, "uav")
        return "drive"

    try:
        import kagglehub
        path = kagglehub.dataset_download(UAV_KAGGLE_DATASET)
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Upload kaggle.json to MyDrive and run the "
            "'Kaggle credentials' cell, or re-upload dataset_split.zip to "
            "MyDrive/bridge_crack_detection/ to keep the original split."
        ) from exc
    images, masks = _find_image_mask_dirs(path)
    stage(images, masks, out, source="uav", cap=0, resize=512,
          val_frac=0.15, test_frac=0.15, seed=42)
    return "kaggle"


def download_deepcrack(data_root: str) -> tuple[str, str, str, str]:
    """Returns (train_img, train_lab, test_img, test_lab) dirs."""
    zip_path = os.path.join(data_root, "deepcrack.zip")
    raw = os.path.join(data_root, "deepcrack_raw")
    if not os.path.exists(os.path.join(raw, "train_img")):
        download_url(DEEPCRACK_ZIP_URL, zip_path)
        unzip(zip_path, raw)
    return (
        os.path.join(raw, "train_img"),
        os.path.join(raw, "train_lab"),
        os.path.join(raw, "test_img"),
        os.path.join(raw, "test_lab"),
    )


def download_uav11k(data_root: str) -> tuple[str, str]:
    """Returns (images_dir, masks_dir) for the Auto-ROS-LAB UAV 11k dataset."""
    zip_path = os.path.join(data_root, "uav11k.zip")
    raw = os.path.join(data_root, "uav11k_raw")
    if not any(glob.glob(os.path.join(raw, "**", "image*"), recursive=True)) or not os.path.exists(raw):
        import gdown
        gdown.download(id=UAV11K_DRIVE_ID, output=zip_path, quiet=False)
        unzip(zip_path, raw)
    return _find_image_mask_dirs(raw)


def stage(images_dir: str, masks_dir: str, out: str, source: str, cap: int,
          resize: int, val_frac: float, test_frac: float, seed: int = 42) -> None:
    """Convert one source into the merged dataset_split layout via prepare_dataset."""
    from scripts.prepare_dataset import main as prepare
    prepare([
        "--images", images_dir,
        "--masks", masks_dir,
        "--out", out,
        "--source", source,
        "--cap", str(cap),
        "--resize", str(resize),
        "--val-frac", str(val_frac),
        "--test-frac", str(test_frac),
        "--seed", str(seed),
    ])
