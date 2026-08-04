import os

import cv2
import numpy as np

from src.models.checkpoint import load_unet_checkpoint
from scripts.train import resolve_training_preset
from scripts import prepare_dataset, train


def _synthetic_pair(img_dir, msk_dir, stem, size=64, crack_row=32):
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    img[crack_row, :, :] = 40
    msk = np.zeros((size, size), dtype=np.uint8)
    msk[crack_row, :] = 255
    cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), img)
    cv2.imwrite(os.path.join(msk_dir, f"{stem}.png"), msk)


def test_train_smoke(tmp_path):
    """End-to-end: tiny synthetic dataset -> train one epoch -> servable checkpoint."""
    data_dir = tmp_path / "data"
    for split in ("train", "val"):
        img_dir = data_dir / split / "images"
        msk_dir = data_dir / split / "masks"
        img_dir.mkdir(parents=True)
        msk_dir.mkdir(parents=True)
    for i in range(4):
        _synthetic_pair(str(data_dir / "train" / "images"), str(data_dir / "train" / "masks"), f"t{i}")
    for i in range(2):
        _synthetic_pair(str(data_dir / "val" / "images"), str(data_dir / "val" / "masks"), f"v{i}")

    out = tmp_path / "smoke.pth"
    train.main([
        "--data-dir", str(data_dir), "--out", str(out),
        "--epochs", "1", "--batch-size", "2", "--lr", "1e-3",
        "--resize", "64", "--features", "8,16", "--seed", "0",
        "--amp",
    ])

    assert out.exists()
    model, features = load_unet_checkpoint(str(out), "cpu")
    assert features == [8, 16]
    assert model.encoder[0].conv[0].out_channels == 8


def test_training_preset_by_gpu():
    """GPU-class presets: T4/L4 low batch, A100/V100 high batch, 512px both."""
    assert resolve_training_preset([32, 64, 128, 256], "Tesla T4") == (512, 8)
    assert resolve_training_preset([64, 128, 256, 512], "Tesla T4") == (512, 4)
    assert resolve_training_preset([32, 64, 128, 256], "NVIDIA A100-SXM4-40GB") == (512, 16)
    assert resolve_training_preset([64, 128, 256, 512], "NVIDIA A100-SXM4-40GB") == (512, 8)


def test_prepare_dataset_layout_and_binarization(tmp_path):
    src_imgs = tmp_path / "src_images"
    src_msks = tmp_path / "src_masks"
    src_imgs.mkdir()
    src_msks.mkdir()
    for i in range(4):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(src_imgs / f"p{i}.jpg"), img)
        msk = np.zeros((32, 32), dtype=np.uint8)
        msk[10, 10] = 1
        msk[20, 20] = 3
        msk[30, 30] = 255
        cv2.imwrite(str(src_msks / f"p{i}.png"), msk)

    out = tmp_path / "converted"
    prepare_dataset.main([
        "--images", str(src_imgs), "--masks", str(src_msks), "--out", str(out),
        "--val-frac", "0.25", "--test-frac", "0.25", "--classes", "1", "3", "--seed", "0",
    ])

    assert len(os.listdir(out / "train" / "images")) == 2
    assert len(os.listdir(out / "val" / "images")) == 1
    assert len(os.listdir(out / "test" / "images")) == 1
    for split in ("train", "val", "test"):
        for mask_path in os.listdir(out / split / "masks"):
            mask = cv2.imread(str(out / split / "masks" / mask_path), cv2.IMREAD_GRAYSCALE)
            assert set(np.unique(mask)) <= {0, 255}


def test_prepare_dataset_source_cap_manifest(tmp_path):
    """--source prefixes filenames, --cap bounds pairs, manifest records stats."""
    src_imgs = tmp_path / "imgs"
    src_msks = tmp_path / "msks"
    src_imgs.mkdir()
    src_msks.mkdir()
    for i in range(6):
        cv2.imwrite(str(src_imgs / f"f{i}.jpg"), np.full((32, 32, 3), 120, dtype=np.uint8))
        msk = np.zeros((32, 32), dtype=np.uint8)
        msk[5:8, :] = 255
        cv2.imwrite(str(src_msks / f"f{i}.png"), msk)
    # a no-crack file that should be dropped by --drop-prefix
    cv2.imwrite(str(src_imgs / "noncrack_a.jpg"), np.full((32, 32, 3), 120, dtype=np.uint8))
    cv2.imwrite(str(src_msks / "noncrack_a.png"), np.zeros((32, 32), dtype=np.uint8))

    out = tmp_path / "capped"
    prepare_dataset.main([
        "--images", str(src_imgs), "--masks", str(src_msks), "--out", str(out),
        "--source", "dc", "--cap", "4", "--val-frac", "0.25", "--test-frac", "0.0",
        "--seed", "0", "--drop-prefix", "noncrack",
    ])

    train_files = sorted(os.listdir(out / "train" / "images"))
    assert len(train_files) == 3
    assert all(name.startswith("dc_") for name in train_files)
    assert not any("noncrack" in name for name in train_files)

    import json
    with open(out / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["dc"]["train"] == 3
    assert manifest["dc"]["val"] == 1
    assert manifest["dc"]["mean_crack_ratio"] > 0.09
