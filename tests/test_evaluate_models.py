import os
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts import evaluate_models
from src.models.unet import UNet


def _synthetic_data(img_dir, msk_dir, n=6, size=64):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    for i in range(n):
        img = np.full((size, size, 3), 180, dtype=np.uint8)
        img[size // 2, :, :] = 30
        msk = np.zeros((size, size), dtype=np.uint8)
        msk[size // 2, :] = 255
        cv2.imwrite(os.path.join(img_dir, f"i{i}.jpg"), img)
        cv2.imwrite(os.path.join(msk_dir, f"i{i}.png"), msk)


def test_evaluate_models_comparison(tmp_path):
    img_dir = str(tmp_path / "imgs")
    msk_dir = str(tmp_path / "msks")
    _synthetic_data(img_dir, msk_dir)

    ck1 = tmp_path / "tiny_a.pth"
    ck2 = tmp_path / "tiny_b.pth"
    torch.save(UNet(features=[8, 16]).state_dict(), ck1)
    torch.save(UNet(features=[16, 32], se=True).state_dict(), ck2)

    report = tmp_path / "report.md"
    evaluate_models.main([
        "--checkpoints", str(ck1), str(ck2),
        "--labels", "tiny-a", "tiny-b",
        "--deepcrack-images", img_dir,
        "--deepcrack-masks", msk_dir,
        "--thresholds", "0.5",
        "--report", str(report),
    ])

    assert report.exists()
    text = report.read_text()
    assert "tiny-a" in text and "tiny-b" in text
    assert "Dice" in text


def test_discover_checkpoints_filters_missing(tmp_path, monkeypatch):
    dl = tmp_path / "Downloads"
    dl.mkdir()
    (dl / "best_unet.pth").write_bytes(b"x")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    found = evaluate_models.discover_checkpoints()
    assert str(dl / "best_unet.pth") in found
    assert "checkpoints/best_unet.pth" not in found
