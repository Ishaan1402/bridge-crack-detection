import os

import cv2
import numpy as np
import torch

from scripts import flag_bad_pairs
from src.models.unet import UNet


def _pair(stem, dice, pred_area, gt_mean, source=None):
    return {
        "stem": stem,
        "source": source or flag_bad_pairs.source_of(stem),
        "gt_mean": gt_mean,
        "pred_area": pred_area,
        "dice": dice,
        "iou": dice / (2 - dice) if dice < 1 else 1.0,
    }


def test_flag_results_known_bad_and_clean():
    pairs = [
        _pair("merged11k_CRACK500_20160326_150319_1081_641", dice=0.0, pred_area=0.05, gt_mean=0.04),
        _pair("merged11k_CRACK500_20160326_150319_361_641", dice=0.0, pred_area=0.03, gt_mean=0.05),
        _pair("merged11k_DeepCrack_IMG6-1", dice=0.0, pred_area=0.02, gt_mean=0.04),
        _pair("uav_something", dice=0.8, pred_area=0.03, gt_mean=0.02),      # healthy
        _pair("merged11k_CFD_x", dice=0.0, pred_area=0.00005, gt_mean=0.04),  # pred too small -> keep
        _pair("merged11k_GAPS384_y", dice=0.0, pred_area=0.04, gt_mean=0.0), # empty GT -> keep
    ]

    flagged, kept = flag_bad_pairs.flag_results(pairs)
    flagged_stems = {p["stem"] for p in flagged}

    for name in flag_bad_pairs.KNOWN_BAD:
        assert name in flagged_stems
    assert "uav_something" not in flagged_stems
    assert "merged11k_CFD_x" not in flagged_stems
    assert "merged11k_GAPS384_y" not in flagged_stems
    assert len(kept) == 3


def test_low_prediction_floor_still_flags_known_bad():
    """The known-bad DeepCrack case predicts only a small region; a lower
    floor must still catch it (dice=0 + any non-trivial prediction + GT content)."""
    pairs = [
        _pair("merged11k_DeepCrack_IMG6-1", dice=0.0, pred_area=0.0003, gt_mean=0.04),
        _pair("uav_clean", dice=0.8, pred_area=0.0003, gt_mean=0.02),
    ]
    flagged, _ = flag_bad_pairs.flag_results(pairs)
    assert "merged11k_DeepCrack_IMG6-1" in {p["stem"] for p in flagged}


def test_src_stats_with_and_without_flags():
    pairs = [
        _pair("merged11k_CRACK500_a", dice=0.0, pred_area=0.05, gt_mean=0.04),
        _pair("merged11k_CRACK500_b", dice=0.7, pred_area=0.03, gt_mean=0.02),
        _pair("uav_x", dice=0.8, pred_area=0.02, gt_mean=0.02),
    ]
    flagged, kept = flag_bad_pairs.flag_results(pairs)
    all_stats = flag_bad_pairs._src_stats(pairs)
    kept_stats = flag_bad_pairs._src_stats(kept)

    assert all_stats["CRACK500"]["n"] == 2
    assert all_stats["CRACK500"]["zeros"] == 1
    assert kept_stats["CRACK500"]["n"] == 1
    assert kept_stats["CRACK500"]["zeros"] == 0
    assert kept_stats["uav"]["n"] == 1


def test_main_smoke(tmp_path):
    img_dir = tmp_path / "imgs"
    msk_dir = tmp_path / "msks"
    img_dir.mkdir()
    msk_dir.mkdir()
    for i in range(6):
        img = np.full((64, 64, 3), 180, dtype=np.uint8)
        img[32, :, :] = 30
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[32, :] = 255
        cv2.imwrite(str(img_dir / f"uav_x{i}.jpg"), img)
        cv2.imwrite(str(msk_dir / f"uav_x{i}.png"), msk)

    ckpt = tmp_path / "tiny.pth"
    torch.save(UNet(features=[8, 16]).state_dict(), ckpt)
    panels = tmp_path / "panels"

    flag_bad_pairs.main([
        "--checkpoint", str(ckpt),
        "--images", str(img_dir),
        "--masks", str(msk_dir),
        "--panels", str(panels),
    ])

    assert os.path.isdir(panels)
