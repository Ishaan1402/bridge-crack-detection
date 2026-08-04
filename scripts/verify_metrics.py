"""
Independent verification harness for crack-seg checkpoints.

Loads a checkpoint, runs the published UNet architecture (auto-detecting the
narrow [32,64,128,256] vs wide [64,128,256,512] variants and the older
``up.*`` / ``final.*`` key naming used by the HPO checkpoints), and reports
global (micro) and per-image (macro) Dice / IoU / Recall / Precision / F1
across several thresholds.

Usage:
    .venv/bin/python scripts/verify_metrics.py \
        --checkpoint checkpoints/best_unet.pth \
        --images input/DeepCrack/test_img \
        --masks input/DeepCrack/test_lab \
        --mode direct --thresholds 0.3 0.4 0.5 0.6 0.7
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

from src.config.schema import SystemSettings
from src.inference.sliding_window import IMAGENET_MEAN, IMAGENET_STD, SlidingWindowPredictor
from src.models.checkpoint import load_unet_checkpoint


MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
STD = np.array(IMAGENET_STD, dtype=np.float32)


def predict_direct(model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Single full-image forward pass (fully-convolutional model)."""
    norm = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def predict_sliding(model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device,
                    settings: SystemSettings, patch_size: int = 448, overlap: float = 0.5) -> np.ndarray:
    """Sliding-window inference with Gaussian blending (matches production)."""
    inference = settings.inference.copy(update={"patch_size": patch_size, "overlap": overlap})
    run_settings = settings.copy(update={"inference": inference})
    predictor = SlidingWindowPredictor(model, run_settings, device)
    probs, _, _ = predictor.predict_large_image(image_rgb)
    return probs


def metrics_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-6) -> dict:
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {"dice": dice, "iou": iou, "recall": recall, "precision": precision, "f1": dice}


def evaluate(image_paths, mask_paths, model, device, mode: str, thresholds, settings, patch_size=448, overlap=0.5):
    results = {t: {"g_tp": 0, "g_fp": 0, "g_fn": 0, "per_image": []} for t in thresholds}
    gt_pos = []  # positive-pixel count per image, for the "crack-only" macro variant

    for img_path, msk_path in zip(image_paths, mask_paths):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")
        gt_bin = (gt > 127).astype(np.uint8)
        gt_pos.append(int(gt_bin.sum()))

        if mode == "direct":
            probs = predict_direct(model, image_rgb, device)
        else:
            probs = predict_sliding(model, image_rgb, device, settings, patch_size, overlap)

        gt_flat = gt_bin.ravel().astype(np.int64)
        for t in thresholds:
            pred = (probs > t).astype(np.uint8)
            pred_flat = pred.ravel().astype(np.int64)
            tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
            fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
            fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
            res = results[t]
            res["g_tp"] += tp
            res["g_fp"] += fp
            res["g_fn"] += fn
            res["per_image"].append((tp, fp, fn))

    print(f"\n=== {mode.upper()} inference | {len(image_paths)} images | patch={patch_size} overlap={overlap} ===")
    header = (f"{'thr':>5} | {'Dice':>6} {'IoU':>6} {'Recall':>7} {'Prec':>6} {'F1':>6}   |"
              f" {'mDice':>6} {'mIoU':>6} {'mRec':>7} {'mPrec':>6} {'mF1':>6}   |"
              f" {'mDice*':>6} {'mF1*':>6}")
    print(header)
    print("-" * len(header))
    for t in thresholds:
        r = results[t]
        g = metrics_from_counts(r["g_tp"], r["g_fp"], r["g_fn"])
        pi = [metrics_from_counts(*c) for c in r["per_image"]]
        macro = {k: float(np.mean([m[k] for m in pi])) for k in ("dice", "iou", "recall", "precision", "f1")}
        # Macro over images that contain cracks only (exclude all-negative targets)
        cracky = [m for m, pos in zip(pi, gt_pos) if pos > 0]
        macro_star = {k: float(np.mean([m[k] for m in cracky])) for k in ("dice", "f1")} if cracky else {"dice": float("nan"), "f1": float("nan")}
        print(f"{t:5.2f} | {g['dice']:6.3f} {g['iou']:6.3f} {g['recall']:7.3f} {g['precision']:6.3f} {g['f1']:6.3f}   |"
              f" {macro['dice']:6.3f} {macro['iou']:6.3f} {macro['recall']:7.3f} {macro['precision']:6.3f} {macro['f1']:6.3f}   |"
              f" {macro_star['dice']:6.3f} {macro_star['f1']:6.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_unet.pth")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--images", default="input/DeepCrack/test_img")
    ap.add_argument("--masks", default="input/DeepCrack/test_lab")
    ap.add_argument("--mode", choices=["direct", "sliding", "both"], default="both")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--patch-size", type=int, default=448)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N images (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = SystemSettings.load_from_yaml(args.config)
    model, features = load_unet_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.checkpoint} | features={features} "
          f"| params={n_params/1e6:.2f}M | device={device}")

    image_paths = sorted(glob.glob(os.path.join(args.images, "*")))
    mask_paths = sorted(glob.glob(os.path.join(args.masks, "*")))
    if not image_paths or not mask_paths:
        raise SystemExit("No images/masks found.")
    if args.limit:
        image_paths, mask_paths = image_paths[:args.limit], mask_paths[:args.limit]
    print(f"Found {len(image_paths)} image/mask pairs.")

    t0 = time.time()
    if args.mode in ("direct", "both"):
        evaluate(image_paths, mask_paths, model, device, "direct", args.thresholds, settings)
    if args.mode in ("sliding", "both"):
        evaluate(image_paths, mask_paths, model, device, "sliding", args.thresholds, settings,
                 args.patch_size, args.overlap)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
