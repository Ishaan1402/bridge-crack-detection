import os
import argparse
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from src.config.schema import SystemSettings
from src.models.unet import UNet
from src.inference.sliding_window import SlidingWindowPredictor

def run_sweep(args):
    """
    Offline benchmarking utility to sweep over confidence threshold values,
    evaluating performance on structural ground truth masks, and drawing P-R profiles.
    """
    # 1. Load System Settings
    settings = SystemSettings.load_from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Instantiate and compile model
    model = UNet(
        in_channels=settings.model.in_channels,
        out_channels=settings.model.out_channels,
        features=settings.model.features
    )
    checkpoint_path = args.checkpoint if args.checkpoint else settings.model.checkpoint_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Weights not found at current location: {checkpoint_path}. "
            "Suggest running `python scripts/download_checkpoint.py` first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictor = SlidingWindowPredictor(model, settings, device)

    # 3. Gather annotation files (e.g. under split data partitions)
    img_paths = sorted(glob.glob(os.path.join(args.data_dir, "**", "images", "*.*"), recursive=True))
    mask_paths = sorted(glob.glob(os.path.join(args.data_dir, "**", "masks", "*.*"), recursive=True))

    # Match images and masks only (filtering directories)
    img_paths = [p for p in img_paths if os.path.isfile(p)]
    mask_paths = [p for p in mask_paths if os.path.isfile(p)]

    if not img_paths:
         # Try matching raw data directories directly if nested matching gets zero elements
         img_paths = sorted(glob.glob(os.path.join(args.data_dir, "images", "*.*")))
         mask_paths = sorted(glob.glob(os.path.join(args.data_dir, "masks", "*.*")))

    if not img_paths:
        raise ValueError(f"No validation/test dataset couples found under path: {args.data_dir}")

    # Ensure equal pairings
    assert len(img_paths) == len(mask_paths), f"Mismatch count: {len(img_paths)} images vs {len(mask_paths)} annotations."

    thresholds = np.arange(0.1, 1.0, 0.1)
    precisions, recalls, f1_scores, ious = [], [], [], []

    print(f"Loaded {len(img_paths)} dataset image/mask pairs. Running offline sweep over: {thresholds.tolist()}")

    # Iterate threshold limits
    for t in thresholds:
        tp_total, fp_total, fn_total, union_total = 0, 0, 0, 0

        for img_p, mask_p in zip(img_paths, mask_paths):
            img = cv2.imread(img_p)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            _, gt_binary = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

            _, pred_binary, _ = predictor.predict_large_image(img_rgb, threshold=t)

            # Accumulate spatial pixel detections
            tp_total += np.sum((pred_binary == 1) & (gt_binary == 1))
            fp_total += np.sum((pred_binary == 1) & (gt_binary == 0))
            fn_total += np.sum((pred_binary == 0) & (gt_binary == 1))
            union_total += np.sum((pred_binary == 1) | (gt_binary == 1))

        # Precision, Recall, Dice (F1), and IoU
        p = tp_total / (tp_total + fp_total + 1e-8)
        r = tp_total / (tp_total + fn_total + 1e-8)
        f1 = (2 * p * r) / (p + r + 1e-8)
        iou = tp_total / (union_total + 1e-8)

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        ious.append(iou)

        print(f"Threshold: {t:.2f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

    # Generate output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot Precision-Recall Profile output
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', color='purple', linewidth=2, label='U-Net Sweep')
    for i, t in enumerate(thresholds):
        plt.annotate(f"{t:.1f}", (recalls[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Sliding Window Inference)')
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(args.output_dir, "pr_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved P-R Curve figure plotting layout to: {out_path}")

    # Compute optimal metric operating indices
    best_idx = np.argmax(f1_scores)
    print("=" * 60)
    print(f"Optimal operational threshold yielding highest Dice F1:")
    print(f"Threshold:  {thresholds[best_idx]:.2f}")
    print(f"Precision:  {precisions[best_idx]:.4f}")
    print(f"Recall:     {recalls[best_idx]:.4f} (Safety-Imbalance Coverage)")
    print(f"F1 Score:   {f1_scores[best_idx]:.4f}")
    print(f"IoU Score:  {ious[best_idx]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision-Recall Threshold Evaluation Sweeper")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Overriding path to UNet checkpoints file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data folder root holding images & masks splits")
    parser.add_argument("--output_dir", type=str, default="reports", help="Reports destination folder")
    args = parser.parse_args()

    run_sweep(args)
