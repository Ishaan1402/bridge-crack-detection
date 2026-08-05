"""
Evaluate multiple crack-seg checkpoints on the same eval sets and compare.

Every checkpoint is run through the same protocol (same thresholds, same
inference mode), so the numbers are directly comparable. The loader
auto-detects each checkpoint's architecture (narrow/wide, SE/deep
supervision, interpolate upsample), so existing and v3 checkpoints all work.

Default eval sets:
  - DeepCrack test (cross-domain, held out): input/DeepCrack/test_*
Extra sets (e.g. the staged test split, or your real UAV test split later):
  --extra-set NAME IMAGES_DIR MASKS_DIR   (repeatable)

Example:
    PYTHONPATH=. .venv/bin/python scripts/evaluate_models.py \
        --checkpoints checkpoints/best_unet.pth /Users/ishaan/Downloads/best_unet.pth \
        --labels "narrow-v2" "wide-v1" \
        --thresholds 0.5 \
        --report reports/eval_all.md
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config.schema import SystemSettings
from src.models.checkpoint import load_unet_checkpoint
from scripts.verify_metrics import metrics_from_counts, predict_direct, predict_sliding


DEEPCRACK_IMAGES = "input/DeepCrack/test_img"
DEEPCRACK_MASKS = "input/DeepCrack/test_lab"


def discover_checkpoints() -> list[str]:
    """Common checkpoint locations, filtered to those that exist."""
    candidates = [
        "checkpoints/best_unet.pth",                       # published narrow v2
        str(Path.home() / "Downloads" / "best_unet.pth"),  # original wide v1
        "checkpoints/unet_v3_narrow.pth",                  # v3 from Colab
        "checkpoints/unet_v3_wide.pth",
    ]
    return [p for p in candidates if os.path.exists(p)]


def default_label(path: str) -> str:
    stem = Path(path).stem
    if "Downloads" in path and stem == "best_unet":
        return "wide-v1 (original)"
    if stem == "best_unet":
        return "narrow-v2 (published)"
    return stem.replace("_", "-")


def collect_metrics(image_paths, mask_paths, model, device, mode, settings,
                    thresholds, patch_size=448, overlap=0.5) -> dict:
    """Global + per-image metrics per threshold, without printing."""
    results = {t: {"g_tp": 0, "g_fp": 0, "g_fn": 0, "per": []} for t in thresholds}
    for img_path, msk_path in zip(image_paths, mask_paths):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")
        gt_bin = gt > 127

        if mode == "direct":
            probs = predict_direct(model, image_rgb, device)
        elif mode == "sliding":
            probs = predict_sliding(model, image_rgb, device, settings, patch_size, overlap)
        else:  # both
            probs = predict_direct(model, image_rgb, device)

        for t in thresholds:
            pred = probs > t
            tp = int(np.sum(pred & gt_bin))
            fp = int(np.sum(pred & ~gt_bin))
            fn = int(np.sum(~pred & gt_bin))
            results[t]["g_tp"] += tp
            results[t]["g_fp"] += fp
            results[t]["g_fn"] += fn
            results[t]["per"].append((tp, fp, fn))

    out = {}
    for t in thresholds:
        r = results[t]
        global_m = metrics_from_counts(r["g_tp"], r["g_fp"], r["g_fn"])
        per = [metrics_from_counts(*c) for c in r["per"]]
        macro = {k: float(np.mean([m[k] for m in per])) for k in ("dice", "iou", "recall", "precision", "f1")}
        out[t] = {"global": global_m, "macro": macro}
    return out


def _fmt_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append(" | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare crack-seg checkpoints on eval sets")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="Checkpoint paths (default: discover existing known ones)")
    ap.add_argument("--labels", nargs="+", default=None, help="Display labels, one per checkpoint")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    ap.add_argument("--mode", choices=["direct", "sliding", "both"], default="direct")
    ap.add_argument("--deepcrack-images", default=DEEPCRACK_IMAGES)
    ap.add_argument("--deepcrack-masks", default=DEEPCRACK_MASKS)
    ap.add_argument("--extra-set", action="append", nargs=3, metavar=("NAME", "IMAGES", "MASKS"),
                    help="Additional eval set, repeatable (e.g. staged or real UAV test split)")
    ap.add_argument("--report", default=None, help="Write a markdown report to this path")
    ap.add_argument("--limit", type=int, default=0, help="Cap images per set (0 = all)")
    args = ap.parse_args(argv)

    checkpoints = args.checkpoints or discover_checkpoints()
    if not checkpoints:
        raise SystemExit("No checkpoints found. Pass --checkpoints explicitly.")
    labels = args.labels or [default_label(c) for c in checkpoints]
    if len(labels) != len(checkpoints):
        raise SystemExit("--labels must match --checkpoints count")

    sets = [("deepcrack", args.deepcrack_images, args.deepcrack_masks)]
    if args.extra_set:
        sets += [(name, im, ms) for name, im, ms in args.extra_set]

    settings = SystemSettings.load_from_yaml("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for ckpt, label in zip(checkpoints, labels):
        if not os.path.exists(ckpt):
            print(f"SKIP {label}: {ckpt} not found")
            continue
        model, features = load_unet_checkpoint(ckpt, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        models[label] = (ckpt, model)
        print(f"Loaded {label}: {ckpt} (features={features}, {n_params:.1f}M)")

    if not models:
        raise SystemExit("No loadable checkpoints.")

    report_lines = ["# Crack-seg model comparison\n"]

    for set_name, im_dir, ms_dir in sets:
        if not (os.path.isdir(im_dir) and os.path.isdir(ms_dir)):
            print(f"SKIP eval set {set_name}: missing {im_dir} or {ms_dir}")
            continue
        images = sorted(os.listdir(im_dir))
        masks = sorted(os.listdir(ms_dir))
        if args.limit:
            images, masks = images[: args.limit], masks[: args.limit]
        image_paths = [os.path.join(im_dir, n) for n in images]
        mask_paths = [os.path.join(ms_dir, n) for n in masks]
        print(f"\n=== Eval set: {set_name} ({len(image_paths)} images, mode={args.mode}) ===")
        report_lines.append(f"## Eval set: {set_name} ({len(image_paths)} images, mode={args.mode})\n")

        for thr in args.thresholds:
            rows = [["model", "Dice", "IoU", "Recall", "Prec", "F1", "mDice"]]
            for label, (ckpt, model) in models.items():
                m = collect_metrics(image_paths, mask_paths, model, device, args.mode,
                                    settings, [thr])
                g = m[thr]["global"]
                rows.append([
                    label,
                    f"{g['dice']:.4f}", f"{g['iou']:.4f}", f"{g['recall']:.4f}",
                    f"{g['precision']:.4f}", f"{g['f1']:.4f}", f"{m[thr]['macro']['dice']:.4f}",
                ])
            table = _fmt_table(rows)
            print(f"\nthreshold={thr}\n{table}")
            report_lines.append(f"### threshold={thr}\n\n{table}\n")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(report_lines))
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
