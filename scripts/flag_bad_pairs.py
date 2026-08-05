"""
Flag likely image/mask misalignments in an eval set.

Signature: the model predicts a non-trivial crack region (pred_area above a
floor) but has near-zero overlap with the ground-truth mask (dice below a
ceiling). That combination is the hallmark of a wrong pairing (the GT belongs
to a different image/tile) rather than a model failure.

No auto-fix: pairs are only flagged and excluded from the reported metrics.
The known-bad filenames are checked as a sanity test that the heuristic finds
the cases confirmed by visual inspection.

Example:
    PYTHONPATH=. .venv/bin/python scripts/flag_bad_pairs.py \
        --checkpoint checkpoints/unet_v3_narrow.pth \
        --images input/dataset_split/test/images \
        --masks input/dataset_split/test/masks \
        --panels reports/flagged
"""

from __future__ import annotations

import argparse
import collections
import os

import cv2
import numpy as np
import torch

from src.config.schema import SystemSettings
from src.models.checkpoint import load_unet_checkpoint


KNOWN_BAD = [
    "merged11k_CRACK500_20160326_150319_1081_641",
    "merged11k_CRACK500_20160326_150319_361_641",
    "merged11k_DeepCrack_IMG6-1",
]


def source_of(stem: str) -> str:
    parts = stem.split("_")
    if parts and parts[0] == "merged11k" and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else stem


def _src_stats(pairs: list[dict]) -> dict:
    by_src = collections.defaultdict(list)
    for p in pairs:
        by_src[p["source"]].append(p)
    out = {}
    for src, rows in sorted(by_src.items()):
        dices = [r["dice"] for r in rows]
        out[src] = {
            "n": len(rows),
            "mean_dice": float(np.mean(dices)),
            "zeros": sum(1 for d in dices if d < 0.001),
        }
    return out


def flag_results(pairs: list[dict], dice_max: float = 0.01,
                 pred_min: float = 0.001, gt_min: float = 0.0) -> tuple[list[dict], list[dict]]:
    """
    Return (flagged, kept) pairs.

    A pair is flagged when the model confidently predicts a crack region but
    has ~zero overlap with the GT: dice < dice_max, pred_area >= pred_min, and
    the GT has at least gt_min content (so it's not a trivial empty-empty case).
    """
    flagged = [
        p for p in pairs
        if p["dice"] < dice_max and p["pred_area"] >= pred_min and p["gt_mean"] > gt_min
    ]
    flagged_stems = {p["stem"] for p in flagged}
    kept = [p for p in pairs if p["stem"] not in flagged_stems]
    return flagged, kept


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Flag likely image/mask misalignments")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint (default: config)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dice-max", type=float, default=0.01, help="Flag if dice below this")
    ap.add_argument("--pred-min", type=float, default=0.001, help="Flag only if predicted area >= this fraction")
    ap.add_argument("--gt-min", type=float, default=0.0, help="Flag only if GT content >= this fraction")
    ap.add_argument("--panels", default=None, help="Save original | GT | prediction panels for flagged pairs")
    ap.add_argument("--limit", type=int, default=0, help="Cap images scanned (0 = all)")
    args = ap.parse_args(argv)

    settings = SystemSettings.load_from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or settings.model.checkpoint_path
    model, _ = load_unet_checkpoint(checkpoint, device)

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).to(device)
        with torch.inference_mode():
            return torch.sigmoid(model(x)).squeeze().cpu().numpy()

    mask_map = {os.path.splitext(n)[0]: os.path.join(args.masks, n) for n in os.listdir(args.masks)}
    pairs = []
    for name in sorted(os.listdir(args.images)):
        stem = os.path.splitext(name)[0]
        if stem not in mask_map:
            continue
        if args.limit and len(pairs) >= args.limit:
            break
        rgb = cv2.cvtColor(cv2.imread(os.path.join(args.images, name)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(mask_map[stem], cv2.IMREAD_GRAYSCALE) > 127
        pred = predict(rgb) > args.threshold
        tp = int(np.sum(pred & gt)); fp = int(np.sum(pred & ~gt)); fn = int(np.sum(~pred & gt))
        eps = 1e-6
        pairs.append({
            "stem": stem,
            "source": source_of(stem),
            "gt_mean": float(gt.mean()),
            "pred_area": float(pred.mean()),
            "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
            "iou": (tp + eps) / (tp + fp + fn + eps),
        })

    flagged, kept = flag_results(pairs, args.dice_max, args.pred_min, args.gt_min)

    print(f"Scanned {len(pairs)} pairs | flagged {len(flagged)}")
    print("\nKnown-bad sanity check:")
    flagged_stems = {p["stem"] for p in flagged}
    for name in KNOWN_BAD:
        print(f"  {'PASS' if name in flagged_stems else 'FAIL'} {name}")

    print("\nFlagged (sorted by dice):")
    for p in sorted(flagged, key=lambda p: p["dice"]):
        print(f"  {p['stem']:40s} src={p['source']:10s} gt_mean={p['gt_mean']:.4f} "
              f"pred_area={p['pred_area']:.4f} dice={p['dice']:.4f} iou={p['iou']:.4f}")

    print("\nPer-source metrics (ALL pairs):")
    print(f"{'source':12s} {'n':>5s} {'meanDice':>8s} {'zeros':>6s}")
    for src, s in _src_stats(pairs).items():
        print(f"{src:12s} {s['n']:5d} {s['mean_dice']:8.4f} {s['zeros']:6d}")

    print("\nPer-source metrics (FLAGGED EXCLUDED):")
    print(f"{'source':12s} {'n':>5s} {'meanDice':>8s} {'zeros':>6s}")
    for src, s in _src_stats(kept).items():
        print(f"{src:12s} {s['n']:5d} {s['mean_dice']:8.4f} {s['zeros']:6d}")

    if args.panels:
        os.makedirs(args.panels, exist_ok=True)
        for p in flagged:
            bgr = cv2.imread(os.path.join(args.images, p["stem"] + ".jpg"))
            gt = cv2.imread(mask_map[p["stem"]], cv2.IMREAD_GRAYSCALE)
            pred = (predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) > args.threshold).astype(np.uint8) * 255
            panel = np.hstack((bgr, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)))
            cv2.imwrite(os.path.join(args.panels, p["stem"] + ".jpg"), panel)
        print(f"\nPanels saved to {args.panels}")


if __name__ == "__main__":
    main()
