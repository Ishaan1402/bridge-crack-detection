"""
Convert an external crack dataset (paired images + masks) into the project layout:

    {out}/{split}/images/*.jpg
    {out}/{split}/masks/*.png    (binary 0/255)

This is exactly the layout the Colab notebook and scripts/train.py expect, so
any source can be dropped into the existing training scheme.

Examples:
    # Binary grayscale masks (DeepCrack, Crack500, CFD, NCCD-PF, ...)
    PYTHONPATH=. .venv/bin/python scripts/prepare_dataset.py \
        --images path/Images --masks path/Masks --out input/dataset_split \
        --resize 448 --test-frac 0.2

    # Multi-class label masks (e.g. GAPs: keep only the crack classes)
    PYTHONPATH=. .venv/bin/python scripts/prepare_dataset.py \
        --images path/Images --masks path/Labels --out input/dataset_split \
        --classes 1 2 3 5 --resize 448
"""

from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np


def _pair_files(images_dir: str, masks_dir: str):
    image_paths = sorted(os.listdir(images_dir))
    mask_paths = sorted(os.listdir(masks_dir))

    img_map = {os.path.splitext(p)[0]: os.path.join(images_dir, p) for p in image_paths}
    pairs = []
    for m in mask_paths:
        stem = os.path.splitext(m)[0]
        if stem in img_map:
            pairs.append((img_map[stem], os.path.join(masks_dir, m)))
    if not pairs:
        raise SystemExit(
            f"No image/mask pairs found. Check that filenames share stems: {images_dir} / {masks_dir}"
        )
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Convert external crack data to the project layout")
    ap.add_argument("--images", required=True, help="Directory of RGB images")
    ap.add_argument("--masks", required=True, help="Directory of masks/labels")
    ap.add_argument("--out", required=True, help="Output dir with train/val/test splits")
    ap.add_argument("--resize", type=int, default=0, help="Square resize (0 = keep native size)")
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="For multi-class label masks: which class IDs count as crack")
    ap.add_argument("--source", default="source", help="Source tag: prefixes output filenames and records manifest stats")
    ap.add_argument("--cap", type=int, default=0, help="Max pairs to convert from this source (0 = all)")
    ap.add_argument("--drop-prefix", default=None,
                    help="Skip pairs whose image stem starts with this (e.g. noncrack)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0, help="Cap on pairs to process (0 = all)")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    pairs = _pair_files(args.images, args.masks)
    random.shuffle(pairs)
    if args.drop_prefix:
        pairs = [
            p for p in pairs
            if not os.path.splitext(os.path.basename(p[0]))[0].lower().startswith(args.drop_prefix.lower())
        ]
    if args.max_images:
        pairs = pairs[: args.max_images]
    if args.cap:
        pairs = pairs[: args.cap]

    n_test = int(len(pairs) * args.test_frac)
    n_val = int(len(pairs) * args.val_frac)
    splits = {
        "test": pairs[:n_test],
        "val": pairs[n_test:n_test + n_val],
        "train": pairs[n_test + n_val:],
    }

    for split, split_pairs in splits.items():
        img_dir = os.path.join(args.out, split, "images")
        msk_dir = os.path.join(args.out, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for img_path, msk_path in split_pairs:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  skip (unreadable image): {img_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  skip (unreadable mask): {msk_path}")
                continue

            # Binarize: either explicit class IDs or the standard >127 threshold
            if args.classes:
                binary = np.isin(mask, args.classes).astype(np.uint8) * 255
            else:
                binary = (mask > 127).astype(np.uint8) * 255

            if args.resize:
                image_rgb = cv2.resize(image_rgb, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
                binary = cv2.resize(binary, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)

            stem = f"{args.source}_{os.path.splitext(os.path.basename(img_path))[0]}"
            cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(os.path.join(msk_dir, f"{stem}.png"), binary)

    # Update the per-source manifest with counts and mean crack ratio
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    crack_ratios = []
    for split in ("train", "val", "test"):
        mask_dir = os.path.join(args.out, split, "masks")
        if not os.path.isdir(mask_dir):
            continue
        for name in os.listdir(mask_dir):
            mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                crack_ratios.append(float(np.mean(mask > 127)))
    manifest[args.source] = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "mean_crack_ratio": float(np.mean(crack_ratios)) if crack_ratios else 0.0,
        "resize": args.resize,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for split, split_pairs in splits.items():
        print(f"{split:6s}: {len(split_pairs)} pairs")
    print(f"Source '{args.source}' registered in {manifest_path}")
    print(f"Mean crack ratio: {manifest[args.source]['mean_crack_ratio']:.4f}")
    print(f"Output written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
