"""
Train the crack-seg U-Net with the project's standard scheme.

Data layout (identical to the Colab notebook / dataset_split.zip):
    {data_dir}/train/images/*   {data_dir}/train/masks/*
    {data_dir}/val/images/*     {data_dir}/val/masks/*
    {data_dir}/test/images/*    {data_dir}/test/masks/*   (optional)

Example:
    PYTHONPATH=. .venv/bin/python scripts/train.py \
        --data-dir input/dataset_split --epochs 25 --batch-size 8 \
        --lr 1e-4 --resize 448 --se --dropout 0.1 --deep-supervision \
        --features 32,64,128,256 --out checkpoints/unet_v3.pth
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A

from src.dataset.crack_dataset import CrackDataset, get_train_transform, get_val_test_transform
from src.models.losses import BCEDiceLoss
from src.models.unet import UNet
from src.utils.device import select_device


def _mask_permute(masks: torch.Tensor) -> torch.Tensor:
    """Normalize mask layout to (B, 1, H, W)."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks.permute(0, 3, 1, 2)
    return masks.contiguous().float()


def _metrics(probs: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (probs > threshold)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    eps = 1e-6
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Global + per-image (macro) metrics at threshold 0.5."""
    model.eval()
    g = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    per_image = []
    with torch.inference_mode():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = _mask_permute(masks).to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            gt = masks.cpu().numpy()[:, 0] > 0.5
            for p, g_ in zip(probs, gt):
                m = _metrics(p, g_)
                per_image.append(m)
                pred = p > 0.5
                g["tp"] += float(np.sum(pred & g_))
                g["fp"] += float(np.sum(pred & ~g_))
                g["fn"] += float(np.sum(~pred & g_))
    global_m = {
        "dice": (2 * g["tp"] + 1e-6) / (2 * g["tp"] + g["fp"] + g["fn"] + 1e-6),
        "iou": (g["tp"] + 1e-6) / (g["tp"] + g["fp"] + g["fn"] + 1e-6),
        "recall": (g["tp"] + 1e-6) / (g["tp"] + g["fn"] + 1e-6),
        "precision": (g["tp"] + 1e-6) / (g["tp"] + g["fp"] + 1e-6),
    }
    macro = {k: float(np.mean([m[k] for m in per_image])) for k in ("dice", "iou", "recall", "precision")}
    return {"global": global_m, "macro": macro}


def _build_loaders(data_dir: str, batch_size: int, resize: int, shuffle_val: bool = False):
    def pairs(split: str):
        base = os.path.join(data_dir, split)
        imgs = sorted(glob.glob(os.path.join(base, "images", "*")))
        masks = sorted(glob.glob(os.path.join(base, "masks", "*")))
        return imgs, masks

    train_imgs, train_masks = pairs("train")
    val_imgs, val_masks = pairs("val")
    if not train_imgs or not val_imgs:
        raise FileNotFoundError(
            f"Expected {data_dir}/train and {data_dir}/val with images/ and masks/ subfolders."
        )

    train_transform = get_train_transform()
    val_transform = get_val_test_transform()
    if resize:
        train_transform = A.Compose([A.Resize(resize, resize)] + train_transform.transforms)
        val_transform = A.Compose([A.Resize(resize, resize)] + val_transform.transforms)

    train_ds = CrackDataset(train_imgs, train_masks, train_transform)
    val_ds = CrackDataset(val_imgs, val_masks, val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle_val, num_workers=0)
    return train_loader, val_loader


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train crack-seg U-Net")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="Checkpoint output path (.pth)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=0, help="Square resize for both images and masks (0 = native)")
    ap.add_argument("--features", type=str, default="32,64,128,256", help="Encoder widths, comma-separated")
    ap.add_argument("--se", action="store_true", help="Squeeze-and-excitation blocks")
    ap.add_argument("--dropout", type=float, default=0.0, help="Bottleneck Dropout2d rate")
    ap.add_argument("--deep-supervision", action="store_true", help="Auxiliary decoder heads during training")
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None, help="Optional pretrained state dict to start from")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device()
    features = [int(x) for x in args.features.split(",")]
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=features,
        se=args.se,
        dropout=args.dropout,
        deep_supervision=args.deep_supervision,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        print(f"Loaded pretrained weights from {args.checkpoint}")

    criterion = BCEDiceLoss(bce_weight=args.bce_weight, aux_weight=args.aux_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = _build_loaders(args.data_dir, args.batch_size, args.resize)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | features={features} | se={args.se} | dropout={args.dropout} "
          f"| deep_supervision={args.deep_supervision} | params={n_params/1e6:.2f}M")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = _mask_permute(masks).to(device)
            optimizer.zero_grad()
            if args.deep_supervision:
                logits, aux = model.forward_deep(imgs)
                loss = criterion(logits, masks, aux)
            else:
                loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        macro_dice = val_metrics["macro"]["dice"]
        g = val_metrics["global"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss/len(train_loader):.4f} "
            f"| val Dice={macro_dice:.4f} (macro) {g['dice']:.4f} (global) "
            f"| IoU={g['iou']:.4f} | Rec={g['recall']:.4f} | Prec={g['precision']:.4f} "
            f"| {time.time()-t0:.1f}s"
        )

        if macro_dice > best_dice:
            best_dice = macro_dice
            torch.save(model.state_dict(), args.out)
            print(f"  -> new best, saved to {args.out}")

    print(f"Done. Best val macro Dice: {best_dice:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
