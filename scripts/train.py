"""
Train the crack-seg U-Net with the project's standard scheme.

Data layout (identical to the Colab notebook / dataset_split.zip):
    {data_dir}/train/images/*   {data_dir}/train/masks/*
    {data_dir}/val/images/*     {data_dir}/val/masks/*
    {data_dir}/test/images/*    {data_dir}/test/masks/*   (optional)

Example (v3 run):
    PYTHONPATH=. .venv/bin/python scripts/train.py \
        --data-dir input/dataset_split --out checkpoints/unet_v3.pth \
        --epochs 30 --lr 1e-3 --resize 512 --features 32,64,128,256 \
        --se --dropout 0.1 --deep-supervision --aug strong --amp
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset.crack_dataset import CrackDataset, get_train_transform, get_val_test_transform
from src.models.losses import BCEDiceLoss
from src.models.unet import UNet
from src.utils.device import select_device


HIGH_GPU_MARKERS = ("A100", "V100", "H100", "A10G", "L40", "RTX 4090", "RTX 3090", "A6000")


def resolve_training_preset(features: list[int], gpu_name: str = "") -> tuple[int, int]:
    """
    Pick (resolution, batch_size) from the GPU model.

    T4/L4-class GPUs: 512px, batch 8 (narrow) / 4 (wide).
    A100/V100-class GPUs: 512px, batch 16 (narrow) / 8 (wide).
    """
    wide = features[0] > 32
    high = any(marker in gpu_name.upper() for marker in HIGH_GPU_MARKERS)
    batch = (16 if not wide else 8) if high else (8 if not wide else 4)
    return 512, batch


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


def _build_augmentation(aug: str, resize: int):
    """Training transforms. 'strong' adds rotation/scale/shift, noise, elastic."""
    if aug == "strong":
        # albumentations 2.x: GaussNoise uses std_range, normalized to [0, 1]
        try:
            gauss_noise = A.GaussNoise(std_range=(0.01, 0.03), p=0.3)
        except (ValueError, TypeError):
            gauss_noise = A.GaussNoise(var_limit=(10.0, 40.0), p=0.3)
        base = [
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            gauss_noise,
            A.ElasticTransform(alpha=1.0, sigma=1.0, p=0.2),  # sigma >= 1 in albumentations 2.x
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        base = list(get_train_transform().transforms)
    if resize:
        base = [A.Resize(resize, resize)] + base
    return A.Compose(base)


def _build_loaders(data_dir: str, batch_size: int, resize: int, aug: str, device: torch.device):
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

    train_transform = _build_augmentation(aug, resize)
    val_transform = list(get_val_test_transform().transforms)
    if resize:
        val_transform = [A.Resize(resize, resize)] + val_transform
    val_transform = A.Compose(val_transform)

    nw = 2 if sys.platform.startswith("linux") else 0
    pin = device.type == "cuda"
    train_loader = DataLoader(
        CrackDataset(train_imgs, train_masks, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        CrackDataset(val_imgs, val_masks, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin,
    )
    return train_loader, val_loader


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train crack-seg U-Net")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="Checkpoint output path (.pth)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto from GPU preset")
    ap.add_argument("--resolution", "--resize", dest="resolution", type=int, default=0,
                    help="Square resize; 0 = auto from GPU preset (512)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    ap.add_argument("--early-stop", type=int, default=5, help="Patience on val macro-Dice (0 disables)")
    ap.add_argument("--aug", choices=["standard", "strong"], default="standard")
    ap.add_argument("--features", type=str, default="32,64,128,256", help="Encoder widths, comma-separated")
    ap.add_argument("--se", action="store_true", help="Squeeze-and-excitation blocks")
    ap.add_argument("--dropout", type=float, default=0.0, help="Bottleneck Dropout2d rate")
    ap.add_argument("--deep-supervision", action="store_true", help="Auxiliary decoder heads during training")
    ap.add_argument("--upsample", choices=["conv_transpose", "interpolate"], default="conv_transpose")
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="Mixed precision (default: on for CUDA, off otherwise)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None, help="Optional state dict to start from (resume/pretrain)")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    features = [int(x) for x in args.features.split(",")]
    resolution, preset_batch = resolve_training_preset(features, gpu_name)
    resolution = args.resolution or resolution
    batch_size = args.batch_size or preset_batch
    use_amp = args.amp if args.amp is not None else device.type == "cuda"

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=features,
        se=args.se,
        dropout=args.dropout,
        deep_supervision=args.deep_supervision,
        upsample_mode=args.upsample,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        print(f"Loaded initial weights from {args.checkpoint}")

    criterion = BCEDiceLoss(bce_weight=args.bce_weight, aux_weight=args.aux_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_scaler = getattr(torch.amp, "GradScaler", None)
    if amp_scaler is not None:
        scaler = amp_scaler("cuda", enabled=use_amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

    def set_lr(epoch: int) -> None:
        """Linear warmup -> cosine decay, applied before each epoch's training."""
        if args.lr_scheduler != "cosine":
            return
        warmup = max(args.warmup_epochs, 1)
        if epoch <= warmup:
            factor = epoch / warmup
        else:
            progress = (epoch - warmup) / max(1, args.epochs - warmup)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor

    train_loader, val_loader = _build_loaders(args.data_dir, batch_size, resolution, args.aug, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} ({gpu_name or 'cpu'}) | resolution={resolution} | batch={batch_size} "
          f"| amp={use_amp} | features={features} | se={args.se} | dropout={args.dropout} "
          f"| deep_supervision={args.deep_supervision} | upsample={args.upsample} "
          f"| params={n_params/1e6:.2f}M")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # EMA bookkeeping
    ema_state = None
    if args.ema > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def evaluate_with_ema():
        if ema_state is None:
            return evaluate(model, val_loader, device)
        saved = model.state_dict()
        model.load_state_dict(ema_state)
        try:
            return evaluate(model, val_loader, device)
        finally:
            model.load_state_dict(saved)

    def save_checkpoint() -> None:
        torch.save(ema_state or model.state_dict(), args.out)

    best_dice = -1.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        set_lr(epoch)
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = _mask_permute(masks).to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if args.deep_supervision:
                    logits, aux = model.forward_deep(imgs)
                    loss = criterion(logits, masks, aux)
                else:
                    loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        if args.ema > 0:
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_state[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
                else:
                    ema_state[k].copy_(v.detach())

        val_metrics = evaluate_with_ema()
        macro_dice = val_metrics["macro"]["dice"]
        g = val_metrics["global"]
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.2e} | train_loss={train_loss/len(train_loader):.4f} "
            f"| val Dice={macro_dice:.4f} (macro) {g['dice']:.4f} (global) "
            f"| IoU={g['iou']:.4f} | Rec={g['recall']:.4f} | Prec={g['precision']:.4f} "
            f"| {time.time()-t0:.1f}s"
        )

        if scheduler is not None and args.lr_scheduler == "plateau" and epoch >= args.warmup_epochs:
            scheduler.step(macro_dice)

        if macro_dice > best_dice + 1e-4:
            best_dice = macro_dice
            no_improve = 0
            save_checkpoint()
            print(f"  -> new best, saved to {args.out}")
        else:
            no_improve += 1
            if args.early_stop and no_improve >= args.early_stop:
                print(f"Early stopping (no val improvement for {args.early_stop} epochs).")
                break

    print(f"Done. Best val macro Dice: {best_dice:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
