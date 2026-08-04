"""
Robust checkpoint loading for the crack-seg U-Net.

The published weights have drifted across training runs:

- older checkpoints (notebook / HPO runs, e.g. ``unet_747.pt``) are the
  *narrow* ``[32, 64, 128, 256]`` variant and name the upsampling blocks
  ``up.*`` and the output head ``final.*``;
- the current ``src.models.unet.UNet`` uses ``up_transposes.*`` /
  ``final_conv.*`` and defaults to the *wide* ``[64, 128, 256, 512]`` width.

``load_unet_checkpoint`` auto-detects the width from the state dict, remaps
the legacy key names, and performs a strict load, so either checkpoint family
can be served without configuration changes.
"""

from __future__ import annotations

import torch

from src.models.unet import UNet


def inspect_state_dict(state_dict: dict) -> list[int]:
    """Detect the U-Net channel widths from the first encoder convolution."""
    if not isinstance(state_dict, dict) or "encoder.0.conv.0.weight" not in state_dict:
        raise ValueError(
            "Unexpected checkpoint format: expected a raw state dict from the "
            "crack-seg U-Net (found keys like 'encoder.0.conv.0.weight')."
        )

    first_feat = state_dict["encoder.0.conv.0.weight"].shape[0]
    if first_feat == 32:
        return [32, 64, 128, 256]
    if first_feat == 64:
        return [64, 128, 256, 512]
    raise ValueError(
        f"Unrecognized first encoder width {first_feat}; expected 32 (narrow) "
        "or 64 (wide)."
    )


def remap_legacy_keys(state_dict: dict) -> dict:
    """Map legacy ``up.*`` / ``final.*`` names onto the current module names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("up."):
            key = "up_transposes." + key[3:]
        elif key.startswith("final."):
            key = "final_conv." + key[6:]
        remapped[key] = value
    return remapped


def load_unet_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[UNet, list[int]]:
    """
    Build a U-Net matching the checkpoint and load its weights strictly.

    Returns:
        (model, detected_features)
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    features = inspect_state_dict(state_dict)

    model = UNet(in_channels=3, out_channels=1, features=features).to(device)
    model.load_state_dict(remap_legacy_keys(state_dict))
    model.eval()
    return model, features
