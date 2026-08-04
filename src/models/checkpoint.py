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
    """Detect the U-Net channel widths from the encoder convolutions."""
    if not isinstance(state_dict, dict) or "encoder.0.conv.0.weight" not in state_dict:
        raise ValueError(
            "Unexpected checkpoint format: expected a raw state dict from the "
            "crack-seg U-Net (found keys like 'encoder.0.conv.0.weight')."
        )

    features = []
    i = 0
    while f"encoder.{i}.conv.0.weight" in state_dict:
        features.append(int(state_dict[f"encoder.{i}.conv.0.weight"].shape[0]))
        i += 1
    return features


def remap_legacy_keys(state_dict: dict, up_name: str = "up_transposes") -> dict:
    """Map legacy ``up.*`` / ``final.*`` names onto the current module names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("up."):
            key = f"{up_name}.{key[3:]}"
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
    # Tolerate checkpoints saved as {"state_dict": ..., ...} (e.g. with training metadata)
    if (
        isinstance(state_dict, dict)
        and "state_dict" in state_dict
        and isinstance(state_dict["state_dict"], dict)
        and "encoder.0.conv.0.weight" in state_dict["state_dict"]
    ):
        state_dict = state_dict["state_dict"]
    features = inspect_state_dict(state_dict)

    # Detect optional upgrades from the state dict so v2 and v3 checkpoints
    # are both servable. Dropout has no parameters and is inference-irrelevant.
    se = any(k.endswith(".se.fc.1.weight") for k in state_dict)
    deep_supervision = "deep_heads.0.weight" in state_dict

    def _build(upsample_mode: str) -> UNet:
        return UNet(
            in_channels=3,
            out_channels=1,
            features=features,
            se=se,
            deep_supervision=deep_supervision,
            upsample_mode=upsample_mode,
        ).to(device)

    # Prefer the checkpoint-compatible ConvTranspose decoder; fall back to the
    # interpolate decoder (bilinear + 1x1 conv) if the strict load fails.
    try:
        model = _build("conv_transpose")
        model.load_state_dict(remap_legacy_keys(state_dict))
    except RuntimeError:
        model = _build("interpolate")
        model.load_state_dict(remap_legacy_keys(state_dict, up_name="up_convs"))

    model.eval()
    return model, features
