"""Gradio demo Space for the crack-seg U-Net v3 on ZeroGPU."""

import logging
import os

import cv2
import numpy as np
import torch

from src.config.schema import SystemSettings
from src.inference.sliding_window import SlidingWindowPredictor
from src.models.checkpoint import load_unet_checkpoint
from src.utils.device import select_device
from src.utils.visualizations import create_visual_overlay

# `spaces` is required on ZeroGPU but effect-free locally; `gradio` is only
# needed to build the UI. Both are optional imports so the app can be smoke
# tested on CPU without installing either.
try:
    import spaces
except ImportError:  # pragma: no cover - local CPU runs
    spaces = None

try:
    import gradio as gr
except ImportError:  # pragma: no cover - local CPU runs
    gr = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crack_seg_space")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "unet_v3.pth")
HF_REPO_ID = "ishaan1402/crack-seg"
HF_FILENAME = "unet_v3.pth"

# Parity with the FastAPI service limits (50 MB payload, 8192 px sides).
MAX_IMAGE_DIM = 8192
MAX_PIXELS = 17_500_000  # ~50 MB decoded RGB

settings = SystemSettings.load_from_yaml(CONFIG_PATH)
device = select_device()

if not os.path.exists(CHECKPOINT_PATH):
    logger.info("Downloading %s from %s ...", HF_FILENAME, HF_REPO_ID)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=os.path.dirname(CHECKPOINT_PATH),
        local_dir_use_symlinks=False,
    )

model, detected_features = load_unet_checkpoint(CHECKPOINT_PATH, device)
predictor = SlidingWindowPredictor(model, settings, device)
logger.info(
    "Model ready: %s (features=%s, params=%.1fM, device=%s)",
    HF_FILENAME,
    detected_features,
    sum(p.numel() for p in model.parameters()) / 1e6,
    device,
)


def estimate_duration(image: np.ndarray) -> int:
    """Declare the ZeroGPU request duration from the expected patch count."""
    if image is None:
        return 10
    h, w = image.shape[:2]
    patch = predictor.patch_size
    stride = int(patch * (1.0 - settings.inference.overlap))

    def n_starts(size: int) -> int:
        if size <= patch:
            return 1
        starts = list(range(0, size - patch + 1, stride))
        if starts[-1] + patch < size:
            starts.append(size - patch)
        return len(starts)

    n_patches = n_starts(h) * n_starts(w)
    return int(min(120, max(10, 8 + n_patches * 0.04)))


def _segment(image: np.ndarray):
    """Run sliding-window inference and return (overlay_rgb, result_label)."""
    if image is None:
        raise ValueError("Please upload an image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        raise ValueError(
            f"Image is {max(h, w)} px on its longest side; the limit is {MAX_IMAGE_DIM} px."
        )
    if h * w > MAX_PIXELS:
        raise ValueError(
            "Image decodes to more than ~50 MB; please upload a smaller or downscaled version."
        )

    probs, binary_mask, crack_area_ratio = predictor.predict_large_image(image)
    overlay_bgr = create_visual_overlay(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), binary_mask, "mask", settings
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb, f"Crack area: {crack_area_ratio * 100:.2f}%"


segment = spaces.GPU(duration=estimate_duration)(_segment) if spaces is not None else _segment

if gr is not None:
    demo = gr.Interface(
        fn=segment,
        inputs=gr.Image(
            type="numpy",
            label="Bridge photo (JPEG/PNG)",
            sources=["upload"],
        ),
        outputs=[
            gr.Image(label="Crack mask overlay"),
            gr.Label(label="Result"),
        ],
        title="Bridge Crack Detection — U-Net v3",
        description=(
            "Pixel-level crack segmentation with the narrow U-Net v3 (7.8M params), "
            "trained on UAV and multi-source bridge imagery. Large photos are tiled "
            "into overlapping 448 px patches with a Gaussian-weighted blend."
        ),
        examples=[["examples/example_1.jpeg"]],
    )
else:
    demo = None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("gradio is not installed; install requirements.txt to launch the UI.")
    demo.launch()
