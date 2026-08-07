import numpy as np
import cv2
from src.config.schema import SystemSettings
from src.metrics.eval import generate_density_heatmap

def create_visual_overlay(
    image_bgr: np.ndarray,
    b_mask: np.ndarray,
    overlay_type: str,
    settings: SystemSettings
) -> np.ndarray:

    if overlay_type == "mask":
        # Original binary mask overlay
        overlay = image_bgr.copy()
        overlay[b_mask == 1] = (0.5 * image_bgr[b_mask == 1] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        return overlay
    elif overlay_type == "heatmap":
        # Heatmap overlay
        heatmap_rgb = generate_density_heatmap(b_mask, settings)
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(image_bgr, 0.6, heatmap_bgr, 0.4, 0)
    elif overlay_type == "both":
        # Comparative overlays: [Original Image | Mask | Heatmap]
        mask_overlay = image_bgr.copy()
        mask_overlay[b_mask == 1] = (0.5 * image_bgr[b_mask == 1] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

        heatmap_rgb = generate_density_heatmap(b_mask, settings)
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        heatmap_overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_bgr, 0.4, 0)

        # Set the overlays side by side
        return np.hstack((image_bgr, mask_overlay, heatmap_overlay))
    else:
        raise ValueError(f"Unknown overlay type: {overlay_type}")
