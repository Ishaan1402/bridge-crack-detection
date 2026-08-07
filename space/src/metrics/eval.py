import numpy as np
import cv2
from src.config.schema import SystemSettings

def generate_density_heatmap(binary_mask: np.ndarray, settings: SystemSettings) -> np.ndarray:
    """
    Constructs a localized density heatmap representing the concentration percentage
    of identified crack pixels over non-overlapping grids.

    The continuous risk profile is upscaled with bilinear interpolation to match the image,
    clipped at custom maximum risk thresholds, and mapped to the JET color spectrum (Blue -> Red).

    Args:
        binary_mask (np.ndarray): Predicted segmentations of shape (H, W), values either 0 or 1.
        settings (SystemSettings): System configuration parameters detailing metrics specs.

    Returns:
        colored_heatmap (np.ndarray): RGB colored heatmap of shape (H, W, 3).
    """
    h_img, w_img = binary_mask.shape
    c_size = settings.metrics.density_cell_size
    max_d = settings.metrics.max_density_threshold

    # Calculate grid dimensions mapping ceil adjustments
    grid_h = int(np.ceil(h_img / c_size))
    grid_w = int(np.ceil(w_img / c_size))

    # Pad binary map borders to align with exact cell reductions
    pad_h = (grid_h * c_size) - h_img
    pad_w = (grid_w * c_size) - w_img
    padded_mask = cv2.copyMakeBorder(binary_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Reshape grid into a 4D array to vectorize mean computation across block panels
    # dimensions map to: (grid_h, cell_height, grid_w, cell_width)
    reshaped = padded_mask.reshape(grid_h, c_size, grid_w, c_size)
    grid_densities = reshaped.mean(axis=(1, 3))

    # Crop back any padding residue to keep exact relative proportions
    raw_densities_cropped = grid_densities[:grid_h, :grid_w]

    # Handle single cell/pixel scaling corner cases safely
    interp_method = cv2.INTER_NEAREST if max(grid_h, grid_w) == 1 else cv2.INTER_LINEAR
    upscaled = cv2.resize(raw_densities_cropped, (w_img, h_img), interpolation=interp_method)

    # Clip densities above the max threshold boundary and scale to uint8 range
    capped = np.clip(upscaled / max_d, 0.0, 1.0)
    uint8_canvas = (capped * 255.0).astype(np.uint8)

    # Convert the gray density map to JET color spectrum
    colored_heatmap = cv2.applyColorMap(uint8_canvas, cv2.COLORMAP_JET)

    # Convert BGR (OpenCV default) back to RGB color space
    return cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
