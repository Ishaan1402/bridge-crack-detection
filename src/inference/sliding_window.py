import numpy as np
import torch
import torch.nn as nn
import cv2
from src.config.schema import SystemSettings

class SlidingWindowPredictor:
    """
    A sliding window predictor that runs inference on larger images
    using 2D Gaussian-weighted blending to smooth boundaries.
    """
    def __init__(self, model: nn.Module, settings: SystemSettings, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.settings = settings

        # Precompute Gaussian weight map
        self.patch_size = settings.inference.patch_size
        self.gaussian_kernel = self._generate_gaussian_kernel(
            self.patch_size, settings.inference.sigma_scale
        )

        # ImageNet normalization params
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _generate_gaussian_kernel(self, size: int, sigma_scale: float) -> np.ndarray:
        """
        Generates a 2D Gaussian density kernel matrix to weight predictions and
        downweight borders where convolutions will lack padding context.
        """
        center = (size - 1) / 2.0
        sigma = size * sigma_scale
        x = np.arange(size) - center
        y = np.arange(size) - center
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))

        # prevent dividing by zero
        return np.maximum(kernel, 1e-4)

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Normalizes patch imagery to PyTorch standardized format and transfers to device.
        """
        # RGB uint8 -> normalized float32, apply standard mean/std
        normalized = (patch.astype(np.float32) / 255.0 - self.mean) / self.std
        # HWC -> CHW 
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict_large_image(self, image_rgb: np.ndarray, threshold: float = None) -> tuple:
        """
        Splits a high-resolution image into tiles, performs model inference using
        sliding window, and blends predictions back using 2D Gaussian kernel.

        Returns:
            final_probs (np.ndarray): Full-resolution continuous probability maps in range [0, 1].
            binary_mask (np.ndarray): Predicted segmentation crack mask (0 or 1).
            crack_area_ratio (float): Ratio of cracked pixels to total image pixels.
        """
        h_img, w_img, _ = image_rgb.shape
        t_val = threshold if threshold is not None else self.settings.inference.default_threshold
        stride = int(self.patch_size * (1.0 - self.settings.inference.overlap))

        # Pad image reflective-borders if input is smaller than patch_size
        # Not recommended, best to keep at least 448x448
        if h_img < self.patch_size or w_img < self.patch_size:
            pad_h = max(0, self.patch_size - h_img)
            pad_w = max(0, self.patch_size - w_img)
            padded_img = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            p_map, b_mask, _ = self.predict_large_image(padded_img, t_val)
            return p_map[:h_img, :w_img], b_mask[:h_img, :w_img], float(np.sum(b_mask[:h_img, :w_img]) / (h_img * w_img))

        # Generate scanning coordinates across vertical and horizontal planes
        y_starts = list(range(0, h_img - self.patch_size + 1, stride))
        if y_starts[-1] + self.patch_size < h_img:
            y_starts.append(h_img - self.patch_size)

        x_starts = list(range(0, w_img - self.patch_size + 1, stride))
        if x_starts[-1] + self.patch_size < w_img:
            x_starts.append(w_img - self.patch_size)

        # Accumulators for weighted predictions and normalization weights
        accum_p = np.zeros((h_img, w_img), dtype=np.float32)
        accum_w = np.zeros((h_img, w_img), dtype=np.float32)

        # Perform sliding window inference
        for y in y_starts:
            for x in x_starts:
                patch = image_rgb[y : y + self.patch_size, x : x + self.patch_size]
                patch_tensor = self._preprocess_patch(patch)

                with torch.no_grad():
                    logits = self.model(patch_tensor)
                    # Convert logits to probability scores using sigmoid
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

                # Weighted accumulation
                accum_p[y : y + self.patch_size, x : x + self.patch_size] += probs * self.gaussian_kernel
                accum_w[y : y + self.patch_size, x : x + self.patch_size] += self.gaussian_kernel

        # Element-wise division to normalize final blended probabilities
        final_probs = accum_p / np.maximum(accum_w, 1e-5)
        binary_mask = (final_probs > t_val).astype(np.uint8)
        crack_area_ratio = float(np.sum(binary_mask) / (h_img * w_img))

        return final_probs, binary_mask, crack_area_ratio
