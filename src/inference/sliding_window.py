import numpy as np
import torch
import torch.nn as nn
import cv2
from src.config.schema import SystemSettings


class SlidingWindowPredictor:
    """
    High-resolution inference for the crack-seg U-Net.

    Images larger than the training patch size are split into overlapping
    tiles, run through the model in batches (with optional horizontal /
    vertical flip test-time augmentation), and blended back together with a
    2D Gaussian weight map to remove seam lines. Images at or below the patch
    size use a single full-image forward pass instead: the model is fully
    convolutional, so this is faster and avoids padded-border artifacts.
    """
    def __init__(self, model: nn.Module, settings: SystemSettings, device: torch.device,
                 batch_size: int = None, tta: bool = None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.settings = settings

        # Precompute Gaussian weight map
        self.patch_size = settings.inference.patch_size
        self.gaussian_kernel = self._generate_gaussian_kernel(
            self.patch_size, settings.inference.sigma_scale
        )
        self.batch_size = batch_size or settings.inference.batch_size
        # Batching pays off on GPUs (kernel-launch overhead) but can hurt
        # single-image CPU paths; default to 8 on CUDA when not configured.
        if device.type == "cuda" and self.batch_size == 1:
            self.batch_size = 8
        self.tta = settings.inference.tta if tta is None else tta

        # ImageNet normalization params (kept as device tensors for batched ops)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

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

    def _preprocess_batch(self, patches_rgb: np.ndarray) -> torch.Tensor:
        """
        (B, H, W, 3) uint8 RGB -> (B, 3, H, W) float32, ImageNet-normalized.
        Mirrors the training-time normalization exactly.
        """
        tensor = torch.from_numpy(patches_rgb).permute(0, 3, 1, 2).to(self.device)
        return (tensor.float() / 255.0 - self.mean) / self.std

    def _run_model(self, batch: torch.Tensor, tta: bool = None) -> np.ndarray:
        """
        Forward a batch with optional flip TTA. Returns (B, H, W) probabilities.
        TTA averages the sigmoid outputs of the original and h/v/both-flipped
        views, which typically adds ~1-2 Dice points at 4x inference cost.
        """
        use_tta = self.tta if tta is None else tta
        with torch.inference_mode():
            probs = torch.sigmoid(self.model(batch))
            if use_tta:
                for dims in ([3], [2], [2, 3]):
                    flipped_input = torch.flip(batch, dims)
                    probs = probs + torch.flip(torch.sigmoid(self.model(flipped_input)), dims)
                probs = probs / 4.0
        return probs.squeeze(1).cpu().numpy()

    def _predict_direct(self, image_rgb: np.ndarray, threshold: float, tta: bool = None) -> tuple:
        """Single full-image forward pass for inputs at or below the patch size."""
        h_img, w_img, _ = image_rgb.shape
        probs = self._run_model(self._preprocess_batch(image_rgb[np.newaxis]), tta)[0]
        binary_mask = (probs > threshold).astype(np.uint8)
        crack_area_ratio = float(np.sum(binary_mask) / (h_img * w_img))
        return probs, binary_mask, crack_area_ratio

    def predict_large_image(
        self,
        image_rgb: np.ndarray,
        threshold: float = None,
        overlap: float = None,
        tta: bool = None
    ) -> tuple:
        """
        Runs inference on images of any size and returns:
            final_probs   (np.ndarray): continuous probability map in [0, 1]
            binary_mask   (np.ndarray): thresholded segmentation mask (0/1)
            crack_area_ratio (float):   cracked pixels / total pixels
        """
        h_img, w_img, _ = image_rgb.shape
        t_val = threshold if threshold is not None else self.settings.inference.default_threshold
        overlap_val = overlap if overlap is not None else self.settings.inference.overlap

        # Fast, artifact-free path for images that fit in a single forward pass
        if max(h_img, w_img) <= self.patch_size:
            return self._predict_direct(image_rgb, t_val, tta)

        stride = int(self.patch_size * (1.0 - overlap_val))

        # Pad reflective-borders if one side is smaller than patch_size
        if h_img < self.patch_size or w_img < self.patch_size:
            pad_h = max(0, self.patch_size - h_img)
            pad_w = max(0, self.patch_size - w_img)
            padded_img = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            p_map, b_mask, _ = self.predict_large_image(
                padded_img, threshold=t_val, overlap=overlap_val, tta=tta
            )
            return p_map[:h_img, :w_img], b_mask[:h_img, :w_img], float(np.sum(b_mask[:h_img, :w_img]) / (h_img * w_img))

        # Generate scanning coordinates across vertical and horizontal planes
        y_starts = list(range(0, h_img - self.patch_size + 1, stride))
        if y_starts[-1] + self.patch_size < h_img:
            y_starts.append(h_img - self.patch_size)

        x_starts = list(range(0, w_img - self.patch_size + 1, stride))
        if x_starts[-1] + self.patch_size < w_img:
            x_starts.append(w_img - self.patch_size)

        coords = [(y, x) for y in y_starts for x in x_starts]
        kernel = self.gaussian_kernel

        # Accumulators for weighted predictions and normalization weights
        accum_p = np.zeros((h_img, w_img), dtype=np.float32)
        accum_w = np.zeros((h_img, w_img), dtype=np.float32)

        # Batched sliding window inference
        for start in range(0, len(coords), self.batch_size):
            chunk = coords[start:start + self.batch_size]
            patches = np.stack([
                image_rgb[y:y + self.patch_size, x:x + self.patch_size] for y, x in chunk
            ])
            probs_batch = self._run_model(self._preprocess_batch(patches), tta)
            for (y, x), probs in zip(chunk, probs_batch):
                accum_p[y:y + self.patch_size, x:x + self.patch_size] += probs * kernel
                accum_w[y:y + self.patch_size, x:x + self.patch_size] += kernel

        # Element-wise division to normalize final blended probabilities
        final_probs = accum_p / np.maximum(accum_w, 1e-5)
        binary_mask = (final_probs > t_val).astype(np.uint8)
        crack_area_ratio = float(np.sum(binary_mask) / (h_img * w_img))

        return final_probs, binary_mask, crack_area_ratio
