import pytest
import numpy as np
import torch
import torch.nn as nn
from src.config.schema import SystemSettings, AppSettings, ModelSettings, InferenceSettings, MetricsSettings
from src.inference.sliding_window import SlidingWindowPredictor

class DummySegmentationModel(nn.Module):
    """
    Dummy segmentation model outputting constant logits.
    """
    def __init__(self, output_val: float = 1.0):
        super().__init__()
        self.output_val = output_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, H, W)
        # Output: (B, 1, H, W) with logits equal to constant
        outputs = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32) * self.output_val
        return outputs


@pytest.fixture
def dummy_settings():
    return SystemSettings(
        app=AppSettings(host="0.0.0.0", port=8000, debug=True, log_level="info"),
        model=ModelSettings(
            checkpoint_path="dummy.pth",
            hf_repo_id="dummy/repo",
            hf_filename="dummy.pth",
            in_channels=3,
            out_channels=1,
            features=[64, 128]
        ),
        inference=InferenceSettings(
            # testing at 256 for efficiency, it's 448 in main config
            patch_size=256,
            overlap=0.5,
            sigma_scale=0.125,
            default_threshold=0.5
        ),
        metrics=MetricsSettings(
            density_cell_size=16,
            max_density_threshold=0.05
        )
    )


def test_gaussian_kernel_generation(dummy_settings):
    """Verifies the generated Gaussian kernel has correct dimensions and weights."""
    model = DummySegmentationModel()
    predictor = SlidingWindowPredictor(model, dummy_settings, device=torch.device("cpu"))

    kernel = predictor.gaussian_kernel
    assert kernel.shape == (256, 256)
    # The center of the patch should have the highest weight
    assert kernel[128, 128] > kernel[0, 0]
    # Boundaries should be low weight but not zero 
    assert kernel[0, 0] >= 1e-4


def test_predict_large_image_standard(dummy_settings):
    """Verifies sliding window inference when image larger than patch tile."""
    # sigmoid(2.0) = 0.88 > threshold of 0.5, all positive logits
    model = DummySegmentationModel(output_val=2.0)
    predictor = SlidingWindowPredictor(model, dummy_settings, device=torch.device("cpu"))

    # Image shape: 512x512x3 (larger than patch_size=256)
    dummy_image = np.ones((512, 512, 3), dtype=np.uint8) * 128

    probs, mask, ratio = predictor.predict_large_image(dummy_image)

    assert probs.shape == (512, 512)
    assert mask.shape == (512, 512)
    # Positive logits mean all cells should be detected as cracks
    assert np.all(mask == 1)
    assert abs(ratio - 1.0) < 1e-5


def test_predict_large_image_small_input(dummy_settings):
    """Verifies that input images smaller than patch sizes are handled correctly using padding and cropping."""
    model = DummySegmentationModel(output_val=-2.0) # Sigmoid(-2) = 0.12 < 0.5 threshold
    predictor = SlidingWindowPredictor(model, dummy_settings, device=torch.device("cpu"))

    # Image coordinates (128x128) is smaller than mock patch_size 256
    dummy_small_img = np.ones((128, 128, 3), dtype=np.uint8) * 200

    probs, mask, ratio = predictor.predict_large_image(dummy_small_img)

    assert probs.shape == (128, 128)
    assert mask.shape == (128, 128)
    assert np.all(mask == 0)
    assert ratio == 0.0
