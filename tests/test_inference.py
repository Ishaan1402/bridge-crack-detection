import pytest
import numpy as np
import torch
import torch.nn as nn
from src.config.schema import SystemSettings, AppSettings, ModelSettings, InferenceSettings, MetricsSettings
from src.inference.sliding_window import SlidingWindowPredictor, resolve_batch_size
from src.models.unet import UNet

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


def test_predict_large_image_overlap_param(dummy_settings):
    """Verifies that passing custom param values works and does not modify default predictor's settings."""
    model = DummySegmentationModel(output_val=1.0)
    predictor = SlidingWindowPredictor(model, dummy_settings, device=torch.device("cpu"))

    # Initial setting value
    initial_setting_overlap = dummy_settings.inference.overlap
    assert initial_setting_overlap == 0.5

    dummy_image = np.ones((512, 512, 3), dtype=np.uint8) * 128

    # Pass a different overlap parameter
    probs, _, _ = predictor.predict_large_image(dummy_image, overlap=0.25)

    assert probs.shape == (512, 512)
    # Verify the settings value remains unchanged
    assert predictor.settings.inference.overlap == 0.5


def _real_model_settings(**inference_overrides):
    defaults = dict(patch_size=256, overlap=0.5, sigma_scale=0.125, default_threshold=0.5)
    defaults.update(inference_overrides)
    return SystemSettings(
        app=AppSettings(host="0.0.0.0", port=8000, debug=True, log_level="info"),
        model=ModelSettings(
            checkpoint_path="dummy.pth", hf_repo_id="dummy/repo", hf_filename="dummy.pth",
            in_channels=3, out_channels=1, features=[8, 16],
        ),
        inference=InferenceSettings(**defaults),
        metrics=MetricsSettings(density_cell_size=16, max_density_threshold=0.05),
    )


def test_batched_inference_matches_sequential():
    """Batching patches must not change the blended output vs one-at-a-time."""
    torch.manual_seed(0)
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    settings = _real_model_settings()
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    seq = SlidingWindowPredictor(model, settings, torch.device("cpu"), batch_size=1)
    bat = SlidingWindowPredictor(model, settings, torch.device("cpu"), batch_size=4)

    p_seq, m_seq, r_seq = seq.predict_large_image(image)
    p_bat, m_bat, r_bat = bat.predict_large_image(image)

    assert p_seq.shape == p_bat.shape == (300, 300)
    assert np.allclose(p_seq, p_bat, atol=1e-6)
    assert np.array_equal(m_seq, m_bat)
    assert r_seq == r_bat


def test_direct_path_for_subpatch_image():
    """Images at/below the patch size take a single forward pass."""
    torch.manual_seed(1)
    model = UNet(in_channels=3, out_channels=1, features=[8, 16])
    settings = _real_model_settings()
    predictor = SlidingWindowPredictor(model, settings, torch.device("cpu"))
    image = np.random.randint(0, 255, (200, 256, 3), dtype=np.uint8)

    probs, mask, ratio = predictor.predict_large_image(image)

    assert probs.shape == (200, 256)
    assert mask.shape == (200, 256)
    assert 0.0 <= ratio <= 1.0

    # Manual single forward for an exact match
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(((image.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None])
    with torch.inference_mode():
        expected = torch.sigmoid(model(x)).squeeze().numpy()
    assert np.allclose(probs, expected, atol=1e-6)


def test_tta_averaging_is_identity_for_constant_logits(dummy_settings):
    """TTA must not change outputs for a model with constant logits."""
    model = DummySegmentationModel(output_val=1.0)
    settings = dummy_settings
    image = np.ones((512, 512, 3), dtype=np.uint8) * 128

    plain = SlidingWindowPredictor(model, settings, torch.device("cpu"))
    tta = SlidingWindowPredictor(model, settings, torch.device("cpu"), tta=True)

    p_plain, _, _ = plain.predict_large_image(image)
    p_tta, _, _ = tta.predict_large_image(image)

    assert np.allclose(p_plain, p_tta, atol=1e-6)
    # Per-call override must not persist on the instance
    p_override, _, _ = plain.predict_large_image(image, tta=True)
    p_plain_again, _, _ = plain.predict_large_image(image)
    assert np.allclose(p_override, p_plain_again, atol=1e-6)


def test_overlap_must_be_less_than_one(dummy_settings):
    """overlap >= 1.0 would produce a zero stride and must be rejected."""
    model = DummySegmentationModel(output_val=1.0)
    predictor = SlidingWindowPredictor(model, dummy_settings, torch.device("cpu"))
    image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    with pytest.raises(ValueError, match="overlap"):
        predictor.predict_large_image(image, overlap=1.0)


def test_predict_rejects_non_rgb_input(dummy_settings):
    model = DummySegmentationModel(output_val=1.0)
    predictor = SlidingWindowPredictor(model, dummy_settings, torch.device("cpu"))
    with pytest.raises(ValueError, match="HxWx3"):
        predictor.predict_large_image(np.ones((128, 128), dtype=np.uint8))


def test_resolve_batch_size_auto_and_explicit():
    """0 = auto (1 on CPU, 8 on CUDA); explicit values are always honored."""
    assert resolve_batch_size(0, "cpu") == 1
    assert resolve_batch_size(0, "cuda") == 8
    assert resolve_batch_size(None, "cpu") == 1
    assert resolve_batch_size(1, "cuda") == 1
    assert resolve_batch_size(4, "cpu") == 4
    assert resolve_batch_size(8, "cpu") == 8
