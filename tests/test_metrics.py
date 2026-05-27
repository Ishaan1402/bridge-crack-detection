import pytest
import numpy as np
from src.config.schema import SystemSettings, AppSettings, ModelSettings, InferenceSettings, MetricsSettings
from src.metrics.eval import generate_density_heatmap

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
            patch_size=256,
            overlap=0.5,
            sigma_scale=0.125,
            default_threshold=0.5
        ),
        metrics=MetricsSettings(
            density_cell_size=16,          # 16x16 grid panels
            max_density_threshold=0.25      # 25% crack density = severe (Red)
        )
    )


def test_empty_mask_heatmap(dummy_settings):
    """Verifies that an empty crack mask produces a fully dark blue heatmap, meaning 0 crack density."""
    # Mask size: 32x32 (4 cells of size 16x16)
    empty_mask = np.zeros((32, 32), dtype=np.uint8)

    heatmap = generate_density_heatmap(empty_mask, dummy_settings)

    assert heatmap.shape == (32, 32, 3)
    # Value for 0 is dark blue 
    for r in range(32):
        for c in range(32):
            pixel = heatmap[r, c]
            # Red value (index 0 of RGB) should be very low, Blue (index 2) should be very high
            assert pixel[0] < 50
            assert pixel[2] > 100


def test_high_density_clipping(dummy_settings):
    """Verify that a fully cracked mask triggers max_density_threshold (Solid Red)."""
    # 32x32 image with 100% density exceeding the 0.25 threshold
    solid_mask = np.ones((32, 32), dtype=np.uint8)

    heatmap = generate_density_heatmap(solid_mask, dummy_settings)

    assert heatmap.shape == (32, 32, 3)
    for r in range(32):
        for c in range(32):
            pixel = heatmap[r, c]
            assert pixel[0] > 100
            assert pixel[2] < 50


def test_partial_density_differential(dummy_settings):
    """Verifies that regional density variances are reflected in the array."""
    # Using a 32x32 mask. 
    # Top-left 16x16 block has 100% density (all ones).
    # Bottom-right 16x16 block is completely empty (all zeros).
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:16, 0:16] = 1

    heatmap = generate_density_heatmap(mask, dummy_settings)

    # Top-left region should have high density (Red values)
    top_left_pixel = heatmap[4, 4]
    assert top_left_pixel[0] > 100

    # Bottom-right region should have low density (Blue values)
    bottom_right_pixel = heatmap[28, 28]
    assert bottom_right_pixel[0] < 50
    assert bottom_right_pixel[2] > 100
