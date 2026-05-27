import pytest
import io
import numpy as np
import cv2
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def _generate_test_image_bytes(width: int, height: int) -> bytes:
    """Helper to generate a dummy RGB JPEG image in-memory."""
    # Create random image array
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def test_health_endpoint():
    """Verifies that the /health readiness probe works."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "model_loaded" in data


def test_predict_endpoint_valid_image():
    """Verifies image submissions within complying dimensions and sizes."""
    # Generate 448x448 mock image (perfect matches with patch limits)
    img_bytes = _generate_test_image_bytes(448, 448)

    file_payload = {"file": ("test_image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/predict?overlay_type=mask", files=file_payload)

    # Standard expectation (will return 200 even if mock UNet outputs constant logits)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    # Verify custom headers exist
    assert "x-crack-area-ratio" in response.headers
    assert "x-crack-detected" in response.headers
    assert "x-image-resolution-width" in response.headers
    assert "x-image-resolution-height" in response.headers
    assert "x-inference-time-ms" in response.headers

    # Check values matched
    assert response.headers["x-image-resolution-width"] == "448"
    assert response.headers["x-image-resolution-height"] == "448"


def test_predict_endpoint_invalid_format():
    """Verifies rejection of invalid file extensions."""
    file_payload = {"file": ("test_text.txt", io.BytesIO(b"random text data"), "text/plain")}
    response = client.post("/predict?overlay_type=mask", files=file_payload)

    assert response.status_code == 415
    assert "Unsupported image format" in response.json()["detail"]


def test_predict_endpoint_invalid_resolution_guard():
    """Verifies Pydantic resolution limits [256, 8192] trigger HTTP 400."""
    # 128x128 image is below the 256px resolution limits
    img_bytes = _generate_test_image_bytes(128, 128)

    file_payload = {"file": ("too_small.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/predict", files=file_payload)

    assert response.status_code == 400
    assert "Image does not comply with safety limits" in response.json()["detail"]


def test_predict_endpoint_oversized_payload_rejection():
    """Verifies file uploads greater than 50MB are rejected."""
    # Create fake oversized stream matching 51MB
    large_stream = io.BytesIO(b"0" * (51 * 1024 * 1024))

    file_payload = {"file": ("oversized.jpg", large_stream, "image/jpeg")}
    response = client.post("/predict", files=file_payload)

    assert response.status_code == 413
    assert "Image file size limit exceeded" in response.json()["detail"]
