import io
import time
import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import torch

from src.config.schema import SystemSettings
from src.models.checkpoint import load_unet_checkpoint
from src.inference.sliding_window import SlidingWindowPredictor
from src.utils.visualizations import create_visual_overlay

# Config and hardware
CONFIG_PATH = "config/config.yaml"
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

settings = SystemSettings.load_from_yaml(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for missing weight files
checkpoint_path = settings.model.checkpoint_path
if not os.path.exists(checkpoint_path):
    # Download from Hugging Face if missing
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        print(f"Weights missing. Automatically downloading from {settings.model.hf_repo_id}...")
        hf_hub_download(
            repo_id=settings.model.hf_repo_id,
            filename=settings.model.hf_filename,
            local_dir=os.path.dirname(checkpoint_path),
            local_dir_use_symlinks=False
        )
    except Exception as d_err:
        raise RuntimeError(
            f"Failed to download model weights from {settings.model.hf_repo_id}: {d_err}"
        ) from d_err

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

# Load weights, auto-detecting the checkpoint's channel widths and legacy key naming
model, detected_features = load_unet_checkpoint(checkpoint_path, device)
if detected_features != settings.model.features:
    print(
        f"Note: checkpoint uses features {detected_features}, config declares "
        f"{settings.model.features}; serving the checkpoint's architecture."
    )
model.eval()

predictor = SlidingWindowPredictor(model, settings, device)
app = FastAPI(title="Bridge Surface Crack Segmenter", version="1.0.0")


# Pydantic input schema
class ImagePayloadMetadata(BaseModel):
    width: int = Field(ge=256, le=8192, description="Image width limits: [256, 8192] px")
    height: int = Field(ge=256, le=8192, description="Image height limits: [256, 8192] px")
    channels: int = Field(ge=3, le=3, description="Color formats must strictly be 3-channel (RGB)")
    file_size_mb: float = Field(le=50.0, description="Payload boundary max = 50 MB")


@app.get("/health")
def health_check():
    """
    GET: Reports if server, model loading state, and hardware is ready.
    """
    weights_ready = os.path.exists(settings.model.checkpoint_path)
    return {
        "status": "ready" if weights_ready else "partial",
        "device": str(device),
        "model_loaded": checkpoint_path,
        "weights_present": weights_ready
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(None, ge=0.0, le=1.0, description="Segmentation confidence threshold"),
    overlap: float = Query(None, ge=0.0, le=0.9, description="Patch stride ratio"),
    tta: bool = Query(None, description="Test-time augmentation (flip averaging); default from config"),
    overlay_type: str = Query("mask", pattern="^(mask|heatmap|both)$", description="Visual response output")
):
    """
    POST: Receives high-resolution image uploads, parses inputs, executes  
    sliding window inference, and returns streaming image overlays.
    """
    # Image extension check 
    ext = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""
    if ext not in {".jpg", ".jpeg", ".png", ".tiff", ".webp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Allowed formats: JPEG, PNG, WEBP, TIFF"
        )

    # Size limitation check
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > 50.0:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file size limit exceeded (max 50MB, received: {file_size_mb:.2f}MB)"
        )

    # Decode byte stream array to numpy matrices
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode uploaded image bytes."
        )

    h, w, c = image_bgr.shape

    # Pydantic schema mapping
    try:
        ImagePayloadMetadata(
            width=w,
            height=h,
            channels=c,
            file_size_mb=file_size_mb
        )
    except Exception as validation_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image does not comply with safety limits: {str(validation_err)}"
        )

    # Convert BGR (from OpenCV) to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Execute inference off the event loop (CPU-bound work in a thread pool)
    t_start = time.perf_counter()
    _, b_mask, car = await run_in_threadpool(
        predictor.predict_large_image, image_rgb, threshold, overlap, tta
    )
    t_inference_ms = (time.perf_counter() - t_start) * 1000.0

    # Overlay representations
    output_image = await run_in_threadpool(
        create_visual_overlay, image_bgr, b_mask, overlay_type, settings
    )

    # Re-encode numpy matrix to JPEG byte stream
    _, im_encoded = cv2.imencode(".jpg", output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    stream = io.BytesIO(im_encoded.tobytes())

    # Response metadata headers
    headers = {
        "X-Crack-Area-Ratio": f"{car:.6f}",
        "X-Crack-Detected": "true" if car > 0.0001 else "false",
        "X-Image-Resolution-Width": str(w),
        "X-Image-Resolution-Height": str(h),
        "X-Inference-Time-Ms": f"{t_inference_ms:.2f}",
        "Access-Control-Expose-Headers": "X-Crack-Area-Ratio, X-Crack-Detected, X-Image-Resolution-Width, X-Image-Resolution-Height, X-Inference-Time-Ms"
    }

    return StreamingResponse(stream, media_type="image/jpeg", headers=headers)
