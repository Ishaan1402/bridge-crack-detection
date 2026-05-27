# ClearSpan

Automated crack detection from high-resolution UAV bridge imagery; powered by U-Net, an Encoder-Decoder CNN model with skip connections. 

**Poster:** [Automated Bridge Surface Crack Detection using UAV Imagery and Deep Learning Segmentation](references/bridge_crack_detection_poster_final.pdf) — AI Student Symposium 2026

**Model weights:** [ishaan1402/bridge_crack_detection_U-Net](https://huggingface.co/ishaan1402/bridge_crack_detection_U-Net) on Hugging Face

## Demo

### Inference Demo



**Input | crack mask | density heatmap**

### Evaluation against Test Set



**Original | ground truth | prediction**

---

## SparkNotes

**Problem:** Bridge inspections are slow, dangerous, expensive, and require lane closures or scaffolding. Inspectors need to locate thin cracks in thousands of square feet of concrete. High-res drone images (4K/8K) normally exceed GPU memory limits if fed directly, while downsampling destroys detail in said images.

**What this does:** Outputs clean pixel-level segmentation masks of cracks from high-res UAV bridge photos. Sliding window inference tiles high-res imagery, runs U-Net segmentation on overlapping patches, and blends them using a 2D Gaussian weight map. This application also tracks cracked-area ratio, compiles local density heatmaps, and serves predictions via a FastAPI endpoint.

**Results (test set, 48 images, 448×448 Kaggle tiles):**


| Model                       | Recall    | Precision | Dice     | IoU      |
| --------------------------- | --------- | --------- | -------- | -------- |
| [Baseline](baseline.ipynb)  | 0.78      | 0.073     | 0.13     | 0.07     |
| [U-Net](src/models/unet.py) | **0.823** | **0.586** | **0.68** | **0.52** |


The Random Forest baseline is noisy and is notoriously prone to false positives due to the gridded nature of it's inference (7.3% precision). U-Net is the usable model (+5× Dice/F1 vs baseline). See the [evaluation grid](#test-set-evaluation) above for side-by-side predictions.

---

## Repo Structure

```text
bridge_crack_detection/
├── config/              # YAML configs
├── scripts/             # Developer CLI tools
│   ├── download_checkpoint.py # Model fetcher
│   └── threshold_sweep.py     # Hyperparameter tuning tool evaluating Precision-Recall curves
├── src/                 
│   ├── config/          # Pydantic schema validation
│   ├── dataset/         # PyTorch dataset & Albumentations transforms
│   ├── inference/       # Sliding window predictor
│   ├── metrics/         # Spatial density heatmaps
│   ├── models/          # Custom U-Net, in PyTorch
│   ├── utils/           # Visual overlay output helpers
│   └── app.py           # FastAPI server
└── tests/               # Pytest unit and integration test suite
```

---

## Features

### 1. Sliding Window Inference (`src/inference/`)

High-resolution drone photos cannot fit in the model all at once, so we seperate them into overlapping squares, run the model on each square, then stitch the results back together.

- Default patch size is 448×448 with 50% overlap (each step moves 224px).
- Overlapping regions are averaged together. Patches near the center of each square are weighted more than edges in order to remove the seam lines between tiles.
- Photos smaller than the configured patch size are padded, run through the model, then cropped back to the original size, though this is not recommended. The top and left borders sit on the patch edge where predictions are naturally downweighted, and padding on the right and bottom adds fake content that will skew detections on these sides.

### 2. Crack Density Heatmaps (`src/metrics/`)

Converts the binary crack mask into a colored map showing where damage is clustered.

- Splits the image into grids (e.g. 64×64 pixel blocks).
- Each grid receives a proportioned crack percentage (% of pixels marked as crack).
- Grid is upsampled back to original image size and colored in: [blue = undamaged, red = severely cracked]

### 3. API server (`src/app.py`)

FastAPI app serving predictions over HTTP.

- `**/health**` — verifies if server is up and running and if the model is loaded
- `**/predict**` — receives an image and outputs an overlay. Query params: `threshold`, `overlap`, `overlay_type` (`mask`, `heatmap`, or `both` side by side).
- Validates uploads: allowed formats (jpg/png/webp/tiff), max 50MB, image size between 256 and 8192 px.
- Returns statistics in response headers: cracked area %, inference time, whether any crack was found.

### 4. Docker (`Dockerfile`)

- **Build** — installs dependencies and downloads model weights from Hugging Face at build time.
- **Run** — runs as non-root user (`appuser`), starts Uvicorn.

---

## Local Development & Usage

### 1. Run Local Installation

Requires Python 3.10+ to be installed:

```bash
# Create venv and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all unit and integration tests
.venv/bin/pytest tests/
```

### 2. Launch FastAPI Server

```bash
# Start the server locally on port 8000
.venv/bin/uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Test endpoints via cURL:

```bash
# Get health status
curl http://127.0.0.1:8000/health

# Predict and save a side by side comparative panel
curl -X POST -F "file=@input/example_1.jpeg" \
  "http://127.0.0.1:8000/predict?overlay_type=both&threshold=0.5&overlap=0.5" \
  --output "result_panel_$(date +%Y%m%d_%H%M%S).jpg"
```

### 3. Generate Precision-Recall Curve Profiles

Execute a threshold parameter sweep on test splits:

```bash
python scripts/threshold_sweep.py --data_dir /path/to/my_split_dir --output_dir reports/
```

Match up the best performing thresholds which maxes overall Dice and displays performance charts in `reports/pr_curve.png`.