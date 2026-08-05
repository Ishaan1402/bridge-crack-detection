---
license: apache-2.0
library_name: custom
pipeline_tag: image-segmentation
tags:
- pytorch
- unet
- semantic-segmentation
- bridge-inspection
- uav
---

# Bridge Crack Detection (crack-seg)

Pixel-level semantic segmentation of cracks in high-resolution UAV bridge
imagery with a U-Net encoder-decoder. Inference tiles large images into
overlapping 448×448 patches, blends them with a 2D Gaussian weight map, and
optionally emits a crack-density heatmap. See the
[code repo](https://github.com/Ishaan1402/crack-seg) for the full pipeline.

## Checkpoints

| File | Architecture | Notes |
| --- | --- | --- |
| `unet_v3.pth` | U-Net `[32,64,128,256]` + SE blocks + deep supervision | **Recommended.** Trained 2026-08 on a curated multi-source mix (UAV Kaggle + DeepCrack train + merged crack sources), 512px, BCE+Dice, Adam 5e-4 + cosine, EMA. |
| `unet_v3_ema.pth` | same | EMA-smoothed snapshot of v3 (optional; usually within noise of the raw model). |
| `unet_narrow_v2.pth` | U-Net `[32,64,128,256]`, plain | Previous HPO pick; kept for comparison. `best_unet.pth` is an alias of this. |
| `unet_wide_v1.pth` | U-Net `[64,128,256,512]`, plain | Original notebook model (4× params). |

All checkpoints load through the same code path — the loader
(`src/models/checkpoint.py`) auto-detects channel widths, legacy key naming,
SE/deep-supervision heads, and upsample mode.

## Metrics (threshold 0.5, direct full-image inference)

### Held-out staged test split (447 pairs)

| Model | Dice | IoU | Recall | Precision |
| --- | --- | --- | --- | --- |
| **v3 (`unet_v3.pth`)** | **0.730** | 0.575 | 0.739 | 0.721 |
| narrow-v2 | 0.310 | 0.183 | 0.200 | 0.690 |
| wide-v1 | 0.339 | 0.204 | 0.231 | 0.634 |

The staged test split is the clean in-distribution held-out benchmark (only
the val split is used for checkpoint selection). The old models were trained
on UAV-only data, which is why they drop sharply here.

### DeepCrack test set (237 images)

| Model | Dice | IoU | Recall | Precision |
| --- | --- | --- | --- | --- |
| **v3** | **0.866** | **0.764** | **0.867** | **0.866** |
| narrow-v2 | 0.720 | 0.562 | 0.656 | 0.797 |
| wide-v1 | 0.775 | 0.633 | 0.727 | 0.830 |

⚠️ Provenance note: DeepCrack *train* was part of v3's training data, so the
DeepCrack test number is held-out but **in-distribution**, not cross-domain.
The staged test split above is the harder held-out benchmark.

### Per-source (staged test split, flagged pairs excluded, v3)

| Source | n | Mean Dice |
| --- | --- | --- |
| CRACK500 | 123 | 0.749 |
| DeepCrack (merged copy) | 23 | 0.788 |
| Eugen | 2 | 0.763 |
| Volker | 40 | 0.732 |
| uav | 47 | 0.669 |
| Rissbilder | 122 | 0.653 |
| GAPS384 | 21 | 0.544 |
| CFD | 4 | 0.531 |
| forest | 7 | 0.531 |
| Sylvie | 5 | 0.507 |
| cracktree200 | 3 | 0.233 |
| noncrack (no-crack images, reported separately) | 46 | 0.609 |

## Data-integrity note

Four pairs in the staged test split were excluded after both visual review
and an automated mismatch-signature check (model confidently predicts a crack
where the GT says nothing is there): 3 CRACK500 files and 1 DeepCrack file,
traced to a tile-index/coordinate inconsistency in the upstream merged-dataset
artifact. Exclusion is reporting-only — no labels were auto-fixed, and the
exclusions do not materially change the aggregate metrics above.

## Training (v3)

- Data: UAV Kaggle (fresh stratified 70/15/15 split), DeepCrack train (300),
  and a capped subset of a merged 11.2k crack dataset (CRACK500, CFD, GAPS384,
  Rissbilder, Volker, Sylvie, forest, cracktree200, noncrack), all resized to
  512×512 with binary masks and ImageNet normalization.
- Model: narrow U-Net + squeeze-and-excitation, bottleneck dropout 0.1, deep
  supervision, 512px, BCE+Dice (aux-weighted), Adam lr 5e-4 with 3-epoch
  warmup + cosine decay, EMA, AMP, 30 epochs, best-by-val-Dice.
- Validation Dice ≈ 0.73 global / 0.66 macro at completion.

## Usage

```bash
pip install -r requirements.txt
uvicorn src.app:app --port 8000
curl -X POST -F "file=@bridge.jpg" \
  "http://127.0.0.1:8000/predict?overlay_type=both" -o result.jpg
```

Load a checkpoint directly:

```python
from src.models.checkpoint import load_unet_checkpoint
import torch

model, features = load_unet_checkpoint("unet_v3.pth", torch.device("cpu"))
```

Code repo: [Ishaan1402/crack-seg](https://github.com/Ishaan1402/crack-seg)
