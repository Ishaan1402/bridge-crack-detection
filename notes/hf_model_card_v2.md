# Draft: updated model card for `ishaan1402/crack-seg`

Paste this over the repo's `README.md`, then upload the two checkpoints below.

---

```yaml
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
```

# Bridge Crack Detection (crack-seg)

Pixel-level semantic segmentation of cracks in high-resolution UAV bridge
imagery with a U-Net encoder-decoder. The inference pipeline tiles
high-res images into overlapping 448×448 patches, blends them with a 2D
Gaussian weight map, and optionally emits a crack-density heatmap.

## Checkpoints

Two weight files are hosted here. Both load with the same U-Net code (width
and legacy key naming are auto-detected — see the code repo's
`src/models/checkpoint.py`).

| File | Architecture | Size | Notes |
| --- | --- | --- | --- |
| `unet_narrow_v2.pth` | U-Net `[32,64,128,256]` | 31 MB | HPO-selected (Pathfinder `bridge_crack_unet_v2` trial 102, A100). Default for the repo. |
| `unet_wide_v1.pth` | U-Net `[64,128,256,512]` | 118 MB | Original notebook model (4× params). Stronger on cross-domain (DeepCrack) evaluation. |

`best_unet.pth` is kept as an alias of `unet_narrow_v2.pth` for backward
compatibility.

## Metrics (honest provenance)

| Model | Set | Dice | IoU | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| narrow v2 | UAV *validation* (HPO run, 512 px, thr 0.5) | 0.747 | 0.596 | — | — |
| narrow v2 | DeepCrack test (237 imgs, thr 0.5, direct) | 0.720 | 0.562 | 0.656 | 0.797 |
| wide v1 | DeepCrack test (237 imgs, thr 0.5, direct) | 0.775 | 0.633 | 0.727 | 0.829 |

The 0.747 / 0.596 pair is a **validation-set** result from the HPO campaign,
not the held-out UAV test split. A held-out test-set evaluation is pending;
the model card will be updated with those numbers when available.

## Usage

```bash
pip install -r requirements.txt
uvicorn src.app:app --port 8000
curl -X POST -F "file=@bridge.jpg" \
  "http://127.0.0.1:8000/predict?overlay_type=both" -o result.jpg
```

Code repo: [Ishaan1402/crack-seg](https://github.com/Ishaan1402/crack-seg)
