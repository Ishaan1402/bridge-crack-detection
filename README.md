# Bridge Crack Detection

Pixel-level semantic segmentation for bridge surface cracks.

**Poster:** [Automated Bridge Surface Crack Detection using UAV Imagery and Deep Learning Segmentation](references/bridge_crack_detection_poster_final.pdf) — AI Student Symposium 2026

---

## SparkNotes

**Problem:** Bridge visual inspections are dangerous, slow, expensive, and often require lane closures or scaffolding. Inspectors need to find thin cracks in thousands of square feet of concrete.

**What this does:** A UAV drone captures high-res photos of bridge surfaces (over, underneath, between lanes, etc). The images are fed into this model, which outputs a pixel-level segmentation crack mask giving engineers a prioritized repair map and cracked-area ratio. Output resolution will always match the input.

**Results (test set, 48 images, 448×448 Kaggle tiles):**


| Model                                                | Recall    | Precision | Dice     | IoU      |
| ---------------------------------------------------- | --------- | --------- | -------- | -------- |
| [Baseline](baseline.ipynb) — RF on LBP/HOG patches   | 0.78      | 0.073     | 0.13     | 0.07     |
| [U-Net](unet.ipynb) — encoder-decoder, BCE+Dice loss | **0.823** | **0.586** | **0.68** | **0.52** |


The naive baseline finds most crack pixels but bleeds predictions into the background (7.3% precision, ~8× worse than U-Net). U-Net is the usable model (+5× Dice/F1 vs baseline).

**Repo map:** `[baseline.ipynb](baseline.ipynb)` · `[unet.ipynb](unet.ipynb)` · `[notes/use_case.md](notes/use_case.md)` · `[notes/future_work.md](notes/future_work.md)`

---

## Notes

### Problem and approach

Cracks occupy a median ~0.93% of pixels — severe class imbalance. Most detection papers use object detection (YOLO, Faster R-CNN), which gives boxes and labels. This project uses **semantic segmentation** instead: the output is a mask that traces crack shape and width, which is what repair crews actually need.

See `[notes/use_case.md](notes/use_case.md)` for the full motivation.

### Dataset

315 image/mask pairs from the [UAV-Based Crack Detection](https://www.kaggle.com/datasets/ziya07/uav-based-crack-detection-dataset) Kaggle dataset. Images set to **448×448**.


| Split | Images |
| ----- | ------ |
| Train | 220    |
| Val   | 47     |
| Test  | 48     |


Layout: `{split}/images/` and `{split}/masks/`. Notebooks expect the full dataset zip on Google Drive at `bridge_crack_detection/dataset_split.zip`. 

**Image size:** Neither notebook resizes. Both read images at native resolution and output a mask of the same dimensions. The baseline tiles with `PATCH_SIZE=16`, so height and width must be divisible by 16. The U-Net is fully convolutional, 4 pooling levels. For batched U-Net training, all images in a split need to match dimensions. It's recommended to keep images anywhere from 256x256 to 512x512, as long as they are divisible by 16. 

### Models

**[baseline.ipynb](baseline.ipynb)** — baseline

1. Tile each image into 16×16 patches.
2. Convert each patch to a fixed feature vector: brightness, texture (LBP), edges (HOG).
3. Train `RandomForestClassifier(n_estimators=100, class_weight='balanced')`.
4. Subsample background patches at 10% during training to counter imbalance.
5. Reconstruct patch predictions back into the same resolution mask.

**[unet.ipynb](unet.ipynb)** — main model

1. Custom U-Net: 4-level encoder/decoder, features `[64, 128, 256, 512]`, skip connections.
2. `BCEDiceLoss` (50/50 weight): BCE penalizes per-pixel probabilities that disagree with the ground truth, meaning the more confident the mistake, the bigger the penalty. Dice Loss evaluates how much of the actual cracked region and predicted crack region overlap, pixelwise.
3. Augmentations: flips, color jitter, ImageNet normalization.
4. Adam, lr=1e-4, batch size 8, 25 epochs.

### Evaluation

Both notebooks use the same metrics over the test set:

- **Recall** — did we find the crack pixels? (primary metric; missing a crack is worse than a false alarm)
- **Precision** — how much of the predicted mask is actually cracked?
- **Dice / F1** — overlap quality between actual and predicted
- **IoU** — strict overlap between actual and predicted



Each notebook ends with a visualization grid: input image, ground-truth mask, predicted mask.

### What I'd do next

Details in `[notes/future_work.md](notes/future_work.md)`: temporal crack progression, BIM/3D integration, multi-class defects (rust, spalling, rebar), edge inference on drones.

### Reproduce

```text
1. Upload dataset_split.zip to Google Drive (or local path).
2. Open baseline.ipynb or unet.ipynb in Colab.
3. Run all cells. Training + eval are self-contained.
```

