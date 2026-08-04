# Training data for crack segmentation — sources & how they fit

## What the project's pipeline expects

The training scheme (Colab notebook / `scripts/train.py`) consumes:

```
{data_dir}/{train,val,test}/images/*.jpg
{data_dir}/{train,val,test}/masks/*.png     (binary: 0 = background, 255 = crack)
```

- Images are read as RGB; masks are read grayscale and binarized with
  `> 127` (or an explicit class list).
- Sizes should be divisible by 16 (4 pooling levels); 448×448 or 512×512 is
  the sweet spot for this U-Net.
- Augmentation is flips + color jitter + ImageNet normalization.

Any dataset below that produces paired RGB images + crack masks can be
converted with `scripts/prepare_dataset.py`:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_dataset.py \
    --images path/Images --masks path/Masks --out input/dataset_split \
    --resize 448 --test-frac 0.2
```

For multi-class label masks (e.g. GAPs), pass `--classes 1 2 3 5` to keep
only the crack classes and drop everything else.

---

## Closest to your use case (UAV / concrete bridge)

| Dataset | Domain | Size | Mask format | Where |
| --- | --- | --- | --- | --- |
| **UAV Crack Detection (ziya07)** | UAV concrete | 315 pairs, 448×448 | binary PNG | Kaggle (already your primary) |
| **Crack-Detection-and-Segmentation-Dataset-for-UAV-Inspection** | UAV concrete/structures | 11,298 images, pixel masks | images + masks | github.com/Auto-ROS-LAB/Crack-Detection-and-Segmentation-Dataset-for-UAV-Inspection |
| **NCCD-PF** | pre-failure narrow concrete cracks (abutments, walls, pillars) | ~13k images, binary `Images/` + `Masks/` folders | binary masks (exact fit) | Zenodo (Nature Scientific Data 2023) |
| **Kansas bridges drone inspection** | real drone bridge inspection, 8 defect classes incl. crack | annotated drone imagery/video | YOLOv11 instance-seg outputs (polygons → mask) | Zenodo record 17477702 |
| **DeepCrack** | concrete crack crops | 537 (300 train / 237 test) | binary PNG, 384×544 crops | github.com/yhlleo/DeepCrack (already in `input/DeepCrack`) |
| **HRCD-282** | high-res concrete cracks | 282 high-res images | masks | UCL Discovery / paper "CBRF" |
| **SDNET2018** | bridge decks, walls, pavement | 56k+ images | ⚠️ **image-level labels only, no pixel masks** | sdnet2018.com — use only for classification pretraining, not segmentation |

## Pavement / road (good for robustness, thin-crack variety)

| Dataset | Size | Mask format | Where |
| --- | --- | --- | --- |
| **CrackSeg9k** | 9k+ curated from 10 datasets | consistent PNG masks | github.com/Dhananjay42/crackseg9k; also HF `rimvydasrub/crackseg9k` |
| **CRACK500** | 1896 train / 348 val / 1124 test | binary masks | github.com/khanhha/crack_segmentation |
| **CFD / CrackForest** | 118 images | binary masks | same khanhha repo |
| **GAPs384** | ~509 images, 1920×1080 | gray-value class masks (needs `--classes`) | github.com/... (GAPs; see Pavement-Defect-Datasets index) |

## Ready-to-export / synthetic

- **Roboflow Universe** — search "bridge crack segmentation"; datasets export
  directly to image + mask PNG (e.g. 1.16k-image bridge crack sets). Fastest
  way to get bridge-domain masks today.
- **Hugging Face datasets** — e.g. `rimvydasrub/crackseg9k`,
  `sophy666/LCSD` (low-light cracks), synthetic masonry cracks
  (`DavidHidde/synthetic-masonry-surfaces`). Synthetic cracks are useful for
  domain-randomization pretraining.
- **Mendeley/Figshare mirrors** of CRACK500-with-noise, combined benchmarks,
  etc. (useful for studying label noise).

---

## Recommended strategy

1. **Primary**: keep the UAV Kaggle set for the final test claim — it is the
   only data you've evaluated on, and your real test split must come from it.
2. **Bridge-concrete addition** (highest value): Auto-ROS-LAB UAV 11k +
   NCCD-PF. Both are pixel-mask and convert with zero code changes; the UAV
   11k set in particular is the same capture domain as your use case.
3. **Robustness layer**: add a slice of CrackSeg9k/CRACK500 + DeepCrack so the
   model sees thin, low-contrast cracks at many scales. DeepCrack is already
   on disk — that's ~537 free pairs today.
4. **Avoid** SDNET2018 for mask training (no masks). Roboflow exports are
   fine but check label quality — many are model-generated (e.g. the Kansas
   set) and noisy.
5. **Protocol**: train on a mix, but always report (a) UAV test split Dice/IoU
   and (b) DeepCrack test as a cross-domain number. Keep splits stratified by
   source so no source leaks between train and test.

## Suggested training runs once the data is staged

```bash
# v3 candidate A: plain narrow, higher res, more data
PYTHONPATH=. .venv/bin/python scripts/train.py --data-dir input/dataset_split \
    --epochs 30 --batch-size 8 --lr 1e-4 --resize 512 --features 32,64,128,256 \
    --out checkpoints/unet_v3_plain.pth

# v3 candidate B: the small upgrades
PYTHONPATH=. .venv/bin/python scripts/train.py --data-dir input/dataset_split \
    --epochs 30 --batch-size 8 --lr 1e-4 --resize 512 --features 32,64,128,256 \
    --se --dropout 0.1 --deep-supervision --out checkpoints/unet_v3_upgraded.pth
```

Then evaluate both against the current narrow/wide checkpoints with
`scripts/verify_metrics.py` on the UAV test split and DeepCrack.
