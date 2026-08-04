# Audit — claimed metrics, inference pipeline, and model quality

Branch: `revision_8-2` · Date: 2026-08-03

## Claimed numbers under review

| Metric | Claimed |
| --- | --- |
| Dice | 0.747 |
| IoU | 0.596 |
| Recall | 92% |
| F1 | 0.89 |

## Verdict

**0.747 Dice and 0.596 IoU are real, but they are *validation-set* numbers
from an HPO trial, not held-out test results. The 92% recall and 0.89 F1 are
not backed by any artifact on disk and are mathematically inconsistent with
the Dice/IoU pair — at least one of the four numbers comes from a different
metric level, threshold, or run. Do not publish these as a test-set result
until they are reproduced on the real 48-image test split.**

---

## 1. Where the numbers actually come from

### 0.747 Dice — HPO validation run (`bridge_crack_unet_v2`, trial 102)

The Pathfinder HPO database (`~/pathfinder/.data/hpo_studies.db`) records
trial 102 of study `bridge_crack_unet_v2` (completed 2026-07-07 11:53,
NVIDIA A100) with:

```
primary_score = 0.7469950715700785   (label: "Dice Score")
primary_loss  = 0.41848192517719573
params        = batch_size=3, learning_rate=5.49e-4,
                loss_weight_ratio=0.1953, model_capacity=narrow
epoch_reached = 19
```

The Colab worker reports `hard_dice_score` — binary Dice at threshold 0.5,
averaged per image over the **validation split** (20% sequential split of the
UAV Kaggle dataset, 512 px center crop). So "0.747" is:

- a **validation** number, not the held-out 48-image test set;
- averaged per image (macro), not global pixel Dice;
- from the **final epoch (19)** — the best epoch was actually 18 (0.7532);
- computed on a **sequential 80/20 split** (sorted file order), which is not
  stratified/random and can inflate results.

The checkpoint for this trial is `~/Downloads/unet_747.pt`
(SHA `92cb2200…`), which is byte-identical to `best_unet.pth` currently
published on `ishaan1402/crack-seg`.

### 0.596 IoU — the Jaccard partner of 0.747

IoU and Dice are deterministically linked for the same prediction set:
`IoU = Dice / (2 − Dice) = 0.747 / 1.253 = 0.5962`. So 0.596 is exactly what
you get if someone computed IoU from the same predictions that gave 0.747.
No independent evidence for it exists on disk.

### 92% recall and 0.89 F1 — no backing, and internally inconsistent

For a binary segmentation mask, **Dice ≡ F1** (both equal
`2·TP / (2·TP + FP + FN)`). The claimed set cannot be one evaluation:

- Dice 0.747 + recall 0.92 → precision ≈ 0.63, F1 = **0.747** (not 0.89).
- Recall 0.92 + F1 0.89 → precision ≈ 0.86, Dice = **0.89**, IoU ≈ 0.80.

`rg` across the project, Pathfinder, and Downloads found no eval log, script,
or notebook containing 0.596 / 0.89 / 92%. The likely explanation is that
recall/F1 were computed at a different level (per-image macro F1, or
image-level "crack present in 92% of cracked images") or from a different
threshold/run, then mixed with pixel-level Dice/IoU.

### Reproduction on an available benchmark (DeepCrack test, 237 images)

The UAV test set is not on disk (no Kaggle cache; `dataset_split.zip` not
present), so the true test split cannot be reproduced here. As a
cross-domain proxy, the published weights were evaluated on the DeepCrack
test set with `scripts/verify_metrics.py`:

| Model | Inference | Thr | Global Dice | Global IoU | Recall | Precision | Macro Dice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Published narrow `[32,64,128,256]` | direct | 0.5 | 0.720 | 0.562 | 0.656 | 0.797 | 0.696 |
| Published narrow | sliding (448/0.5) | 0.5 | 0.718 | 0.560 | 0.652 | 0.800 | 0.695 |
| Original wide `[64,128,256,512]` | direct | 0.5 | **0.775** | **0.633** | 0.727 | 0.829 | 0.746 |

Neither model reproduces 0.747 / 0.596 / 0.92 / 0.89 on this benchmark. Note
the wide model is *stronger* cross-domain than the published narrow one.

---

## 2. Critical finding: the published weights cannot be loaded by this repo

`config/config.yaml` declares `features: [64, 128, 256, 512]` and
`src/models/unet.py` names the upsampling blocks `up_transposes.*` and the
head `final_conv.*`. The published `best_unet.pth` is a **narrow**
`[32, 64, 128, 256]` model with `up.*` / `final.*` key names. Verified:

```
$ python -c "import src.app"
RuntimeError: Error(s) in loading state_dict for UNet:
    Missing key(s): up_transposes.0.weight ... final_conv.bias
    Unexpected key(s): up.0.weight ... final.bias
    size mismatch: encoder.0.conv.0.weight [32,3,3,3] vs [64,3,3,3] ...
```

The FastAPI server and the Docker build therefore **crash at startup** with
the currently published checkpoint. The checkpoint loads cleanly only into a
`[32,64,128,256]` UNet after mapping `up.* → up_transposes.*` and
`final.* → final_conv.*` (the harness in `scripts/verify_metrics.py` does
this automatically).

The unit tests never caught this because they only exercise the sliding
window with dummy models, never the real checkpoint.

---

## 3. Is inference optimal?

### What is correct

- Normalization matches training (ImageNet mean/std, RGB).
- Gaussian-weighted blending at 448 px / 50% overlap removes seams; verified
  numerically (sliding ≈ direct within 0.002 Dice).
- Threshold semantics match training (`probs > threshold`).
- Padding path for sub-patch images is sound (`BORDER_REFLECT` + crop).
- Example image check: `input/example_1.jpeg` → 4.1% crack-area ratio,
  0.45 s direct / 0.72 s sliding on CPU.

### What can be improved

1. **Batch patches**: `SlidingWindowPredictor` runs one forward pass per
   patch. Batching all patches into a single tensor forward would cut GPU
   launch overhead substantially (and CPU time somewhat). Use
   `torch.inference_mode()` instead of `no_grad`.
2. **Direct path for small images**: when the image is smaller than the
   patch size, a single full-image forward pass (the model is fully
   convolutional) is cheaper and avoids padded-border downweighting. Add it
   as the default for `max(h, w) <= patch_size`.
3. **Safe checkpoint loading**: `torch.load(..., weights_only=True)` —
   also a security fix (already flagged in `things_to_fix.md`).
4. **TTA**: horizontal/vertical flip averaging typically adds +1–2 Dice
   points for free at inference time.
5. **Threshold**: 0.5 is fine on DeepCrack (Dice is flat between 0.5–0.7);
   keep it configurable (the API already exposes it).

---

## 4. Is the model the best at this job?

- The HPO campaign (narrow/wide UNet, ResNet34/50 encoders, lr, batch,
  loss-weight) is a sound idea, but its conclusions rest on a **sequential
  validation split** and no recorded held-out test evaluation. Best recorded
  val Dice: narrow UNet 0.747 > ResNet34 0.745 > ResNet50 0.717 >
  fixed UNet 0.711.
- On DeepCrack (cross-domain), the **original wide UNet
  (`~/Downloads/best_unet.pth`, 118 MB) is better** than the published narrow
  one (0.775 vs 0.720 global Dice). The narrow model's 0.747 val score does
  not transfer as well out of distribution.
- The v2 checkpoint was saved from the final epoch (19), not the best epoch
  (18, 0.7532). Minor, but the reported number is the weaker of the two.
- Missing for a defensible "best" claim: reproducible train/val/test splits,
  a fixed evaluation protocol, and test-set numbers for the final model.

---

## 5. Recommended next steps

1. **Fix the loader**: auto-detect width and remap `up/final` keys (harness
   already does this); update `config/config.yaml` or the loader so the app
   starts with the published weights.
2. **Get the real test set** (`dataset_split.zip` / Kaggle
   `ziya07/uav-based-crack-detection-dataset`) and evaluate both candidate
   checkpoints (narrow published vs wide original) on the 48-image test
   split with a fixed protocol. This produces the numbers you can actually
   publish.
3. **Apply the cheap inference wins** (batching, direct path, TTA,
   `weights_only=True`).
4. Only then update the model card and build the HF Space — with
   test-set-verified numbers, not HPO validation scores.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_metrics.py \
  --checkpoint checkpoints/best_unet.pth \
  --images input/DeepCrack/test_img --masks input/DeepCrack/test_lab \
  --mode both --thresholds 0.3 0.4 0.5 0.6 0.7
```
