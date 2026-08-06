---
title: Bridge Crack Detection Demo
emoji: 🌉
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
python_version: "3.10"
hardware: zero-gpu
license: apache-2.0
short_description: Try the crack-seg U-Net v3 on your own bridge photos
tags:
  - pytorch
  - unet
  - image-segmentation
  - bridge-inspection
  - uav
---

# Bridge Crack Detection — U-Net v3

Upload a bridge photo and get a pixel-level crack mask with the cracked-area
percentage. The demo serves the `unet_v3.pth` checkpoint of
[ishaan1402/crack-seg](https://huggingface.co/ishaan1402/crack-seg): a narrow
U-Net `[32, 64, 128, 256]` with squeeze-and-excitation blocks (~7.8M params)
trained on UAV and multi-source bridge imagery.

## How it works

Images larger than the 448 px training patch are split into overlapping tiles
(50% overlap), run through the model in batches, and blended back with a
Gaussian weight map so seams are invisible. The overlay marks detected cracks
in green.

## Limits

- Longest side up to 8192 px; decoded size up to ~50 MB.
- This Space runs on free ZeroGPU hardware, so daily GPU time is shared:
  about 5 minutes per day for logged-in free accounts and 2 minutes for
  anonymous visitors. Large images consume more time than small ones.

## Source

Code: [Ishaan1402/crack-seg](https://github.com/Ishaan1402/crack-seg) · Model
card: [ishaan1402/crack-seg](https://huggingface.co/ishaan1402/crack-seg)
