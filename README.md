# co-tip-net

Rust inference and training pipeline for a CO/O/Cu tip quality classifier used in automated AFM/STM tip preparation. Built on the [Burn](https://burn.dev/) deep learning framework.

## Goal

Classify functionalized AFM/STM tips (CO-terminated, oxygen-terminated, bare copper) as good or bad from 16x16 grayscale scan patches. The original model (2,157 params) was trained in Keras/TF 1.12 — this project reimplements inference and adds training in pure Rust.

## Model architecture

```
Input: (batch, 1, 16, 16) — single-channel grayscale

Conv2d(1->4, 3x3) + LeakyReLU(0.1)
Conv2d(4->4, 3x3) + LeakyReLU(0.1)
AvgPool2d(2x2)
Conv2d(4->8, 3x3) + LeakyReLU(0.1)
Conv2d(8->8, 3x3) + LeakyReLU(0.1)
Flatten -> 32
Linear(32->32) + LeakyReLU(0.1)
Linear(32->1) + Sigmoid

Output: P(good tip) in [0, 1]
```

## Usage

### Classify images

```sh
# Single image
co-tip-net classify --model pretrained_weights/model.pt image.png

# Directory of PNGs
co-tip-net classify --model pretrained_weights/model.pt datasets/co/valid/goods/
```

### Extract defect crops from large scans

```sh
# Extract individual defect patches from a full-area scan
co-tip-net extract scan.png --output crops/ --crop-size 40

# Tune detection parameters
co-tip-net extract scan.png --output crops/ \
  --min-contrast 15 \
  --min-isotropy 0.6 \
  --contrast-radius 20
```

The extraction pipeline: line-by-line median leveling, local contrast detection, isotropy filtering, center-of-mass refinement, and cropping.

### Train / fine-tune

```sh
# Train from scratch
co-tip-net train --data datasets/oxygen --epochs 29

# Fine-tune from pretrained CO weights, freezing early conv layers
co-tip-net train --data datasets/oxygen \
  --pretrained pretrained_weights/model.pt \
  --freeze early-conv \
  --epochs 29

# Freeze all conv layers (for very small datasets, <50 images per class)
co-tip-net train --data datasets/oxygen \
  --pretrained pretrained_weights/model.pt \
  --freeze all-conv
```

Dataset directory structure:
```
datasets/oxygen/
  train/{goods,bads}/*.png
  valid/{goods,bads}/*.png
```

## Progress

- [x] Inference pipeline with pretrained CO-tip weights (.pt)
- [x] Weight conversion from Keras H5 to PyTorch format
- [x] CLI with subcommands (classify, train, extract)
- [x] Training pipeline (dataset, batcher, D4 augmentation, TrainStep/InferenceStep)
- [x] Transfer learning with layer freezing (none / early-conv / all-conv)
- [x] Defect extraction from large scans (leveling, detection, cropping)
- [ ] Improve defect detector (step edge / oxide row rejection)
- [ ] Train O-tip and Cu-tip classifiers
- [ ] Load Burn-native .mpk weights in classify
- [ ] Multi-class classifier (Approach B)
- [ ] Live inference during scanning

## Dependencies

| Crate | Purpose |
|-------|---------|
| `burn` (ndarray, autodiff, train) | ML framework |
| `burn-store` | PyTorch weight loading |
| `image` | PNG loading, resize, crop |
| `clap` | CLI argument parsing |
