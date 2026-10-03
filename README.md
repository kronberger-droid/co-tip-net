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

Crops are named after the scan, `<scan>_0000.png`, `<scan>_0001.png`, …, so several scans can be extracted into one directory and every crop stays traceable to its source.

The extraction pipeline: line-by-line median leveling, local contrast detection, isotropy filtering, center-of-mass refinement, and cropping.

Alternatively, flood the leveled scan below the background and classify each connected region as valid, too small, too large, elongated or close to edge (after Schnorrenberg et al., Uni Osnabrück):

```sh
co-tip-net extract scan.png --method flood --output crops/ --debug \
  --flood-level 3 \
  --min-area 25 --max-area 400
```

The water level is in units of the robust noise σ (1.4826 · MAD). With `--debug`, the per-class counts, the region area distribution and a class-colored overlay (`<scan>_debug_flood.png`) are written to help tune the area limits for a given pixel scale.

A third method is a scale-constrained determinant-of-Hessian blob detector, the detector half of SURF, as used for the CNN cutouts on the same poster:

```sh
co-tip-net extract scan.png --method doh --output crops/ --debug \
  --doh-level 4 \
  --min-sigma 2.5 --max-sigma 6.7
```

Only dark blobs whose response peaks between `--min-sigma` and `--max-sigma` are kept. The strength is scaled to the depth of a matched Gaussian dip, so `--doh-level` is in noise σ like the flood level. `--debug` writes `<scan>_debug_doh.png` with a circle of radius 2σ per detection.

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
