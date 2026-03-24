# co-tip-net

Burn (Rust deep learning framework) inference pipeline for a pretrained CO-tip quality classifier used in automated AFM tip preparation.

## Goal

Reimplement the inference path of a Keras/TF 1.12 CNN in Rust using **Burn** with the **LibTorch (`tch`)** backend. The original project lives in `../Auto-CO-AFM/`.

## Model architecture (2 157 params, binary classifier)

```
Input: (batch, 1, 16, 16)  — single-channel grayscale, channels-first

Conv2d(1→4, 3×3, valid)  → LeakyReLU(α=0.1)
Conv2d(4→4, 3×3, valid)  → LeakyReLU(α=0.1)
AvgPool2d(2×2, stride 2)
Conv2d(4→8, 3×3, valid)  → LeakyReLU(α=0.1)
Conv2d(8→8, 3×3, valid)  → LeakyReLU(α=0.1)
Flatten → 32
Linear(32→32)            → LeakyReLU(α=0.1)
Linear(32→1)             → Sigmoid

Output: probability ∈ [0,1]  (1 = good CO tip, 0 = bad)
```

Dropout / SpatialDropout layers are training-only — **do not implement**.

## Weight conversion pipeline

Pretrained weights are in `pretrained_weights/model.h5` (Keras HDF5).
Burn cannot read HDF5 directly.

**Conversion path:** H5 → PyTorch state_dict `.pt` → Burn `PyTorchFileRecorder`

Write a Python helper script (`convert_weights.py`) that:
1. Reads the H5 with `h5py` (no TF dependency needed)
2. Transposes conv kernels: Keras `(H,W,Cin,Cout)` → PyTorch `(Cout,Cin,H,W)`
3. Transposes dense kernels: Keras `(in,out)` → PyTorch `(out,in)`
4. Saves a `state_dict` whose keys match the Burn model field names:
   `conv1.weight`, `conv1.bias`, …, `linear1.weight`, `linear1.bias`, `linear2.weight`, `linear2.bias`

The H5 also stores optimizer state (Adam) and full model/training config as root attrs — keep the original H5 file around for provenance.

## Preprocessing (must match original exactly)

1. Load PNG, resize to 16×16 grayscale
2. Flip vertically (`np.flipud` equivalent)
3. Expand to shape `(1, 1, 16, 16)` — batch dim + channel dim, channels-first
4. Per-image standardisation: `x = (x - mean) / std`

## Crate dependencies

| Crate | Purpose |
|-------|---------|
| `burn` (features: `tch`) | Core framework + LibTorch backend |
| `burn-import` (feature: `pytorch`) | `PyTorchFileRecorder` for `.pt` loading |
| `image` | PNG loading & resize |

## Project layout

```
src/
  main.rs      — CLI entry: parse args, load image(s), run inference, print result
  model.rs     — #[derive(Module)] CoTipNet + CoTipNetConfig, forward()
```

## Implementation notes

- Keep the model generic over `B: Backend` so backends are swappable.
- Use `burn::tensor::activation::{leaky_relu, sigmoid}` — no need for module wrappers on parameter-free activations.
- `AvgPool2d` has no parameters — include in struct but it won't affect weight loading.
- For the CLI, accept a path to a single PNG or a directory of PNGs. Print per-image probability and good/bad classification (threshold 0.5).
- Use the context7 MCP tool to look up Burn API docs if unsure about any type signatures.
