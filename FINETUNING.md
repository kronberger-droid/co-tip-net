# Fine-tuning for O-terminated and Cu tips

## Context

The current model is a binary classifier (2,157 params) trained to distinguish good vs bad CO-functionalized AFM tips on 16x16 grayscale images. The goal is to extend this to also classify oxygen-terminated (O) and bare copper (Cu) tips.

Training will be implemented in Burn (Rust) — no Python dependency beyond the initial H5 weight conversion.

## Dataset structure

Collect labeled 16x16 grayscale PNGs:
```
datasets/
├── co/          (existing, ~346 images)
│   ├── train/{goods,bads}/
│   ├── valid/{goods,bads}/
│   └── test2/{goods,bads}/
├── oxygen/      (new)
│   ├── train/{goods,bads}/
│   ├── valid/{goods,bads}/
│   └── test/{goods,bads}/
└── copper/      (new)
    ├── train/{goods,bads}/
    ├── valid/{goods,bads}/
    └── test/{goods,bads}/
```

## Approach A: Separate binary classifiers (start here)

Train one binary model per tip type, reusing the same architecture.

### Burn training pipeline components

You need to implement these pieces in Rust:

#### 1. Autodiff backend

Wrap the inference backend with `Autodiff` for gradient computation:
```rust
use burn::backend::Autodiff;
type TrainBackend = Autodiff<NdArray>;
```

#### 2. Dataset and DataLoader

Implement `burn::data::dataset::Dataset` trait for loading images:
```rust
struct TipImageDataset {
    items: Vec<(PathBuf, f32)>,  // (path, label: 0.0 or 1.0)
}

impl Dataset<TipImageItem> for TipImageDataset {
    fn get(&self, index: usize) -> Option<TipImageItem>;
    fn len(&self) -> usize;
}
```

Then wrap with `DataLoaderBuilder` for batching and shuffling.

#### 3. Data augmentation

Implement as tensor operations during batch preparation:
- 90/180/270 degree rotations: `tensor.swap_dims(H, W)` + `tensor.flip([dim])`
- Vertical flip: `tensor.flip([H_dim])`
- Per-image standardization: `(x - mean) / std`

The original training used 4 rotations x 2 flips = 8x augmentation per image.

#### 4. Training step

Implement `TrainStep` and `ValidStep` traits on the model:
```rust
impl<B: AutodiffBackend> TrainStep<TipBatch<B>, ClassificationOutput<B>> for CoTipNet<B> {
    fn step(&self, batch: TipBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch.images);
        let loss = BinaryCrossEntropyLoss::new().forward(output, batch.targets);
        TrainOutput::new(self, loss.backward(), ClassificationOutput { loss, output })
    }
}
```

#### 5. Training loop with LearnerBuilder

```rust
let learner = LearnerBuilder::new("artifacts/")
    .devices(vec![device])
    .num_epochs(29)
    .build(model, AdamConfig::new().init(), lr_scheduler);
let trained = learner.fit(train_loader, valid_loader);
```

#### 6. Transfer learning / freezing layers

For fine-tuning with pretrained CO weights:
- Load the pretrained `.pt` weights into the model
- Burn's `Module` trait provides `.no_grad()` to freeze parameters
- Freeze early layers for small datasets:
  - `conv1` + `conv2`: 188 params (low-level features, likely transferable)
  - Only train `conv3`, `conv4`, `linear1`, `linear2`: 1,969 params
  - With <50 images, freeze all conv, only train `linear1` + `linear2`: 1,089 params

#### 7. Save trained weights

```rust
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
trained_model.save_file("weights/oxygen_model", &recorder)?;
```

Note: models trained in Burn save as `.mpk` (MessagePack), not `.pt`. No NHWC permutation issue since training and inference both use NCHW.

### CLI integration

Accept a `--model` flag:
```
co-tip-net --model co image.png         # uses pretrained CO weights (.pt)
co-tip-net --model oxygen image.png     # uses Burn-trained O weights (.mpk)
co-tip-net --train --data datasets/oxygen --epochs 29  # train new model
```

### Implementation order

1. **Data loading** — `TipImageDataset` struct, load PNGs, apply preprocessing
2. **Augmentation** — rotations and flips as tensor ops in a `Batcher` impl
3. **Training loop** — `TrainStep`, `ValidStep`, loss function, optimizer
4. **Transfer learning** — load pretrained weights, freeze layers
5. **CLI** — `--train` / `--model` flags with `clap`

## Approach B: Multi-class classifier (later extension)

Replace the binary output with a multi-class head:

```
Current:   Linear(32 -> 1) + Sigmoid    -> P(good CO tip)
New:       Linear(32 -> N) + Softmax    -> P(class_0), ..., P(class_N-1)
```

Changes:
- `linear2: LinearConfig::new(32, num_classes)` in model
- `CrossEntropyLoss` instead of `BinaryCrossEntropyLoss`
- Output interpretation: `argmax` over classes
- ~32 extra params per class

## Preprocessing (must match for all models)

1. Load PNG, convert to grayscale
2. Resize to 16x16 (Lanczos interpolation)
3. Flip vertically
4. Per-image standardization: `x = (x - mean) / std`
5. Shape to `(1, 1, 16, 16)` — batch + channel dims

Identical for all tip types. The `load_normal_image` function does not change.

## Weight conversion note

The NHWC flatten permutation in `py/h5-to-pt.py` only applies to models trained in Keras. Models trained in Burn use NCHW natively — no permutation needed. This means Burn-trained `.mpk` weights load cleanly without any conversion step.
