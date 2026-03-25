use std::path::Path;

use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::optim::lr_scheduler::constant::ConstantLr;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use burn_store::{ModuleSnapshot, PytorchStore};

use crate::batcher::{TipBatch, TipBatcher};
use crate::dataset::TipImageDataset;
use crate::model::CoTipNet;

/// Controls which layers are frozen during fine-tuning.
pub enum FreezeStrategy {
    /// Train all layers from scratch (no freezing).
    None,
    /// Freeze conv1 + conv2 (low-level features). Train conv3, conv4, linear1, linear2.
    /// 1,969 trainable params. Good default for 50+ images per class.
    EarlyConv,
    /// Freeze all conv layers. Only train linear1 + linear2.
    /// 1,089 trainable params. Use with <50 images per class.
    AllConv,
}

impl<B: AutodiffBackend> TrainStep for CoTipNet<B> {
    type Input = TipBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.forward(batch.images);
        let target_int = batch.targets.int(); // (batch, 1) Int
        let targets_1d = target_int.clone().squeeze(); // (batch,) Int

        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), target_int);

        TrainOutput::new(
            self,
            loss.clone().backward(),
            ClassificationOutput::new(loss, output, targets_1d),
        )
    }
}

impl<B: Backend> InferenceStep for CoTipNet<B> {
    type Input = TipBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let output = self.forward(batch.images);
        let target_int = batch.targets.int();
        let target_1d = target_int.clone().squeeze();

        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), target_int);

        ClassificationOutput::new(loss, output, target_1d)
    }
}

pub fn train<B: AutodiffBackend>(
    data_dir: &Path,
    device: &B::Device,
    num_epochs: usize,
    pretrained_weights: Option<&Path>,
    freeze: FreezeStrategy,
) {
    let train_dataset = TipImageDataset::from_dir(&data_dir.join("train"));
    let valid_dataset = TipImageDataset::from_dir(&data_dir.join("valid"));

    println!(
        "Training: {} images ({} with augmentation), Validation: {} images",
        train_dataset.len(),
        train_dataset.len() * 8,
        valid_dataset.len(),
    );

    let train_batcher = TipBatcher::train();
    let valid_batcher = TipBatcher::valid();

    let train_loader = DataLoaderBuilder::<B, _, _>::new(train_batcher)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4)
        .build(train_dataset);

    let valid_loader = DataLoaderBuilder::<B::InnerBackend, _, _>::new(valid_batcher)
        .batch_size(32)
        .num_workers(4)
        .build(valid_dataset);

    // Initialize model and optionally load pretrained weights
    let mut model = CoTipNet::<B>::init(device);
    if let Some(weights_path) = pretrained_weights {
        let mut store = PytorchStore::from_file(weights_path);
        model
            .load_from(&mut store)
            .expect("Failed to load pretrained weights");
        println!("Loaded pretrained weights from {}", weights_path.display());
    }

    // Freeze layers according to strategy
    // TODO: Implement this — call .no_grad() on the appropriate fields.
    //
    // FreezeStrategy::None     → do nothing
    // FreezeStrategy::EarlyConv → freeze conv1 and conv2
    // FreezeStrategy::AllConv   → freeze conv1, conv2, conv3, and conv4
    //
    // Example:  model.conv1 = model.conv1.no_grad();
    //
    // Note: model fields are currently private. You'll need to make
    // the conv fields pub(crate) in model.rs first.
    match freeze {
        FreezeStrategy::None => {}
        FreezeStrategy::EarlyConv => {
            model.conv1 = model.conv1.no_grad();
            model.conv2 = model.conv2.no_grad();
        }
        FreezeStrategy::AllConv => {
            model.conv1 = model.conv1.no_grad();
            model.conv2 = model.conv2.no_grad();
            model.conv3 = model.conv3.no_grad();
            model.conv4 = model.conv4.no_grad();
        }
    }

    let optimizer = AdamConfig::new().init();
    let lr_scheduler = ConstantLr::new(1e-3);

    let learner = Learner::new(model, optimizer, lr_scheduler);

    let _result = SupervisedTraining::new("artifacts/", train_loader, valid_loader)
        .num_epochs(num_epochs)
        .metrics((LossMetric::<burn::backend::NdArray>::new(),))
        .summary()
        .launch(learner);

    println!("Training complete.");
}
