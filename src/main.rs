mod batcher;
mod dataset;
mod detect;
mod model;
mod preprocess;
mod train;

use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
use burn::{Tensor, prelude::Backend};
use burn_store::{ModuleSnapshot, PytorchStore};
use clap::{Parser, Subcommand, ValueEnum};

use crate::model::CoTipNet;
use crate::preprocess::load_normal_image;
use crate::train::FreezeStrategy;

type B = NdArray;
type TrainB = Autodiff<NdArray>;
type Device = <B as Backend>::Device;

#[derive(Parser)]
#[command(name = "co-tip-net", about = "CO-tip quality classifier for AFM")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Classify tip images as good/bad
    Classify {
        /// Path to a single PNG or a directory of PNGs
        path: PathBuf,

        /// Path to model file (.pt for PyTorch, .mpk for Burn-native)
        #[arg(long)]
        model: PathBuf,
    },

    /// Extract defect crops from a large scan image
    Extract {
        /// Path to input scan image (PNG)
        input: PathBuf,

        /// Output directory for cropped patches
        #[arg(long, default_value = "crops")]
        output: PathBuf,

        /// Crop size in pixels (before resize to 16x16)
        #[arg(long, default_value_t = 40)]
        crop_size: u32,

        /// Radius for local contrast computation (pixels)
        #[arg(long, default_value_t = 20)]
        contrast_radius: usize,

        /// Minimum contrast threshold for detection
        #[arg(long, default_value_t = 10.0)]
        min_contrast: f32,

        /// Minimum isotropy ratio (0.0–1.0). Higher = stricter circular shape filter.
        #[arg(long, default_value_t = 0.3)]
        min_isotropy: f32,
    },

    /// Train a new model or fine-tune from pretrained weights
    Train {
        /// Path to dataset directory (must contain train/ and valid/ subdirs)
        #[arg(long)]
        data: PathBuf,

        /// Number of training epochs
        #[arg(long, default_value_t = 29)]
        epochs: usize,

        /// Path to pretrained weights (.pt) for transfer learning
        #[arg(long)]
        pretrained: Option<PathBuf>,

        /// Layer freezing strategy for fine-tuning
        #[arg(long, value_enum, default_value_t = Freeze::None)]
        freeze: Freeze,
    },
}

#[derive(Clone, ValueEnum)]
enum Freeze {
    /// Train all layers
    None,
    /// Freeze conv1 + conv2
    EarlyConv,
    /// Freeze all conv layers
    AllConv,
}

impl From<Freeze> for FreezeStrategy {
    fn from(f: Freeze) -> Self {
        match f {
            Freeze::None => FreezeStrategy::None,
            Freeze::EarlyConv => FreezeStrategy::EarlyConv,
            Freeze::AllConv => FreezeStrategy::AllConv,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let device: Device = Default::default();

    match cli.command {
        Command::Classify { path, model } => {
            run_classify(&path, &model, &device);
        }

        Command::Extract {
            input,
            output,
            crop_size,
            contrast_radius,
            min_contrast,
            min_isotropy,
        } => {
            let image = image::open(&input)
                .unwrap_or_else(|e| panic!("Failed to open {}: {e}", input.display()))
                .into_luma8();
            detect::extract_defects(
                &image,
                crop_size,
                contrast_radius,
                min_contrast,
                min_isotropy,
                &output,
            );
        }

        Command::Train {
            data,
            epochs,
            pretrained,
            freeze,
        } => {
            train::train::<TrainB>(&data, &device, epochs, pretrained.as_deref(), freeze.into());
        }
    }
}

fn run_classify(path: &Path, model_path: &Path, device: &Device) {
    let mut model = CoTipNet::<B>::init(device);

    match model_path.extension().and_then(|e| e.to_str()) {
        Some("pt") => {
            let mut store = PytorchStore::from_file(model_path);
            model.load_from(&mut store).expect("Failed to load model");
        }
        Some("mpk") => {
            todo!("Load Burn-native .mpk weights — available after training a model");
        }
        _ => {
            panic!(
                "Unknown model format: {}. Expected .pt or .mpk",
                model_path.display()
            );
        }
    }

    if path.is_dir() {
        classify_dir(&model, path, device);
    } else {
        let prob = classify(&model, path, device);
        let label = if prob > 0.5 { "good" } else { "bad" };
        println!("{}: {:.3} ({})", path.display(), prob, label);
    }
}

fn classify(model: &CoTipNet<B>, path: &Path, device: &Device) -> f32 {
    let normalized = load_normal_image(path);
    let tensor = Tensor::<B, 1>::from_floats(normalized.as_slice(), device).reshape([1, 1, 16, 16]);
    model.forward(tensor).into_scalar()
}

fn classify_dir(model: &CoTipNet<B>, dir: &Path, device: &Device) {
    for entry in fs::read_dir(dir).expect("Failed to read dir") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "png") {
            let prob = classify(model, &path, device);
            let label = if prob > 0.5 { "good" } else { "bad" };
            println!("{}: {:.3} ({})", path.display(), prob, label);
        }
    }
}
