mod batcher;
mod dataset;
mod detect;
mod doh;
mod flood;
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

        /// Detection method
        #[arg(long, value_enum, default_value_t = Method::Peaks)]
        method: Method,

        /// Crop size in pixels (before resize to 16x16)
        #[arg(long, default_value_t = 40)]
        crop_size: u32,

        /// [peaks] Radius for local contrast computation (pixels)
        #[arg(long, default_value_t = 20)]
        contrast_radius: usize,

        /// [peaks] Minimum contrast threshold for detection
        #[arg(long, default_value_t = 10.0)]
        min_contrast: f32,

        /// Minimum isotropy ratio (0.0–1.0). Higher = stricter circular shape filter.
        #[arg(long, default_value_t = 0.3)]
        min_isotropy: f32,

        /// [flood] Water level in units of the robust noise σ below the background
        #[arg(long, default_value_t = 3.0)]
        flood_level: f32,

        /// [flood] Minimum region area in pixels [default: (crop_size/8)²]
        #[arg(long)]
        min_area: Option<usize>,

        /// [flood] Maximum region area in pixels [default: (crop_size/2)²]
        #[arg(long)]
        max_area: Option<usize>,

        /// [doh] Detection threshold in units of the robust noise σ
        #[arg(long, default_value_t = 4.0)]
        doh_level: f32,

        /// [doh] Smallest blob σ accepted as a CO, in pixels [default: crop_size/16]
        #[arg(long)]
        min_sigma: Option<f32>,

        /// [doh] Largest blob σ accepted as a CO, in pixels [default: crop_size/6]
        #[arg(long)]
        max_sigma: Option<f32>,

        /// [doh] Number of scales between min and max σ
        #[arg(long, default_value_t = 6)]
        num_scales: usize,

        /// Save intermediate debug images (leveled, contrast map, flood overlay) to output dir.
        #[arg(long, default_value_t = false)]
        debug: bool,
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
enum Method {
    /// Local contrast peaks filtered by shape heuristics
    Peaks,
    /// Flood below the background and classify connected regions by size and shape
    Flood,
    /// Scale-constrained determinant-of-Hessian blob detector (SURF's detector)
    Doh,
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
            method,
            crop_size,
            contrast_radius,
            min_contrast,
            min_isotropy,
            flood_level,
            min_area,
            max_area,
            doh_level,
            min_sigma,
            max_sigma,
            num_scales,
            debug,
        } => {
            let image = image::open(&input)
                .unwrap_or_else(|e| panic!("Failed to open {}: {e}", input.display()))
                .into_luma8();
            match method {
                Method::Peaks => detect::extract_defects(
                    &image,
                    crop_size,
                    contrast_radius,
                    min_contrast,
                    min_isotropy,
                    &output,
                    debug,
                ),
                Method::Flood => {
                    let params = flood::FloodParams {
                        crop_size,
                        level_sigma: flood_level,
                        min_area: min_area
                            .unwrap_or_else(|| flood::FloodParams::default_min_area(crop_size)),
                        max_area: max_area
                            .unwrap_or_else(|| flood::FloodParams::default_max_area(crop_size)),
                        min_isotropy,
                    };
                    flood::extract_defects_flood(&image, &params, &output, debug);
                }
                Method::Doh => {
                    let params = doh::DohParams {
                        crop_size,
                        min_sigma: min_sigma
                            .unwrap_or_else(|| doh::DohParams::default_min_sigma(crop_size)),
                        max_sigma: max_sigma
                            .unwrap_or_else(|| doh::DohParams::default_max_sigma(crop_size)),
                        num_scales,
                        level_sigma: doh_level,
                        min_isotropy,
                    };
                    doh::extract_defects_doh(&image, &params, &output, debug);
                }
            }
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
