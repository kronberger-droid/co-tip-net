mod batcher;
mod dataset;
mod detect;
mod doh;
mod flood;
mod model;
mod preprocess;
mod scan;
mod steps;
mod sxm;
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
        /// Path to input scan: Nanonis .sxm (Z forward) or a grayscale image
        input: PathBuf,

        /// Pixel scale .sxm scans are resampled to, so crop and blob sizes in
        /// pixels mean the same physical size across scan ranges. Ignored for images.
        #[arg(long, default_value_t = scan::DEFAULT_NM_PER_PX)]
        nm_per_px: f32,

        /// Skip .sxm scans that would need more than this much upsampling
        #[arg(long, default_value_t = 2.0)]
        max_upsample: f32,

        /// Output directory for cropped patches
        #[arg(long, default_value = "crops")]
        output: PathBuf,

        /// Detection method
        #[arg(long, value_enum, default_value_t = Method::Peaks)]
        method: Method,

        /// Look for dark depressions or bright protrusions
        #[arg(long, value_enum, default_value_t = Polarity::Both)]
        polarity: Polarity,

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

        /// [flood, doh] Box radius of the background subtracted before detection,
        /// in pixels. Smaller keeps step edges thin [default: crop_size/3]
        #[arg(long)]
        bg_radius: Option<usize>,

        /// [flood, doh] Step-edge mask threshold in robust σ of the height gradient.
        /// Detections on long strong-gradient lines are rejected. 0 disables
        #[arg(long, default_value_t = 3.0)]
        step_level: f32,

        /// [flood] Water level in units of the robust noise σ below the background
        #[arg(long, default_value_t = 2.0)]
        flood_level: f32,

        /// [flood] Minimum region area in pixels [default: (crop_size/8)²]
        #[arg(long)]
        min_area: Option<usize>,

        /// [flood] Maximum region area in pixels [default: (crop_size/2)²]
        #[arg(long)]
        max_area: Option<usize>,

        /// [doh] Detection threshold in units of the robust noise σ
        #[arg(long, default_value_t = 2.0)]
        doh_level: f32,

        /// [doh] Smallest blob σ accepted as a CO, in pixels [default: crop_size/20]
        #[arg(long)]
        min_sigma: Option<f32>,

        /// [doh] Largest blob σ accepted as a CO, in pixels [default: crop_size/5]
        #[arg(long)]
        max_sigma: Option<f32>,

        /// [doh] Minimum rotational symmetry (0-1): share of the variation around
        /// a blob explained by its radial profile
        #[arg(long, default_value_t = 0.4)]
        min_symmetry: f32,

        /// [doh] Reject a blob with this many neighbors of at least half its
        /// strength within half a crop (lattice spots, not isolated COs)
        #[arg(long, default_value_t = 2)]
        max_neighbors: usize,

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

#[derive(Clone, PartialEq, ValueEnum)]
enum Polarity {
    /// Depressions below the background
    Dark,
    /// Protrusions above the background
    Bright,
    /// Both, as separate passes; the contrast of a CO depends on the tip
    Both,
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
            nm_per_px,
            max_upsample,
            output,
            method,
            polarity,
            crop_size,
            contrast_radius,
            min_contrast,
            min_isotropy,
            bg_radius,
            step_level,
            flood_level,
            min_area,
            max_area,
            doh_level,
            min_sigma,
            max_sigma,
            num_scales,
            min_symmetry,
            max_neighbors,
            debug,
        } => {
            let loaded = scan::Scan::open(&input, nm_per_px, max_upsample).unwrap_or_else(|e| {
                eprintln!("Skipping {}: {e}", input.display());
                std::process::exit(1);
            });
            // Crops are named after the scan so several scans can share one
            // output directory; with both polarities, after the pass too.
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "scan".into());
            let passes = match polarity {
                Polarity::Dark => vec![(false, stem)],
                Polarity::Bright => vec![(true, stem)],
                Polarity::Both => vec![
                    (false, format!("{stem}_dark")),
                    (true, format!("{stem}_bright")),
                ],
            };
            let bg_radius = bg_radius.unwrap_or(crop_size as usize / 3);
            for (invert, prefix) in passes {
                let mut scan = loaded.clone();
                if invert {
                    scan.invert();
                }
                match method {
                    Method::Peaks => detect::extract_defects(
                        &scan,
                        crop_size,
                        contrast_radius,
                        min_contrast,
                        min_isotropy,
                        &output,
                        &prefix,
                        debug,
                    ),
                    Method::Flood => {
                        let params = flood::FloodParams {
                            crop_size,
                            bg_radius,
                            step_level,
                            level_sigma: flood_level,
                            min_area: min_area
                                .unwrap_or_else(|| flood::FloodParams::default_min_area(crop_size)),
                            max_area: max_area
                                .unwrap_or_else(|| flood::FloodParams::default_max_area(crop_size)),
                            min_isotropy,
                        };
                        flood::extract_defects_flood(&scan, &params, &output, &prefix, debug);
                    }
                    Method::Doh => {
                        let params = doh::DohParams {
                            crop_size,
                            bg_radius,
                            step_level,
                            min_sigma: min_sigma
                                .unwrap_or_else(|| doh::DohParams::default_min_sigma(crop_size)),
                            max_sigma: max_sigma
                                .unwrap_or_else(|| doh::DohParams::default_max_sigma(crop_size)),
                            num_scales,
                            level_sigma: doh_level,
                            min_isotropy,
                            min_symmetry,
                            max_neighbors,
                        };
                        doh::extract_defects_doh(&scan, &params, &output, &prefix, debug);
                    }
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
