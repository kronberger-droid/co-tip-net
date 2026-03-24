mod model;

use std::fs;
use std::path::Path;

use burn_store::{ModuleSnapshot, PytorchStore};
use image::imageops::FilterType;

use crate::model::CoTipNet;
use burn::{Tensor, backend::NdArray};

type B = NdArray;
type Device = <B as burn::prelude::Backend>::Device;

fn main() {
    let device: Device = Default::default();

    let mut model = CoTipNet::<B>::init(&device);
    let mut store = PytorchStore::from_file("pretrained_weights/model.pt");
    model.load_from(&mut store).expect("Failed to load model");

    classify_dir(
        &model,
        Path::new("datasets/co/valid/goods"),
        "good",
        &device,
    );
    classify_dir(&model, Path::new("datasets/co/valid/bads"), "bad", &device);
}

fn classify(model: &CoTipNet<B>, path: &Path, device: &Device) -> f32 {
    let normalized = load_normal_image(path);
    let tensor = Tensor::<B, 1>::from_floats(normalized.as_slice(), device).reshape([1, 1, 16, 16]);
    model.forward(tensor).into_scalar()
}

fn classify_dir(model: &CoTipNet<B>, dir: &Path, label: &str, device: &Device) {
    let mut correct = 0;
    let mut total = 0;
    for entry in fs::read_dir(dir).expect("Failed to read dir") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "png") {
            let prob = classify(model, &path, device);
            let predicted = if prob > 0.5 { "good" } else { "bad" };
            total += 1;
            if predicted == label {
                correct += 1;
            }
        }
    }
    println!(
        "{label}: {correct}/{total} correct ({:.1}%)",
        correct as f64 / total as f64 * 100.0
    );
}

fn load_normal_image(path: &Path) -> Vec<f32> {
    let image = image::open(path)
        .expect("Failed to load image")
        .grayscale()
        .resize_exact(16, 16, FilterType::Lanczos3)
        .flipv()
        .into_luma8();

    let pixels: Vec<f32> = image.pixels().map(|p| p.0[0] as f32).collect();
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    let std = (pixels.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / pixels.len() as f32).sqrt();
    pixels.iter().map(|x| (x - mean) / std).collect()
}
