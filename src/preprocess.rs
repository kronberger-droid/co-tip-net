use std::path::Path;

use image::imageops::FilterType;

/// Load a 16x16 grayscale image, flip vertically, and standardize per-image.
/// Returns 256 floats in row-major order (H=16, W=16).
pub fn load_normal_image(path: &Path) -> Vec<f32> {
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
