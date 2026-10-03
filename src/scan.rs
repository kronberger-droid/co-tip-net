//! Scan input for extraction: a row-major f32 height map.
//!
//! `.sxm` files are read natively (Z forward, in pm) and resampled to a
//! common pixel scale, so crop and blob sizes in pixels mean the same
//! physical size across scans of different ranges. Other images are read
//! as grayscale in 0–255 and used at their native resolution, since they
//! carry no scale.

use std::error::Error;
use std::path::Path;

use image::imageops::{self, FilterType};
use image::{ImageBuffer, Luma};

use crate::sxm::Sxm;

/// Default pixel scale: 30 nm over 512 px, the most common scan setting.
pub const DEFAULT_NM_PER_PX: f32 = 30.0 / 512.0;

pub struct Scan {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Scan {
    /// Load `.sxm` (resampled to `nm_per_px`) or any image format `image`
    /// reads (native resolution). A scan coarser than `nm_per_px` by more
    /// than `max_upsample` is refused: interpolation cannot recover a CO
    /// that spans only a few original pixels.
    pub fn open(path: &Path, nm_per_px: f32, max_upsample: f32) -> Result<Scan, Box<dyn Error>> {
        let is_sxm = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("sxm"));
        if is_sxm {
            return Self::open_sxm(path, nm_per_px, max_upsample);
        }

        let image = image::open(path)?.to_luma32f();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let data = image.into_raw().into_iter().map(|v| v * 255.0).collect();
        Ok(Scan {
            data,
            width,
            height,
        })
    }

    fn open_sxm(path: &Path, nm_per_px: f32, max_upsample: f32) -> Result<Scan, Box<dyn Error>> {
        let frame = Sxm::open(path)?.frame("Z", true)?.complete_rows();
        if frame.height < 2 {
            return Err("no complete scan rows".into());
        }
        let native = frame.range_nm.0 / frame.width as f32;
        if native > nm_per_px * max_upsample {
            return Err(format!(
                "{native:.3} nm/px is more than {max_upsample}x coarser than {nm_per_px:.3} nm/px"
            )
            .into());
        }

        // Metres to picometres keeps values well above the small absolute
        // epsilons in the detectors.
        let data: Vec<f32> = frame.data.iter().map(|v| v * 1e12).collect();
        let width = (frame.range_nm.0 / nm_per_px).round().max(1.0) as usize;
        let height = (frame.range_nm.1 / nm_per_px).round().max(1.0) as usize;

        Ok(Scan {
            data: resample(&data, frame.width, frame.height, width, height),
            width,
            height,
        })
    }
}

/// Resize with a triangle filter. `image` treats f32 pixels as 0..1 and
/// clamps to that range, thus values are mapped into it and back.
fn resample(data: &[f32], w: usize, h: usize, new_w: usize, new_h: usize) -> Vec<f32> {
    if (w, h) == (new_w, new_h) {
        return data.to_vec();
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(f32::MIN_POSITIVE);

    let normalized: Vec<f32> = data.iter().map(|v| (v - min) / range).collect();
    let buf = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, normalized)
        .expect("buffer size matches dimensions");
    imageops::resize(&buf, new_w as u32, new_h as u32, FilterType::Triangle)
        .into_raw()
        .into_iter()
        .map(|v| v * range + min)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_values_outside_unit_range() {
        // A tilted plane in pm: resampling must not clamp to 0..1.
        let (w, h) = (64, 32);
        let data: Vec<f32> = (0..w * h).map(|i| 1000.0 + (i % w) as f32 * 10.0).collect();
        let out = resample(&data, w, h, 128, 64);
        assert_eq!(out.len(), 128 * 64);
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!((min - 1000.0).abs() < 10.0, "min {min}");
        assert!((max - 1630.0).abs() < 10.0, "max {max}");
    }
}
