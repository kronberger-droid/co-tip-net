//! Defect extraction with a scale-constrained determinant-of-Hessian (DoH)
//! blob detector, the detector half of SURF.
//!
//! Schnorrenberg et al. (Uni Osnabrück) use a "constrained SURF" to find CO
//! molecules for CNN cutouts. Their appearance there depends on tip quality
//! (sharp dip with ring, or a blurred smudge), which breaks a fixed water
//! level but not a second-derivative blob response. Since the image sizes
//! here are small, true Gaussian derivatives replace SURF's box filters.
//!
//! The scale constraint comes from the σ ladder: it spans
//! `[min_sigma, max_sigma]` plus one padding scale on each side. A blob whose
//! response peaks on a padding scale is outside the CO size range and is
//! classified as too small or too large. Ridges (oxide rows, step edges)
//! give a near-zero determinant and rarely respond at all.

use std::path::Path;

use image::{Rgb, RgbImage};

use crate::detect::{Defect, crop_and_save, level_line_median, level_robust_bg};
use crate::flood::RegionClass;
use crate::scan::Scan;
use crate::steps::{step_mask, unmasked_sigma};

/// Parameters for DoH extraction.
pub struct DohParams {
    /// Side length of the square crop around each valid blob (pixels).
    pub crop_size: u32,
    /// Box radius of the background subtracted before detection (pixels).
    pub bg_radius: usize,
    /// Step-edge mask threshold in robust σ of the gradient, `<= 0` disables.
    pub step_level: f32,
    /// Smallest blob σ accepted as a CO (pixels).
    pub min_sigma: f32,
    /// Largest blob σ accepted as a CO (pixels).
    pub max_sigma: f32,
    /// Number of scales between `min_sigma` and `max_sigma`, inclusive.
    pub num_scales: usize,
    /// Detection threshold in units of the robust noise σ. The blob strength
    /// is scaled so it equals the depth of a matched Gaussian dip, making
    /// this comparable to the flood water level.
    pub level_sigma: f32,
    /// Minimum `λ_min / λ_max` of the Hessian at the blob center, evaluated
    /// at `min_sigma`. Curvature ratio, thus stricter than the covariance
    /// ratio of `flood`.
    pub min_isotropy: f32,
}

impl DohParams {
    /// σ defaults scaled to the crop size, since px/nm differs per scan.
    pub fn default_min_sigma(crop_size: u32) -> f32 {
        crop_size as f32 / 16.0
    }

    pub fn default_max_sigma(crop_size: u32) -> f32 {
        crop_size as f32 / 6.0
    }

    /// Geometric σ ladder with one padding scale on each end.
    fn ladder(&self) -> Vec<f32> {
        let n = self.num_scales.max(2);
        let ratio = (self.max_sigma / self.min_sigma).powf(1.0 / (n - 1) as f32);
        (-1..=n as i32)
            .map(|i| self.min_sigma * ratio.powi(i))
            .collect()
    }
}

/// A local maximum of the DoH response in space and scale.
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: usize,
    pub y: usize,
    pub sigma: f32,
    /// `4 · sqrt(σ⁴ · det H)`, equal to the depth of a matched Gaussian dip.
    pub strength: f32,
    /// `λ_min / λ_max` of the Hessian at the keypoint, at `min_sigma`.
    pub isotropy: f32,
    pub class: RegionClass,
}

/// Sampled Gaussian and its first and second derivatives, truncated at 4σ.
fn gaussian_kernels(sigma: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let radius = (4.0 * sigma).ceil() as i32;
    let s2 = sigma * sigma;

    let mut g: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * s2)).exp())
        .collect();
    let norm: f32 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= norm);

    let g1: Vec<f32> = (-radius..=radius)
        .zip(&g)
        .map(|(i, &v)| -(i as f32) / s2 * v)
        .collect();

    let mut g2: Vec<f32> = (-radius..=radius)
        .zip(&g)
        .map(|(i, &v)| ((i * i) as f32 / s2 - 1.0) / s2 * v)
        .collect();
    // Sampling leaves a small DC term, which would turn background offsets
    // into curvature.
    let dc = g2.iter().sum::<f32>() / g2.len() as f32;
    g2.iter_mut().for_each(|v| *v -= dc);

    (g, g1, g2)
}

/// Reflect an out-of-range index back into `0..n` (`d c b a | a b c d`).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let period = 2 * n;
    let m = i.rem_euclid(period);
    (if m < n { m } else { period - 1 - m }) as usize
}

fn convolve_rows(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let r = (kernel.len() / 2) as isize;
    let mut dst = vec![0.0_f32; width * height];
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        for x in 0..width {
            dst[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(k, &w)| w * row[reflect(x as isize + k as isize - r, width)])
                .sum();
        }
    }
    dst
}

fn convolve_cols(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let r = (kernel.len() / 2) as isize;
    let mut dst = vec![0.0_f32; width * height];
    for y in 0..height {
        for (k, &w) in kernel.iter().enumerate() {
            let sy = reflect(y as isize + k as isize - r, height);
            let src_row = &src[sy * width..(sy + 1) * width];
            let dst_row = &mut dst[y * width..(y + 1) * width];
            for (d, &s) in dst_row.iter_mut().zip(src_row) {
                *d += w * s;
            }
        }
    }
    dst
}

/// Scale-normalized Hessian components (σ² Lxx, σ² Lyy, σ² Lxy).
fn hessian(img: &[f32], width: usize, height: usize, sigma: f32) -> [Vec<f32>; 3] {
    let (g, g1, g2) = gaussian_kernels(sigma);
    let s2 = sigma * sigma;
    let scale = |v: Vec<f32>| v.into_iter().map(|x| x * s2).collect::<Vec<_>>();

    let lxx = convolve_cols(&convolve_rows(img, width, height, &g2), width, height, &g);
    let lyy = convolve_cols(&convolve_rows(img, width, height, &g), width, height, &g2);
    let lxy = convolve_cols(&convolve_rows(img, width, height, &g1), width, height, &g1);
    [scale(lxx), scale(lyy), scale(lxy)]
}

/// Dark-blob strength and isotropy from normalized Hessian components.
/// Bright blobs (negative trace) and saddles (negative det) score zero.
fn blob_response(lxx: f32, lyy: f32, lxy: f32) -> (f32, f32) {
    let trace = lxx + lyy;
    let det = lxx * lyy - lxy * lxy;
    if trace <= 0.0 || det <= 0.0 {
        return (0.0, 0.0);
    }
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let isotropy = (trace - disc) / (trace + disc);
    (4.0 * det.sqrt(), isotropy)
}

/// Result of DoH detection: the leveled image, noise level, σ ladder and
/// every above-threshold keypoint with its class.
pub struct Detection {
    pub leveled: Vec<f32>,
    pub sigma: f32,
    pub ladder: Vec<f32>,
    pub keypoints: Vec<Keypoint>,
    pub step_mask: Vec<bool>,
}

/// Level the raw scan, build the DoH scale space and classify its maxima.
pub fn detect(pixels: &[f32], width: usize, height: usize, params: &DohParams) -> Detection {
    let line_leveled = level_line_median(pixels, width, height);
    let leveled = level_robust_bg(&line_leveled, width, height, params.bg_radius);
    let mask = step_mask(
        &line_leveled,
        width,
        height,
        params.step_level,
        params.crop_size as usize,
        params.crop_size as usize / 4,
    );
    // Steps would dominate the spread, thus noise comes from the rest.
    let noise = unmasked_sigma(&leveled, &mask);
    let threshold = params.level_sigma * noise;
    let ladder = params.ladder();

    let response_maps = |s: f32| -> (Vec<f32>, Vec<f32>) {
        let [lxx, lyy, lxy] = hessian(&leveled, width, height, s);
        (0..width * height)
            .map(|i| blob_response(lxx[i], lyy[i], lxy[i]))
            .unzip()
    };
    let responses: Vec<Vec<f32>> = ladder.iter().map(|&s| response_maps(s).0).collect();
    // Shape is judged at the finest accepted scale: smoothing at the blob's
    // own scale makes a short ridge look round.
    let (_, isotropy) = response_maps(ladder[1]);

    let last = ladder.len() - 1;
    let half = (params.crop_size / 2) as usize;
    let mut keypoints = Vec::new();

    for (si, strength) in responses.iter().enumerate() {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let v = strength[y * width + x];
                if v <= 0.0 || v < threshold {
                    continue;
                }

                let neighbors = si.saturating_sub(1)..=(si + 1).min(last);
                let is_max = neighbors.into_iter().all(|sj| {
                    let other = &responses[sj];
                    (y - 1..=y + 1).all(|ny| {
                        (x - 1..=x + 1).all(|nx| {
                            (sj == si && ny == y && nx == x) || other[ny * width + nx] <= v
                        })
                    })
                });
                if !is_max {
                    continue;
                }

                let iso = isotropy[y * width + x];
                let fits = x >= half && y >= half && x + half < width && y + half < height;
                let class = if mask[y * width + x] {
                    RegionClass::StepEdge
                } else if si == 0 {
                    RegionClass::TooSmall
                } else if si == last {
                    RegionClass::TooLarge
                } else if iso < params.min_isotropy {
                    RegionClass::Elongated
                } else if !fits {
                    RegionClass::CloseToEdge
                } else {
                    RegionClass::Valid
                };

                keypoints.push(Keypoint {
                    x,
                    y,
                    sigma: ladder[si],
                    strength: v,
                    isotropy: iso,
                    class,
                });
            }
        }
    }

    Detection {
        leveled,
        sigma: noise,
        ladder,
        keypoints: suppress_duplicates(keypoints),
        step_mask: mask,
    }
}

/// Drop weaker valid keypoints within 2σ of a stronger one, which can occur
/// on flat-bottomed blobs where neighboring pixels tie.
fn suppress_duplicates(mut keypoints: Vec<Keypoint>) -> Vec<Keypoint> {
    keypoints.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    let mut kept: Vec<Keypoint> = Vec::new();
    for kp in keypoints {
        let duplicate = kp.class == RegionClass::Valid
            && kept.iter().any(|k| {
                let dx = k.x as f32 - kp.x as f32;
                let dy = k.y as f32 - kp.y as f32;
                k.class == RegionClass::Valid && (dx * dx + dy * dy).sqrt() < 2.0 * k.sigma
            });
        if !duplicate {
            kept.push(kp);
        }
    }
    kept
}

/// Full extraction pipeline: level → DoH scale space → classify → crop.
///
/// If `debug` is true, prints noise σ, the σ ladder, per-class counts and
/// the scale and strength distribution of valid blobs, and saves
/// `<prefix>_debug_doh.png`: the leveled scan with a circle of radius 2σ per
/// keypoint, colored by class, and the step mask tinted yellow.
pub fn extract_defects_doh(
    scan: &Scan,
    params: &DohParams,
    output_dir: &Path,
    prefix: &str,
    debug: bool,
) {
    let (w, h) = (scan.width, scan.height);
    let det = detect(&scan.data, w, h, params);

    let defects: Vec<Defect> = det
        .keypoints
        .iter()
        .filter(|k| k.class == RegionClass::Valid)
        .map(|k| Defect {
            x: k.x as u32,
            y: k.y as u32,
            contrast: k.strength,
        })
        .collect();

    if debug {
        print_debug_stats(&det, params);
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
        let path = output_dir.join(format!("{prefix}_debug_doh.png"));
        render_overlay(&det, w, h)
            .save(&path)
            .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
        println!("Saved {}", path.display());
    }

    println!("Image {w}x{h}: found {} defects", defects.len());

    crop_and_save(
        &det.leveled,
        w,
        h,
        &defects,
        params.crop_size,
        output_dir,
        prefix,
    );
}

fn print_debug_stats(det: &Detection, params: &DohParams) {
    println!(
        "σ_noise = {:.3}, threshold = {:.3} ({}σ)",
        det.sigma,
        params.level_sigma * det.sigma,
        params.level_sigma
    );
    let ladder: Vec<String> = det.ladder.iter().map(|s| format!("{s:.2}")).collect();
    println!(
        "σ ladder (first and last are padding): {}",
        ladder.join(", ")
    );
    for class in RegionClass::ALL {
        let count = det.keypoints.iter().filter(|k| k.class == class).count();
        println!("  {:>13}: {count}", class.name());
    }

    let valid: Vec<&Keypoint> = det
        .keypoints
        .iter()
        .filter(|k| k.class == RegionClass::Valid)
        .collect();
    if valid.is_empty() {
        return;
    }
    let quantiles = |mut v: Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let q = |f: f32| v[((v.len() - 1) as f32 * f).round() as usize];
        format!(
            "min {:.2} / median {:.2} / max {:.2}",
            q(0.0),
            q(0.5),
            q(1.0)
        )
    };
    println!(
        "Valid σ: {}",
        quantiles(valid.iter().map(|k| k.sigma).collect())
    );
    println!(
        "Valid strength: {}",
        quantiles(valid.iter().map(|k| k.strength).collect())
    );
    println!(
        "Valid isotropy: {}",
        quantiles(valid.iter().map(|k| k.isotropy).collect())
    );
}

fn render_overlay(det: &Detection, width: usize, height: usize) -> RgbImage {
    let min = det.leveled.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = det
        .leveled
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-9);

    let mut img = RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let v = det.leveled[y as usize * width + x as usize];
        let g = ((v - min) / range * 255.0) as u8;
        if det.step_mask[y as usize * width + x as usize] {
            // Tint the mask instead of drawing its many keypoints.
            let [r, gr, b] = RegionClass::StepEdge.color();
            let mix = |c: u8| ((g as u16 + c as u16) / 2) as u8;
            Rgb([mix(r), mix(gr), mix(b)])
        } else {
            Rgb([g, g, g])
        }
    });

    for kp in &det.keypoints {
        if kp.class == RegionClass::StepEdge {
            continue;
        }
        let color = Rgb(kp.class.color());
        let r = 2.0 * kp.sigma;
        let steps = (2.0 * std::f32::consts::PI * r).ceil() as usize * 2;
        for i in 0..steps {
            let a = i as f32 / steps as f32 * 2.0 * std::f32::consts::PI;
            let x = (kp.x as f32 + r * a.cos()).round();
            let y = (kp.y as f32 + r * a.sin()).round();
            if x >= 0.0 && y >= 0.0 && (x as usize) < width && (y as usize) < height {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 256;
    const H: usize = 256;
    const CROP: u32 = 40;

    fn noise(i: usize, amp: f32) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        ((x % 10_000) as f32 / 10_000.0 * 2.0 - 1.0) * amp
    }

    fn gaussian_dip(img: &mut [f32], cx: f32, cy: f32, sx: f32, sy: f32, depth: f32) {
        for y in 0..H {
            for x in 0..W {
                let dx = (x as f32 - cx) / sx;
                let dy = (y as f32 - cy) / sy;
                img[y * W + x] -= depth * (-(dx * dx + dy * dy) / 2.0).exp();
            }
        }
    }

    fn base(tilt: f32) -> Vec<f32> {
        (0..W * H)
            .map(|i| 128.0 + noise(i, 2.0) + tilt * ((i % W) + (i / W)) as f32)
            .collect()
    }

    fn params() -> DohParams {
        DohParams {
            crop_size: CROP,
            bg_radius: CROP as usize,
            step_level: 6.0,
            min_sigma: DohParams::default_min_sigma(CROP),
            max_sigma: DohParams::default_max_sigma(CROP),
            num_scales: 6,
            level_sigma: 4.0,
            min_isotropy: 0.3,
        }
    }

    fn valid(det: &Detection) -> Vec<(usize, usize)> {
        det.keypoints
            .iter()
            .filter(|k| k.class == RegionClass::Valid)
            .map(|k| (k.x, k.y))
            .collect()
    }

    fn near(points: &[(usize, usize)], x: usize, y: usize) -> bool {
        points
            .iter()
            .any(|&(px, py)| px.abs_diff(x) <= 1 && py.abs_diff(y) <= 1)
    }

    /// Same scene as the flood tests: three COs, a speck, a large blob,
    /// a ridge and a CO at the border. Only the three COs may be valid.
    fn scene(tilt: f32) -> Vec<f32> {
        let mut img = base(tilt);
        gaussian_dip(&mut img, 60.0, 60.0, 3.5, 3.5, 30.0);
        gaussian_dip(&mut img, 190.0, 70.0, 3.5, 3.5, 30.0);
        gaussian_dip(&mut img, 120.0, 190.0, 3.5, 3.5, 30.0);
        img[140 * W + 60] -= 30.0;
        gaussian_dip(&mut img, 200.0, 180.0, 14.0, 14.0, 30.0);
        gaussian_dip(&mut img, 120.0, 120.0, 10.0, 2.5, 30.0);
        gaussian_dip(&mut img, 8.0, 200.0, 3.5, 3.5, 30.0);
        img
    }

    fn check_scene(det: &Detection) {
        let v = valid(det);
        assert_eq!(v.len(), 3, "valid: {v:?}");
        for (x, y) in [(60, 60), (190, 70), (120, 190)] {
            assert!(near(&v, x, y), "no valid keypoint at ({x}, {y}): {v:?}");
        }
        let edge = det
            .keypoints
            .iter()
            .any(|k| k.class == RegionClass::CloseToEdge && k.x.abs_diff(8) <= 1);
        assert!(edge, "border CO not flagged: {:?}", det.keypoints);
    }

    #[test]
    fn classifies_synthetic_scan() {
        check_scene(&detect(&scene(0.0), W, H, &params()));
    }

    #[test]
    fn leveling_handles_tilt() {
        check_scene(&detect(&scene(0.2), W, H, &params()));
    }

    /// Tip-quality dependent appearance: a CO imaged with a good tip shows a
    /// bright ring around the dip, and one with a bad tip is a shallow,
    /// lopsided smudge. Both must still be found and centered.
    #[test]
    fn finds_ringed_and_smudged_cos() {
        let mut img = base(0.0);
        // Ringed: dip plus a bright annulus (difference of Gaussians).
        gaussian_dip(&mut img, 70.0, 70.0, 3.0, 3.0, 30.0);
        gaussian_dip(&mut img, 70.0, 70.0, 6.0, 6.0, -12.0);
        // Smudge: shallow and mildly elongated.
        gaussian_dip(&mut img, 180.0, 170.0, 5.5, 4.0, 10.0);

        let det = detect(&img, W, H, &params());
        let v = valid(&det);
        assert_eq!(v.len(), 2, "valid: {v:?}");
        assert!(near(&v, 70, 70), "ringed CO missed: {v:?}");
        assert!(near(&v, 180, 170), "smudged CO missed: {v:?}");
    }

    #[test]
    fn ignores_bright_blobs() {
        let mut img = base(0.0);
        gaussian_dip(&mut img, 128.0, 128.0, 3.5, 3.5, -30.0);
        let det = detect(&img, W, H, &params());
        assert!(valid(&det).is_empty(), "{:?}", det.keypoints);
    }

    #[test]
    fn strength_matches_dip_depth() {
        let mut img = vec![0.0_f32; W * H];
        gaussian_dip(&mut img, 128.0, 128.0, 4.0, 4.0, 20.0);
        let [lxx, lyy, lxy] = hessian(&img, W, H, 4.0);
        let i = 128 * W + 128;
        let (strength, iso) = blob_response(lxx[i], lyy[i], lxy[i]);
        assert!((strength - 20.0).abs() < 1.0, "strength {strength}");
        assert!(iso > 0.95, "isotropy {iso}");
    }
}
