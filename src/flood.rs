//! Defect extraction by flooding, after Schnorrenberg et al. (Uni Osnabrück).
//!
//! Instead of detecting peaks and filtering them through a chain of
//! heuristics, the leveled scan is "flooded" up to a water level below the
//! background. Every connected submerged region is one candidate, and the
//! region as a whole is classified by its size and shape:
//!
//! - too small: noise specks
//! - too large: clusters, step edges, oxide patches
//! - elongated: oxide rows, ridges
//! - close to edge: a full crop would not fit
//! - valid: a single isolated depression, cropped around its centroid
//!
//! CO molecules image as depressions, so only the negative side is flooded.

use std::path::Path;

use image::{GrayImage, Rgb, RgbImage};

use crate::detect::{Defect, crop_and_save, level_gaussian_bg, level_line_median};

/// Parameters for flood-based extraction.
pub struct FloodParams {
    /// Side length of the square crop around each valid region (pixels).
    pub crop_size: u32,
    /// Water level in units of the robust noise σ: pixels with
    /// `leveled < -level_sigma * σ` are submerged.
    pub level_sigma: f32,
    /// Regions with fewer pixels are classified as too small.
    pub min_area: usize,
    /// Regions with more pixels are classified as too large.
    pub max_area: usize,
    /// Minimum `λ_min / λ_max` of the region's mass covariance.
    /// 1.0 is perfectly round, 0.3 is roughly a 1.8:1 aspect ratio.
    pub min_isotropy: f32,
}

impl FloodParams {
    /// Area defaults scaled to the crop size, since px/nm differs per scan.
    pub fn default_min_area(crop_size: u32) -> usize {
        ((crop_size / 8).max(1) as usize).pow(2)
    }

    pub fn default_max_area(crop_size: u32) -> usize {
        ((crop_size / 2) as usize).pow(2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionClass {
    Valid,
    TooSmall,
    TooLarge,
    Elongated,
    CloseToEdge,
}

impl RegionClass {
    pub(crate) const ALL: [RegionClass; 5] = [
        RegionClass::Valid,
        RegionClass::TooSmall,
        RegionClass::TooLarge,
        RegionClass::Elongated,
        RegionClass::CloseToEdge,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            RegionClass::Valid => "valid",
            RegionClass::TooSmall => "too small",
            RegionClass::TooLarge => "too large",
            RegionClass::Elongated => "elongated",
            RegionClass::CloseToEdge => "close to edge",
        }
    }

    /// Overlay color, roughly following the poster's legend.
    pub(crate) fn color(self) -> [u8; 3] {
        match self {
            RegionClass::Valid => [40, 200, 60],
            RegionClass::TooSmall => [80, 200, 230],
            RegionClass::TooLarge => [40, 60, 220],
            RegionClass::Elongated => [210, 60, 210],
            RegionClass::CloseToEdge => [240, 150, 30],
        }
    }
}

/// A connected submerged region.
#[derive(Debug, Clone)]
pub struct Region {
    /// Flat pixel indices (`y * width + x`) belonging to the region.
    pub pixels: Vec<usize>,
    /// Depth-weighted centroid (x, y).
    pub centroid: (f32, f32),
    /// Deepest point below zero, as a positive value.
    pub depth: f32,
    /// `λ_min / λ_max` of the depth-weighted covariance.
    pub isotropy: f32,
    /// Whether any pixel lies on the image border.
    pub touches_border: bool,
}

/// Robust noise estimate: `1.4826 · MAD`, which equals σ for Gaussian noise
/// and is insensitive to the sparse depressions we are looking for.
pub fn robust_sigma(data: &[f32]) -> f32 {
    let mut buf = data.to_vec();
    let median = median_in_place(&mut buf);
    for v in buf.iter_mut() {
        *v = (*v - median).abs();
    }
    1.4826 * median_in_place(&mut buf)
}

fn median_in_place(buf: &mut [f32]) -> f32 {
    let mid = buf.len() / 2;
    let (_, m, _) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Label all 8-connected regions where `leveled < -threshold`.
///
/// Uses an explicit stack: a step edge or oxide terrace can be one region
/// spanning most of the scan, which would overflow a recursive fill.
pub fn flood_regions(leveled: &[f32], width: usize, height: usize, threshold: f32) -> Vec<Region> {
    let submerged = |i: usize| leveled[i] < -threshold;
    let mut visited = vec![false; width * height];
    let mut regions = Vec::new();
    let mut stack = Vec::new();

    for start in 0..width * height {
        if visited[start] || !submerged(start) {
            continue;
        }

        visited[start] = true;
        stack.push(start);
        let mut pixels = Vec::new();

        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, y) = (i % width, i / width);
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let n = ny * width + nx;
                    if !visited[n] && submerged(n) {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
        }

        regions.push(region_stats(pixels, leveled, width, height));
    }

    regions
}

fn region_stats(pixels: Vec<usize>, leveled: &[f32], width: usize, height: usize) -> Region {
    let mut sum_w = 0.0_f32;
    let mut sum_wx = 0.0_f32;
    let mut sum_wy = 0.0_f32;
    let mut depth = 0.0_f32;
    let mut touches_border = false;

    for &i in &pixels {
        let (x, y) = (i % width, i / width);
        let w = -leveled[i];
        sum_w += w;
        sum_wx += w * x as f32;
        sum_wy += w * y as f32;
        depth = depth.max(w);
        touches_border |= x == 0 || y == 0 || x == width - 1 || y == height - 1;
    }

    let (cx, cy) = (sum_wx / sum_w, sum_wy / sum_w);

    let mut s_xx = 0.0_f32;
    let mut s_yy = 0.0_f32;
    let mut s_xy = 0.0_f32;
    for &i in &pixels {
        let dx = (i % width) as f32 - cx;
        let dy = (i / width) as f32 - cy;
        let w = -leveled[i];
        s_xx += w * dx * dx;
        s_yy += w * dy * dy;
        s_xy += w * dx * dy;
    }

    let trace = s_xx + s_yy;
    let det = s_xx * s_yy - s_xy * s_xy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_max = (trace + disc) / 2.0;
    let lambda_min = (trace - disc) / 2.0;
    // A single pixel has zero spread; call it round and let area decide.
    let isotropy = if lambda_max < 1e-9 {
        1.0
    } else {
        lambda_min / lambda_max
    };

    Region {
        pixels,
        centroid: (cx, cy),
        depth,
        isotropy,
        touches_border,
    }
}

pub fn classify(region: &Region, params: &FloodParams, width: usize, height: usize) -> RegionClass {
    let area = region.pixels.len();
    if area < params.min_area {
        return RegionClass::TooSmall;
    }
    if area > params.max_area {
        return RegionClass::TooLarge;
    }
    if region.isotropy < params.min_isotropy {
        return RegionClass::Elongated;
    }

    let half = (params.crop_size / 2) as f32;
    let (cx, cy) = region.centroid;
    let fits = cx.round() >= half
        && cy.round() >= half
        && cx.round() + half < width as f32
        && cy.round() + half < height as f32;
    if region.touches_border || !fits {
        return RegionClass::CloseToEdge;
    }

    RegionClass::Valid
}

/// Result of segmenting a scan: the leveled image, noise level and every
/// region with its class.
pub struct Segmentation {
    pub leveled: Vec<f32>,
    pub sigma: f32,
    pub regions: Vec<(Region, RegionClass)>,
}

/// Level the raw scan, flood it, and classify every region.
pub fn segment(pixels: &[f32], width: usize, height: usize, params: &FloodParams) -> Segmentation {
    // Row median removes scan-line offsets, then a 2D background much wider
    // than a molecule removes slow gradients.
    let line_leveled = level_line_median(pixels, width, height);
    let leveled = level_gaussian_bg(&line_leveled, width, height, params.crop_size as usize);

    let sigma = robust_sigma(&leveled);
    let regions = flood_regions(&leveled, width, height, params.level_sigma * sigma)
        .into_iter()
        .map(|r| {
            let class = classify(&r, params, width, height);
            (r, class)
        })
        .collect();

    Segmentation {
        leveled,
        sigma,
        regions,
    }
}

/// Full extraction pipeline: level → flood → classify → crop valid regions.
///
/// If `debug` is true, prints σ, per-class counts and the area distribution,
/// and saves `debug_flood.png`: the leveled scan with regions colored by class
/// and a cross on each valid centroid.
pub fn extract_defects_flood(
    image: &GrayImage,
    params: &FloodParams,
    output_dir: &Path,
    debug: bool,
) {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    let pixels: Vec<f32> = image.pixels().map(|p| p.0[0] as f32).collect();

    let seg = segment(&pixels, w, h, params);

    let defects: Vec<Defect> = seg
        .regions
        .iter()
        .filter(|(_, class)| *class == RegionClass::Valid)
        .map(|(r, _)| Defect {
            x: r.centroid.0.round() as u32,
            y: r.centroid.1.round() as u32,
            contrast: r.depth,
        })
        .collect();

    if debug {
        print_debug_stats(&seg, params);
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
        let path = output_dir.join("debug_flood.png");
        render_overlay(&seg, w, h)
            .save(&path)
            .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
        println!("Saved {}", path.display());
    }

    println!(
        "Image {}x{}: found {} defects",
        width,
        height,
        defects.len()
    );

    crop_and_save(image, &defects, params.crop_size, output_dir);
}

fn print_debug_stats(seg: &Segmentation, params: &FloodParams) {
    println!(
        "σ = {:.3}, water level = {:.3} ({}σ)",
        seg.sigma,
        -params.level_sigma * seg.sigma,
        params.level_sigma
    );
    println!(
        "Area limits: {}..={} px, min isotropy {}",
        params.min_area, params.max_area, params.min_isotropy
    );
    for class in RegionClass::ALL {
        let count = seg.regions.iter().filter(|(_, c)| *c == class).count();
        println!("  {:>13}: {count}", class.name());
    }

    let mut areas: Vec<usize> = seg.regions.iter().map(|(r, _)| r.pixels.len()).collect();
    if areas.is_empty() {
        return;
    }
    areas.sort_unstable();
    let q = |f: f32| areas[((areas.len() - 1) as f32 * f).round() as usize];
    println!(
        "Region areas: min {} / p25 {} / median {} / p75 {} / max {}",
        q(0.0),
        q(0.25),
        q(0.5),
        q(0.75),
        q(1.0)
    );
}

fn render_overlay(seg: &Segmentation, width: usize, height: usize) -> RgbImage {
    let min = seg.leveled.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = seg
        .leveled
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-9);

    let mut img = RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let v = seg.leveled[y as usize * width + x as usize];
        let g = ((v - min) / range * 255.0) as u8;
        Rgb([g, g, g])
    });

    for (region, class) in &seg.regions {
        let color = class.color();
        for &i in &region.pixels {
            let p = img.get_pixel_mut((i % width) as u32, (i / width) as u32);
            for (channel, tint) in p.0.iter_mut().zip(color) {
                *channel = ((*channel as u16 + 2 * tint as u16) / 3) as u8;
            }
        }
    }

    for (region, class) in &seg.regions {
        if *class != RegionClass::Valid {
            continue;
        }
        let (cx, cy) = (
            region.centroid.0.round() as i64,
            region.centroid.1.round() as i64,
        );
        for d in -3..=3_i64 {
            for (x, y) in [(cx + d, cy), (cx, cy + d)] {
                if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                    img.put_pixel(x as u32, y as u32, Rgb([255, 30, 30]));
                }
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

    /// Deterministic noise in [-amp, amp] so tests need no RNG crate.
    fn noise(i: usize, amp: f32) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        ((x % 10_000) as f32 / 10_000.0 * 2.0 - 1.0) * amp
    }

    fn gaussian_dip(img: &mut [f32], cx: f32, cy: f32, sigma_x: f32, sigma_y: f32, depth: f32) {
        for y in 0..H {
            for x in 0..W {
                let dx = (x as f32 - cx) / sigma_x;
                let dy = (y as f32 - cy) / sigma_y;
                img[y * W + x] -= depth * (-(dx * dx + dy * dy) / 2.0).exp();
            }
        }
    }

    /// Flat terrace with noise, three isolated COs, a speck, a large blob,
    /// a ridge (oxide row) and a CO too close to the border.
    fn synthetic_scan(tilt: f32) -> Vec<f32> {
        let mut img: Vec<f32> = (0..W * H)
            .map(|i| 128.0 + noise(i, 2.0) + tilt * ((i % W) + (i / W)) as f32)
            .collect();

        // Valid COs: radius ~ crop/6.
        gaussian_dip(&mut img, 60.0, 60.0, 3.5, 3.5, 30.0);
        gaussian_dip(&mut img, 190.0, 70.0, 3.5, 3.5, 30.0);
        gaussian_dip(&mut img, 120.0, 190.0, 3.5, 3.5, 30.0);
        // Speck: sharp single-pixel dip.
        img[140 * W + 60] -= 30.0;
        // Large blob.
        gaussian_dip(&mut img, 200.0, 180.0, 14.0, 14.0, 30.0);
        // Ridge: long thin dip along x.
        gaussian_dip(&mut img, 120.0, 120.0, 10.0, 2.5, 30.0);
        // CO near the border.
        gaussian_dip(&mut img, 8.0, 200.0, 3.5, 3.5, 30.0);

        img
    }

    fn params() -> FloodParams {
        FloodParams {
            crop_size: CROP,
            level_sigma: 3.0,
            min_area: FloodParams::default_min_area(CROP),
            max_area: FloodParams::default_max_area(CROP),
            min_isotropy: 0.3,
        }
    }

    fn class_at(seg: &Segmentation, x: f32, y: f32) -> RegionClass {
        seg.regions
            .iter()
            .min_by(|(a, _), (b, _)| {
                let da = (a.centroid.0 - x).powi(2) + (a.centroid.1 - y).powi(2);
                let db = (b.centroid.0 - x).powi(2) + (b.centroid.1 - y).powi(2);
                da.total_cmp(&db)
            })
            .map(|(_, c)| *c)
            .expect("no regions found")
    }

    fn check(seg: &Segmentation) {
        let valid: Vec<_> = seg
            .regions
            .iter()
            .filter(|(_, c)| *c == RegionClass::Valid)
            .map(|(r, _)| r.centroid)
            .collect();
        assert_eq!(valid.len(), 3, "valid centroids: {valid:?}");
        for (x, y) in [(60.0, 60.0), (190.0, 70.0), (120.0, 190.0)] {
            assert!(
                valid
                    .iter()
                    .any(|(cx, cy)| (cx - x).abs() < 1.0 && (cy - y).abs() < 1.0),
                "no valid region at ({x}, {y}): {valid:?}"
            );
        }

        assert_eq!(class_at(seg, 60.0, 140.0), RegionClass::TooSmall);
        assert_eq!(class_at(seg, 200.0, 180.0), RegionClass::TooLarge);
        assert_eq!(class_at(seg, 120.0, 120.0), RegionClass::Elongated);
        assert_eq!(class_at(seg, 8.0, 200.0), RegionClass::CloseToEdge);
    }

    #[test]
    fn classifies_synthetic_scan() {
        let seg = segment(&synthetic_scan(0.0), W, H, &params());
        check(&seg);
    }

    #[test]
    fn leveling_handles_tilt() {
        let seg = segment(&synthetic_scan(0.2), W, H, &params());
        check(&seg);
    }

    #[test]
    fn robust_sigma_ignores_outliers() {
        let mut data: Vec<f32> = (0..10_000).map(|i| noise(i, 1.0)).collect();
        let clean = robust_sigma(&data);
        for v in data.iter_mut().take(200) {
            *v -= 50.0;
        }
        let dirty = robust_sigma(&data);
        assert!((dirty - clean).abs() / clean < 0.1, "{clean} vs {dirty}");
    }

    #[test]
    fn flood_survives_huge_region() {
        // A full-image region must not overflow the stack.
        let leveled = vec![-1.0_f32; 1024 * 1024];
        let regions = flood_regions(&leveled, 1024, 1024, 0.5);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].pixels.len(), 1024 * 1024);
    }
}
