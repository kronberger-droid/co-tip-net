use std::path::Path;

use image::GrayImage;

/// A detected defect position in pixel coordinates.
#[derive(Debug, Clone)]
pub struct Defect {
    /// Center x (column)
    pub x: u32,
    /// Center y (row)
    pub y: u32,
    /// Absolute local contrast (higher = more prominent)
    pub contrast: f32,
}

/// Level a grayscale image by subtracting the row-wise median.
/// This removes horizontal scan line artifacts common in STM/AFM images.
///
/// Input: raw pixel values as f32, shape (height, width).
/// Output: leveled pixel values (same shape), centered around zero.
pub fn level_line_median(pixels: &[f32], width: usize, _height: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(pixels.len());
    for row in pixels.chunks_exact(width) {
        let mut sorted = row.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        for &pixel in row {
            result.push(pixel - median);
        }
    }
    result
}

/// Compute local contrast map: for each pixel, the absolute difference
/// from its neighborhood mean.
///
/// `radius` defines the neighborhood: a (2*radius+1) × (2*radius+1) square.
/// Pixels near the border (within `radius` of the edge) should be set to 0.0.
///
/// Returns a contrast map of the same dimensions.
pub fn local_contrast(leveled: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    // TODO: Implement local contrast computation.
    //
    // For each pixel (x, y) not within `radius` of the border:
    //   1. Compute the mean of all pixels in the square neighborhood
    //      from (x-radius, y-radius) to (x+radius, y+radius)
    //   2. contrast[y * width + x] = |pixel - neighborhood_mean|
    //
    // For border pixels, set contrast to 0.0 (we can't crop there anyway).
    //
    // This catches both bright-on-dark (STM bumps) and dark-on-bright
    // (AFM rings) features equally.
    let mut contrast = vec![0.0_f32; width * height];

    for y in radius..height - radius {
        for x in radius..width - radius {
            let mut sum = 0.0;
            for ny in (y - radius)..=(y + radius) {
                for nx in (x - radius)..=(x + radius) {
                    sum += leveled[ny * width + nx];
                }
            }
            let count = (2 * radius + 1) * (2 * radius + 1);
            let mean = sum / count as f32;
            contrast[y * width + x] = (leveled[y * width + x] - mean).abs();
        }
    }
    contrast
}

/// Compute the isotropy ratio at a point: how circular vs directional the
/// local contrast is. Returns min(var_h, var_v) / max(var_h, var_v).
///
/// - 1.0 = perfectly isotropic (circular defect)
/// - 0.0 = completely directional (step edge or lattice row)
///
/// `radius` is how far along each cross-section to sample.
fn isotropy(leveled: &[f32], width: usize, height: usize, cx: usize, cy: usize, radius: usize) -> f32 {
    // Bounds check
    if cx < radius || cy < radius || cx + radius >= width || cy + radius >= height {
        return 0.0;
    }

    // Horizontal cross-section: pixels at (cx-radius..=cx+radius, cy)
    let h_slice: Vec<f32> = (cx - radius..=cx + radius)
        .map(|x| leveled[cy * width + x])
        .collect();

    // Vertical cross-section: pixels at (cx, cy-radius..=cy+radius)
    let v_slice: Vec<f32> = (cy - radius..=cy + radius)
        .map(|y| leveled[y * width + cx])
        .collect();

    let var_h = variance(&h_slice);
    let var_v = variance(&v_slice);

    let max = var_h.max(var_v);
    if max < 1e-9 {
        return 0.0; // flat region, not a defect
    }
    var_h.min(var_v) / max
}

fn variance(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

/// Find local maxima in the contrast map above a threshold,
/// filter by isotropy (rejects step edges and lattice rows),
/// then apply non-maximum suppression to avoid overlapping detections.
///
/// `min_contrast`: minimum contrast value to consider
/// `min_distance`: minimum pixel distance between two detections
/// `leveled`: the leveled image data (for isotropy computation)
/// `min_isotropy`: minimum isotropy ratio (0.0–1.0), e.g. 0.3
pub fn find_peaks(
    contrast: &[f32],
    leveled: &[f32],
    width: usize,
    height: usize,
    min_contrast: f32,
    min_distance: u32,
    min_isotropy: f32,
) -> Vec<Defect> {
    let iso_radius = (min_distance / 2) as usize;

    // Collect all pixels above threshold
    let mut candidates: Vec<Defect> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let c = contrast[y * width + x];
            if c >= min_contrast {
                let iso = isotropy(leveled, width, height, x, y, iso_radius);
                if iso >= min_isotropy {
                    candidates.push(Defect {
                        x: x as u32,
                        y: y as u32,
                        contrast: c,
                    });
                }
            }
        }
    }

    // Sort by contrast descending (strongest peaks first)
    candidates.sort_by(|a, b| b.contrast.partial_cmp(&a.contrast).unwrap());

    // Non-maximum suppression: greedily keep peaks that are far enough apart
    let mut kept: Vec<Defect> = Vec::new();
    for candidate in candidates {
        let dominated = kept.iter().any(|existing| {
            let dx = candidate.x as f32 - existing.x as f32;
            let dy = candidate.y as f32 - existing.y as f32;
            (dx * dx + dy * dy).sqrt() < min_distance as f32
        });
        if !dominated {
            kept.push(candidate);
        }
    }

    kept
}

/// Refine a defect position to the center of mass of |leveled| values
/// in a window around the initial position. Returns the refined (x, y)
/// and the distance the center shifted.
fn refine_center(
    leveled: &[f32],
    width: usize,
    height: usize,
    cx: u32,
    cy: u32,
    radius: u32,
) -> (u32, u32, f32) {
    let r = radius as usize;
    let x0 = (cx as usize).saturating_sub(r);
    let y0 = (cy as usize).saturating_sub(r);
    let x1 = (cx as usize + r).min(width - 1);
    let y1 = (cy as usize + r).min(height - 1);

    let mut sum_w = 0.0_f32;
    let mut sum_wx = 0.0_f32;
    let mut sum_wy = 0.0_f32;

    for y in y0..=y1 {
        for x in x0..=x1 {
            let w = leveled[y * width + x].abs();
            sum_w += w;
            sum_wx += w * x as f32;
            sum_wy += w * y as f32;
        }
    }

    if sum_w < 1e-9 {
        return (cx, cy, 0.0);
    }

    let new_x = (sum_wx / sum_w).round() as u32;
    let new_y = (sum_wy / sum_w).round() as u32;
    let dx = new_x as f32 - cx as f32;
    let dy = new_y as f32 - cy as f32;
    let shift = (dx * dx + dy * dy).sqrt();

    (new_x, new_y, shift)
}

/// Crop square patches around detected defects and save as PNGs.
///
/// `crop_size`: side length of the square crop (in pixels of the original image)
/// Defects too close to the image border (where a full crop can't fit) are skipped.
pub fn crop_and_save(image: &GrayImage, defects: &[Defect], crop_size: u32, output_dir: &Path) {
    let half = crop_size / 2;
    let (img_w, img_h) = image.dimensions();

    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let mut saved = 0;
    for defect in defects {
        // Skip if crop would go out of bounds
        if defect.x < half
            || defect.y < half
            || defect.x + half >= img_w
            || defect.y + half >= img_h
        {
            continue;
        }

        let crop = image::imageops::crop_imm(
            image,
            defect.x - half,
            defect.y - half,
            crop_size,
            crop_size,
        )
        .to_image();

        let filename = output_dir.join(format!("defect_{:04}.png", saved));
        crop.save(&filename)
            .unwrap_or_else(|e| panic!("Failed to save {}: {e}", filename.display()));
        saved += 1;
    }

    println!("Saved {saved} crops to {}", output_dir.display());
}

/// Full extraction pipeline: level → detect → crop.
pub fn extract_defects(
    image: &GrayImage,
    crop_size: u32,
    contrast_radius: usize,
    min_contrast: f32,
    min_isotropy: f32,
    output_dir: &Path,
) {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);

    // Convert to f32
    let pixels: Vec<f32> = image.pixels().map(|p| p.0[0] as f32).collect();

    // Level
    let leveled = level_line_median(&pixels, w, h);

    // Compute contrast map
    let contrast = local_contrast(&leveled, w, h, contrast_radius);

    // Find peaks (with isotropy filter)
    let peaks = find_peaks(&contrast, &leveled, w, h, min_contrast, crop_size, min_isotropy);

    // Refine centers and reject non-point-like features
    let max_shift = crop_size as f32 / 4.0;
    let defects: Vec<Defect> = peaks
        .into_iter()
        .filter_map(|d| {
            let (nx, ny, shift) = refine_center(&leveled, w, h, d.x, d.y, crop_size / 2);
            if shift <= max_shift {
                Some(Defect {
                    x: nx,
                    y: ny,
                    contrast: d.contrast,
                })
            } else {
                None
            }
        })
        .collect();

    println!(
        "Image {}x{}: found {} defects",
        width,
        height,
        defects.len()
    );

    // Crop and save
    crop_and_save(image, &defects, crop_size, output_dir);
}
