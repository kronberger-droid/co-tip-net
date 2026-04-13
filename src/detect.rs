use std::path::Path;

use image::GrayImage;

/// Save an f32 buffer as a grayscale PNG, rescaling to [0, 255].
fn save_debug_image(data: &[f32], width: usize, height: usize, path: &Path) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    let pixels: Vec<u8> = if range < 1e-9 {
        vec![128u8; data.len()]
    } else {
        data.iter()
            .map(|&v| ((v - min) / range * 255.0) as u8)
            .collect()
    };

    let img = GrayImage::from_raw(width as u32, height as u32, pixels)
        .expect("Failed to create debug image");
    img.save(path)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
}

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

/// Level a grayscale image by subtracting a 2D Gaussian-blurred background.
///
/// Unlike row-wise median, this handles diagonal structures (oxide rows)
/// without creating horizontal streaking artifacts. The blur radius should be
/// much larger than the features of interest (e.g. 3–5× the defect size).
///
/// Uses separable box-blur iterated 3 times to approximate a Gaussian.
pub fn level_gaussian_bg(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut bg = pixels.to_vec();
    for _ in 0..3 {
        bg = box_blur_h(&bg, width, height, radius);
        bg = box_blur_v(&bg, width, height, radius);
    }

    pixels
        .iter()
        .zip(bg.iter())
        .map(|(&p, &b)| p - b)
        .collect()
}

/// Horizontal box blur (1D, per row).
fn box_blur_h(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut dst = vec![0.0_f32; width * height];
    let diam = 2 * radius + 1;

    for y in 0..height {
        let row_start = y * width;

        let mut sum = 0.0_f32;
        for x in 0..radius.min(width) {
            sum += src[row_start + x];
        }
        sum += src[row_start] * (radius + 1).min(diam) as f32;

        for x in 0..width {
            let right = (x + radius).min(width - 1);
            sum += src[row_start + right];

            if x > radius {
                let left = (x as isize - radius as isize - 1).max(0) as usize;
                sum -= src[row_start + left];
            } else {
                sum -= src[row_start];
            }

            dst[row_start + x] = sum / diam as f32;
        }
    }
    dst
}

/// Vertical box blur (1D, per column).
fn box_blur_v(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut dst = vec![0.0_f32; width * height];
    let diam = 2 * radius + 1;

    for x in 0..width {
        let mut sum = 0.0_f32;
        for y in 0..radius.min(height) {
            sum += src[y * width + x];
        }
        sum += src[x] * (radius + 1).min(diam) as f32;

        for y in 0..height {
            let bottom = (y + radius).min(height - 1);
            sum += src[bottom * width + x];

            if y > radius {
                let top = (y as isize - radius as isize - 1).max(0) as usize;
                sum -= src[top * width + x];
            } else {
                sum -= src[x];
            }

            dst[y * width + x] = sum / diam as f32;
        }
    }
    dst
}

/// Compute local contrast map: for each pixel, the absolute difference
/// from its neighborhood mean.
///
/// `radius` defines the neighborhood: a (2*radius+1) × (2*radius+1) square.
/// Pixels near the border (within `radius` of the edge) are set to 0.0.
///
/// Returns a contrast map of the same dimensions.
pub fn local_contrast(leveled: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
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

/// Compute the "blobness" of a local region using the structure tensor.
///
/// The structure tensor accumulates gradient outer products over a window,
/// yielding a 2×2 matrix whose eigenvalues characterise the local geometry:
/// - Both eigenvalues large and similar → isotropic blob (point defect)
/// - One large, one small → ridge / edge (CuO row, step edge)
/// - Both small → flat region
///
/// Returns `λ_min / λ_max` ∈ [0, 1]:
/// - Close to 1.0 → point-like feature
/// - Close to 0.0 → directional feature (ridge / edge)
fn blobness(
    leveled: &[f32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> f32 {
    if cx < radius + 1
        || cy < radius + 1
        || cx + radius + 1 >= width
        || cy + radius + 1 >= height
    {
        return 0.0;
    }

    let sigma = radius as f32 / 2.0;
    let sigma2 = 2.0 * sigma * sigma;

    let mut s_xx = 0.0_f32;
    let mut s_yy = 0.0_f32;
    let mut s_xy = 0.0_f32;
    let mut w_sum = 0.0_f32;

    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let x = (cx as isize + dx) as usize;
            let y = (cy as isize + dy) as usize;

            let ix = (leveled[y * width + x + 1] - leveled[y * width + x - 1]) / 2.0;
            let iy = (leveled[(y + 1) * width + x] - leveled[(y - 1) * width + x]) / 2.0;

            let r2 = (dx * dx + dy * dy) as f32;
            let w = (-r2 / sigma2).exp();

            s_xx += w * ix * ix;
            s_yy += w * iy * iy;
            s_xy += w * ix * iy;
            w_sum += w;
        }
    }

    s_xx /= w_sum;
    s_yy /= w_sum;
    s_xy /= w_sum;

    let trace = s_xx + s_yy;
    let det = s_xx * s_yy - s_xy * s_xy;

    if trace < 1e-9 {
        return 0.0;
    }

    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_min = (trace - disc) / 2.0;
    let lambda_max = (trace + disc) / 2.0;

    if lambda_max < 1e-9 {
        return 0.0;
    }

    lambda_min / lambda_max
}

/// Find local maxima in the contrast map above a threshold,
/// filter by blobness (rejects step edges, ridges, and lattice rows),
/// then apply non-maximum suppression to avoid overlapping detections.
///
/// `min_contrast`: minimum contrast value to consider
/// `min_distance`: minimum pixel distance between two detections
/// `leveled`: the leveled image data (for blobness computation)
/// `min_isotropy`: minimum blobness ratio (0.0–1.0), e.g. 0.3
pub fn find_peaks(
    contrast: &[f32],
    leveled: &[f32],
    width: usize,
    height: usize,
    min_contrast: f32,
    min_distance: u32,
    min_isotropy: f32,
) -> Vec<Defect> {
    let blob_radius = (min_distance / 2) as usize;

    let local_r = 2_usize; // 5×5 neighborhood for local max check
    let mut candidates: Vec<Defect> = Vec::new();
    for y in local_r..height.saturating_sub(local_r) {
        for x in local_r..width.saturating_sub(local_r) {
            let c = contrast[y * width + x];
            if c < min_contrast {
                continue;
            }

            let is_local_max = (y - local_r..=y + local_r).all(|ny| {
                (x - local_r..=x + local_r)
                    .all(|nx| (ny == y && nx == x) || contrast[ny * width + nx] <= c)
            });
            if !is_local_max {
                continue;
            }

            let blob = blobness(leveled, width, height, x, y, blob_radius);
            if blob >= min_isotropy {
                candidates.push(Defect {
                    x: x as u32,
                    y: y as u32,
                    contrast: c,
                });
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

/// Measure how concentrated the *depression* signal is around the center.
///
/// Only counts negative leveled values (the actual depression), ignoring
/// positive values (bright ring in oxide-tip images, lattice bright spots).
/// This way oxide-tip ring defects pass (depression is at center) and
/// Cu-tip diffuse depressions also pass (depression is still centered,
/// just broader), while lattice patterns fail (negative values are spread
/// uniformly across the patch).
///
/// Returns fraction of total negative intensity within inner half-radius.
fn depression_concentration(
    leveled: &[f32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> f32 {
    let r = radius as isize;
    let inner_r2 = (radius as f32 / 2.0).powi(2);
    let outer_r2 = (radius as f32).powi(2);

    let mut inner_sum = 0.0_f32;
    let mut total_sum = 0.0_f32;

    for dy in -r..=r {
        for dx in -r..=r {
            let px = cx as isize + dx;
            let py = cy as isize + dy;
            if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
                continue;
            }
            let dist2 = (dx * dx + dy * dy) as f32;
            if dist2 > outer_r2 {
                continue;
            }
            let val = leveled[py as usize * width + px as usize];
            if val >= 0.0 {
                continue; // only count depressions
            }
            let neg = -val; // make positive for summing
            total_sum += neg;
            if dist2 <= inner_r2 {
                inner_sum += neg;
            }
        }
    }

    if total_sum < 1e-9 {
        return 0.0;
    }
    inner_sum / total_sum
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

/// Crop square patches around detected defects from the original image
/// and save as grayscale PNGs (no normalisation — the classifier handles that).
///
/// `crop_size`: side length of the square crop (in pixels)
/// Defects too close to the image border (where a full crop can't fit) are skipped.
pub fn crop_and_save(
    image: &GrayImage,
    defects: &[Defect],
    crop_size: u32,
    output_dir: &Path,
) {
    let half = crop_size / 2;
    let (img_w, img_h) = image.dimensions();

    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let mut saved = 0;
    for defect in defects {
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
///
/// If `debug` is true, saves intermediate images to `output_dir`:
/// - `debug_line_leveled.png`: after row-wise median subtraction
/// - `debug_leveled.png`: after Gaussian background subtraction
/// - `debug_contrast.png`: local contrast map
pub fn extract_defects(
    image: &GrayImage,
    crop_size: u32,
    contrast_radius: usize,
    min_contrast: f32,
    min_isotropy: f32,
    output_dir: &Path,
    debug: bool,
) {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);

    // Convert to f32
    let pixels: Vec<f32> = image.pixels().map(|p| p.0[0] as f32).collect();

    // Step 1: row-wise median to remove scan-line offsets
    let line_leveled = level_line_median(&pixels, w, h);

    // Step 2: 2D Gaussian background subtraction for detection only.
    //         Removes slow gradients without the streaking that row-median
    //         creates near oxide regions.
    let bg_radius = contrast_radius * 3;
    let leveled = level_gaussian_bg(&line_leveled, w, h, bg_radius);

    // Compute contrast map
    let contrast = local_contrast(&leveled, w, h, contrast_radius);

    if debug {
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
        save_debug_image(&line_leveled, w, h, &output_dir.join("debug_line_leveled.png"));
        save_debug_image(&leveled, w, h, &output_dir.join("debug_leveled.png"));
        save_debug_image(&contrast, w, h, &output_dir.join("debug_contrast.png"));
        println!("Saved debug images to {}", output_dir.display());
    }

    // Find peaks (with structure tensor blobness filter)
    let peaks = find_peaks(
        &contrast,
        &leveled,
        w,
        h,
        min_contrast,
        crop_size,
        min_isotropy,
    );

    if debug {
        println!("Peaks after blobness filter: {}", peaks.len());
    }

    // Refine centers and apply lightweight filters
    let max_shift = crop_size as f32 / 4.0;
    let half = (crop_size / 2) as usize;
    let min_depression_conc = 0.35; // depression signal must be ≥35% concentrated in inner half

    let mut reject_shift = 0;
    let mut reject_bright = 0;
    let mut reject_conc = 0;

    let defects: Vec<Defect> = peaks
        .into_iter()
        .filter_map(|d| {
            let (nx, ny, shift) = refine_center(&leveled, w, h, d.x, d.y, crop_size / 2);
            if shift > max_shift {
                reject_shift += 1;
                return None;
            }

            let cx = nx as usize;
            let cy = ny as usize;

            // Reject bright protrusions — CO molecules appear as dark depressions
            if leveled[cy * w + cx] > 0.0 {
                reject_bright += 1;
                return None;
            }

            // Reject patches where depression signal is spread evenly (lattice)
            let conc = depression_concentration(&leveled, w, h, cx, cy, half);
            if conc < min_depression_conc {
                reject_conc += 1;
                return None;
            }

            Some(Defect {
                x: nx,
                y: ny,
                contrast: d.contrast,
            })
        })
        .collect();

    if debug {
        println!(
            "Rejected: shift={reject_shift} bright={reject_bright} depression_conc={reject_conc}"
        );
        println!("After filters: {}", defects.len());
    }

    // Post-refinement NMS: center refinement can cause initially-distant
    // peaks to converge onto the same defect, creating duplicates.
    let mut final_defects: Vec<Defect> = Vec::new();
    for d in defects {
        let dominated = final_defects.iter().any(|existing| {
            let dx = d.x as f32 - existing.x as f32;
            let dy = d.y as f32 - existing.y as f32;
            (dx * dx + dy * dy).sqrt() < crop_size as f32
        });
        if !dominated {
            final_defects.push(d);
        }
    }

    println!(
        "Image {}x{}: found {} defects",
        width,
        height,
        final_defects.len()
    );

    // Crop from original image — the classifier does its own normalisation
    crop_and_save(image, &final_defects, crop_size, output_dir);
}
