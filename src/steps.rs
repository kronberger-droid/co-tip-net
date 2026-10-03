//! Step-edge mask for real STM scans.
//!
//! A monatomic Cu(110) step is ~130 pm over a few pixels, while a CO dip is
//! ~20–30 pm deep and a few pixels wide. After background subtraction both
//! leave dark features of similar DoH strength, thus the step has to be
//! found on the unfiltered height map instead: strong gradients that form
//! long connected lines. A CO dip only produces a small gradient ring.

use crate::flood::{flood_regions, median, robust_sigma};

/// Pixels on or near a step edge.
///
/// `data` is the row-leveled height map (no background subtraction).
/// Gradient pixels above `median + level · σ` form candidate regions, and
/// those spanning at least `min_extent` pixels are kept and dilated by
/// `dilate`. A `level` of zero or less disables the mask.
pub fn step_mask(
    data: &[f32],
    width: usize,
    height: usize,
    level: f32,
    min_extent: usize,
    dilate: usize,
) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    if level <= 0.0 || width < 3 || height < 3 {
        return mask;
    }

    let gradient = sobel_magnitude(data, width, height);
    let med = median(&gradient);
    let sigma = robust_sigma(&gradient);
    // flood_regions submerges values below -threshold, thus negate around
    // the median to select the strong gradients.
    let negated: Vec<f32> = gradient.iter().map(|g| med - g).collect();

    for region in flood_regions(&negated, width, height, level * sigma) {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for &i in &region.pixels {
            let (x, y) = (i % width, i / width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        if (x1 - x0).max(y1 - y0) + 1 >= min_extent {
            for &i in &region.pixels {
                mask[i] = true;
            }
        }
    }

    dilate_mask(&mask, width, height, dilate)
}

/// Robust spread of `data` over the pixels outside `mask`, falling back to
/// all pixels when nearly everything is masked.
pub fn unmasked_sigma(data: &[f32], mask: &[bool]) -> f32 {
    let free: Vec<f32> = data
        .iter()
        .zip(mask)
        .filter(|&(_, &m)| !m)
        .map(|(&v, _)| v)
        .collect();
    if free.len() < data.len() / 10 {
        robust_sigma(data)
    } else {
        robust_sigma(&free)
    }
}

fn sobel_magnitude(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let at = |x: usize, y: usize| data[y * width + x];
    let mut out = vec![0.0_f32; width * height];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            out[y * width + x] = (gx * gx + gy * gy).sqrt() / 8.0;
        }
    }
    out
}

/// Square dilation by `radius`, separable via running counts.
fn dilate_mask(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    let pass = |src: &[bool], len: usize, lines: usize, idx: &dyn Fn(usize, usize) -> usize| {
        let mut dst = vec![false; src.len()];
        for line in 0..lines {
            let mut count = 0usize;
            for i in 0..radius.min(len) {
                count += src[idx(line, i)] as usize;
            }
            for i in 0..len {
                if i + radius < len {
                    count += src[idx(line, i + radius)] as usize;
                }
                if i > radius {
                    count -= src[idx(line, i - radius - 1)] as usize;
                }
                dst[idx(line, i)] = count > 0;
            }
        }
        dst
    };
    let rows = pass(mask, width, height, &|y, x| y * width + x);
    pass(&rows, height, width, &|x, y| y * width + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 200;
    const H: usize = 160;

    fn noise(i: usize) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        (x % 10_000) as f32 / 10_000.0 * 4.0 - 2.0
    }

    /// Terrace step of 130 along a slanted line, plus one CO-sized dip.
    fn scene() -> Vec<f32> {
        (0..W * H)
            .map(|i| {
                let (x, y) = ((i % W) as f32, (i / W) as f32);
                let edge = 100.0 + 0.2 * y;
                let step = 130.0 / (1.0 + (-(x - edge) / 1.5).exp());
                let (dx, dy) = (x - 40.0, y - 80.0);
                let dip = 30.0 * (-(dx * dx + dy * dy) / (2.0 * 3.5 * 3.5)).exp();
                step - dip + noise(i)
            })
            .collect()
    }

    #[test]
    fn masks_step_but_not_dip() {
        let mask = step_mask(&scene(), W, H, 6.0, 40, 4);
        let at = |x: usize, y: usize| mask[y * W + x];
        assert!(at(100, 2) && at(116, 80) && at(131, 157), "step not masked");
        assert!(!at(40, 80), "CO dip masked");
        assert!(!at(10, 10) && !at(190, 150), "flat terrace masked");
    }

    #[test]
    fn disabled_with_zero_level() {
        assert!(!step_mask(&scene(), W, H, 0.0, 40, 4).iter().any(|&m| m));
    }

    #[test]
    fn dilation_is_square() {
        let mut mask = vec![false; 25];
        mask[12] = true; // center of 5x5
        let d = dilate_mask(&mask, 5, 5, 1);
        let set: Vec<usize> = (0..25).filter(|&i| d[i]).collect();
        assert_eq!(set, vec![6, 7, 8, 11, 12, 13, 16, 17, 18]);
    }
}
