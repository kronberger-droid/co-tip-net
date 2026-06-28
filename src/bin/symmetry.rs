fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("label,eigen_ratio,rot_corr");
    for split in ["train", "valid", "test2"] {
        for (class, label) in [("goods", 1.0), ("bads", 0.0)] {
            let dir = format!("datasets/co/{split}/{class}");
            for entry in std::fs::read_dir(&dir)? {
                let path = entry?.path();
                let image_flat = load_normal_image(&path);
                let image_grid = to_grid(&image_flat);

                let (r_c, c_c) = centroid(&image_grid);
                let centered = recenter(&image_grid, r_c, c_c);

                let inertia_eigenvalue =
                    inertia_eigenvalue_central(&image_grid, r_c, c_c);
                let rot_autocorrelation = rot_autocorrelation(&centered);
                println!("{label},{inertia_eigenvalue},{rot_autocorrelation}");
            }
        }
    }

    Ok(())
}

fn centroid(grid: &[[f32; 16]; 16]) -> (f32, f32) {
    let mut sum_w = 0.0;
    let mut sum_rw = 0.0;
    let mut sum_cw = 0.0;
    #[allow(clippy::needless_range_loop)]
    for r in 0..16 {
        for c in 0..16 {
            let w = grid[r][c].abs();
            sum_w += w;
            sum_rw += w * r as f32;
            sum_cw += w * c as f32;
        }
    }
    (sum_rw / sum_w, sum_cw / sum_w)
}

fn inertia_eigenvalue_central(
    grid: &[[f32; 16]; 16],
    r_c: f32,
    c_c: f32,
) -> f32 {
    let mut sum_rr = 0.0;
    let mut sum_cc = 0.0;
    let mut sum_rc = 0.0;

    #[allow(clippy::needless_range_loop)]
    for r in 0..16 {
        for c in 0..16 {
            let w = grid[r][c].abs();
            let dr = r as f32 - r_c;
            let dc = c as f32 - c_c;

            sum_rr += w * dr.powi(2);
            sum_cc += w * dc.powi(2);
            sum_rc += w * dr * dc;
        }
    }

    let trace = sum_rr + sum_cc;
    let det = sum_rr * sum_cc - sum_rc * sum_rc;

    let disc = (trace.powi(2) - 4.0 * det).sqrt();
    let lambda_min = (trace - disc) / 2.0;
    let lambda_max = (trace + disc) / 2.0;

    lambda_min / lambda_max
}

// TODO: Implement this.
//
// Shift `grid` so the centroid (r_c, c_c) lands at the image center (7.5, 7.5).
// Pixels that fall outside the 16x16 window should be dropped; empty cells
// should be filled with 0.0 (zero after standardization = "background").
//
// Simplest approach (~6 lines):
//   1. Compute integer shifts: dr = (7.5 - r_c).round() as i32, same for dc.
//   2. Allocate a zeroed output grid.
//   3. For each (r, c) in the source, write into (r + dr, c + dc) if in bounds.
//
// Decide: integer shift (fast, ±0.5 px error) or bilinear (sub-pixel, smoother)?
// Decide: zero-pad (recommended) or wraasdjaskdjaijkajsdaisdjaksdjsadfhkajkdfjasl:hjksjdfklasjdfkhp-around?
fn recenter(grid: &[[f32; 16]; 16], r_c: f32, c_c: f32) -> [[f32; 16]; 16] {
    let dr = (7.5 - r_c).round() as i32;
    let dc = (7.5 - c_c).round() as i32;
    todo!("Your turn — see the comment above")
}

fn to_grid(flat: &[f32]) -> [[f32; 16]; 16] {
    let mut grid = [[0.6; 16]; 16];
    for r in 0..16 {
        grid[r].copy_from_slice(&flat[r * 16..(r + 1) * 16]);
    }
    grid
}

fn load_normal_image(path: &std::path::Path) -> Vec<f32> {
    let image = image::open(path)
        .expect("Failed to load image")
        .grayscale()
        .resize_exact(16, 16, image::imageops::FilterType::Lanczos3)
        .flipv()
        .into_luma8();

    let pixels: Vec<f32> = image.pixels().map(|p| p.0[0] as f32).collect();
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    let std = (pixels.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / pixels.len() as f32)
        .sqrt();
    pixels.iter().map(|x| (x - mean) / std).collect()
}

fn rotate_90(grid: &[[f32; 16]; 16]) -> [[f32; 16]; 16] {
    let mut rotated = [[0.0; 16]; 16];
    #[allow(clippy::needless_range_loop)]
    for r in 0..16 {
        for c in 0..16 {
            rotated[c][15 - r] = grid[r][c];
        }
    }
    rotated
}

fn pearson(a: &[[f32; 16]; 16], b: &[[f32; 16]; 16]) -> f32 {
    // Images are already standardized (mean≈0), so:
    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;
    for r in 0..16 {
        for c in 0..16 {
            sum_ab += a[r][c] * b[r][c];
            sum_aa += a[r][c] * a[r][c];
            sum_bb += b[r][c] * b[r][c];
        }
    }
    sum_ab / (sum_aa.sqrt() * sum_bb.sqrt())
}

fn rot_autocorrelation(grid: &[[f32; 16]; 16]) -> f32 {
    let mut rotated = *grid;
    let mut sum = 0.0;
    for _ in 0..3 {
        rotated = rotate_90(&rotated);
        sum += pearson(&rotated, grid);
    }
    sum / 3.0
}
