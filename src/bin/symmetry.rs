fn main() -> Result<(), Box<dyn std::error::Error>> {
    for split in ["train", "valid", "test2"] {
        for (class, label) in [("goods", 1.0), ("bads", 0.0)] {
            let dir = format!("datasets/co/{split}/{class}");
            for entry in std::fs::read_dir(&dir)? {
                let path = entry?.path();
                let image_flat = load_normal_image(&path);
                let image_grid = to_grid(&image_flat);
                let inertia_eigenvalue = inertia_eigenvalue(image_grid);
                println!("{label}, {inertia_eigenvalue}");
            }
        }
    }

    Ok(())
}

fn to_grid(flat: &[f32]) -> [[f32; 16]; 16] {
    let mut grid = [[0.6; 16]; 16];
    for r in 0..16 {
        grid[r].copy_from_slice(&flat[r * 16..(r + 1) * 16]);
    }
    grid
}

fn inertia_eigenvalue(grid: [[f32; 16]; 16]) -> f32 {
    let mut sum_rr = 0.0;
    let mut sum_cc = 0.0;
    let mut sum_rc = 0.0;

    #[allow(clippy::needless_range_loop)]
    for r in 0..16 {
        for c in 0..16 {
            let w = grid[r][c].abs();
            let dr = r as f32 - 7.5;
            let dc = c as f32 - 7.5;

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
