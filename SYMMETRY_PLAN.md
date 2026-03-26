# Symmetry metric analysis plan

## Goal

Investigate whether simple mathematical symmetry measures can replace
the 2157-parameter CNN for CO-tip quality classification.

## Status

### Done

- [x] Inertia tensor eigenvalue ratio (λ_min / λ_max)
  - Measures spatial eccentricity of intensity distribution
  - **Result: distributions overlap heavily, not a useful classifier alone**
  - Good tips: ~0.47–0.98, Bad tips: ~0.31–0.98
  - Insight: measures mass spread, not pattern structure — symmetric bad tips score high

### Next

- [ ] **Rotational autocorrelation**
  - Correlate 16×16 grid with its 90°, 180°, 270° rotations
  - Average the Pearson correlation coefficients
  - Captures whether the pattern *looks the same* under rotation (not just mass distribution)
  - Implementation: rotate grid (transpose + reverse rows), then correlate

- [ ] **Combine metrics + visualize**
  - Output CSV with columns: `label, eigen_ratio, rot_corr`
  - Plot in Typst/lilaq: histograms per metric colored by class, and a 2D scatter (eigen_ratio vs rot_corr)
  - Visually assess separability

### Future candidates (if needed)

- [ ] Radial power spectrum ratio (2D FFT, variance within radial rings)
- [ ] Zernike moments (ratio of m=0 modes to all modes)
- [ ] Extended isotropy (4+ directional cross-sections, not just H/V)
- [ ] Radial intensity profile kurtosis (contrast falloff shape)

## Key implementation details

- All code lives in `src/bin/symmetry.rs` (standalone binary, no Burn dependency)
- Image loading is inlined (~15 lines) to avoid pulling in the full crate
- Grid representation: `[[f32; 16]; 16]` (stack-allocated, fixed size)
- Dataset path: `datasets/co/{train,valid,test2}/{goods,bads}/*.png`
- Preprocessing matches the CNN pipeline: resize 16×16, flipv, per-image standardize

## 90° rotation recipe

```rust
fn rotate_90(grid: &[[f32; 16]; 16]) -> [[f32; 16]; 16] {
    let mut rotated = [[0.0; 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            rotated[c][15 - r] = grid[r][c];
        }
    }
    rotated
}
```

## Pearson correlation recipe

```rust
fn pearson(a: &[[f32; 16]; 16], b: &[[f32; 16]; 16]) -> f32 {
    // Images are already standardized (mean≈0), so:
    // r = Σ(a*b) / sqrt(Σa² · Σb²)
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
```
