//! Validate wass's OT against exact 1D references.
//!
//! On a unit-spaced 1D grid the Wasserstein-1 distance has a closed form
//! (`wasserstein_1d` = sum of |CDF| differences), and a point mass shifted by k
//! bins is exactly W1 = k. Entropic OT (`sinkhorn_log`) must converge to that
//! exact value as the regularization shrinks. Disagreement beyond the entropic
//! bias is a bug.
//!
//! ```sh
//! cargo run --release --example ot_validation
//! ```

use ndarray::{Array1, Array2};
use wass::{earth_mover_distance, sinkhorn_log, wasserstein_1d};

fn grid_cost(n: usize, p: f32) -> Array2<f32> {
    Array2::from_shape_fn((n, n), |(i, j)| (i as f32 - j as f32).abs().powf(p))
}

fn normalize(v: &mut [f32]) {
    let s: f32 = v.iter().sum();
    for x in v.iter_mut() {
        *x /= s;
    }
}

fn gaussian_hist(n: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| {
            let z = (i as f32 - mean) / std;
            (-0.5 * z * z).exp()
        })
        .collect();
    normalize(&mut v);
    v
}

fn main() {
    let n = 50;
    let cost1 = grid_cost(n, 1.0);

    println!("=== exact reference: point-mass shift, W1 == shift ===");
    for k in [1usize, 7, 20] {
        let mut a = vec![0.0f32; n];
        a[0] = 1.0;
        let mut b = vec![0.0f32; n];
        b[k] = 1.0;
        let w1_cdf = wasserstein_1d(&a, &b);
        let emd = earth_mover_distance(&Array1::from(a), &Array1::from(b), &cost1);
        println!("  shift {k:>2}: exact={k:.1}  wasserstein_1d={w1_cdf:.3}  earth_mover={emd:.3}");
    }

    println!("\n=== sinkhorn_log -> exact W1 as reg shrinks (two gaussians) ===");
    let a = gaussian_hist(n, 15.0, 4.0);
    let b = gaussian_hist(n, 32.0, 6.0);
    let exact = wasserstein_1d(&a, &b);
    let (aa, bb) = (Array1::from(a), Array1::from(b));
    println!("  exact W1 (CDF) = {exact:.4}");
    for reg in [5.0f32, 1.0, 0.3, 0.1, 0.03] {
        let (_, c) = sinkhorn_log(&aa, &bb, &cost1, reg, 1000);
        println!(
            "  reg={reg:<5} sinkhorn cost={c:.4}  rel err vs exact = {:.3}",
            (c - exact).abs() / exact
        );
    }
}
