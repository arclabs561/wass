//! Sorting paths must not panic on NaN (Rust 1.81+ sort can panic when a
//! comparator is not a total order). The result may be NaN or an error.

use ndarray::array;
use wass::{sinkhorn_hierarchical, sliced_wasserstein};

#[test]
fn hierarchical_sinkhorn_with_nan_costs_does_not_panic() {
    let n = 32;
    let m = 32;
    let a = vec![1.0 / n as f32; n];
    let b = vec![1.0 / m as f32; m];
    let cost: Vec<f32> = (0..n * m)
        .map(|k| if k % 3 == 0 { f32::NAN } else { (k % 7) as f32 })
        .collect();
    let _ = sinkhorn_hierarchical(&a, &b, &cost, n, m, 0.5, 4, 2, 50, 1e-4);
}

#[test]
fn sliced_wasserstein_with_nan_points_does_not_panic() {
    let x = array![[0.0f32, f32::NAN], [1.0, 2.0], [f32::NAN, 0.5]];
    let y = array![[0.5f32, 0.5], [2.0, f32::NAN]];
    let _ = sliced_wasserstein(&x, &y, 16, 3, 2.0);
}
