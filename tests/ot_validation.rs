//! OT solvers against exact 1D references.
//!
//! On a unit-spaced 1D grid the Wasserstein-1 distance has a closed form
//! (`wasserstein_1d` = sum of |CDF| differences), and a point mass shifted by k
//! bins is exactly W1 = k. Entropic OT (`sinkhorn_log`) must converge to that
//! exact value as the regularization shrinks. Disagreement beyond the entropic
//! bias is a bug.

use ndarray::{Array1, Array2};
use wass::{earth_mover_distance, sinkhorn_log, wasserstein_1d};

const N: usize = 50;

fn grid_cost(n: usize, p: f32) -> Array2<f32> {
    Array2::from_shape_fn((n, n), |(i, j)| (i as f32 - j as f32).abs().powf(p))
}

fn gaussian_hist(n: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| {
            let z = (i as f32 - mean) / std;
            (-0.5 * z * z).exp()
        })
        .collect();
    let s: f32 = v.iter().sum();
    for x in v.iter_mut() {
        *x /= s;
    }
    v
}

fn point_mass(n: usize, at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    v[at] = 1.0;
    v
}

#[test]
fn point_mass_shift_has_w1_equal_to_shift() {
    let cost1 = grid_cost(N, 1.0);
    for k in [1usize, 7, 20] {
        let a = point_mass(N, 0);
        let b = point_mass(N, k);

        let w1_cdf = wasserstein_1d(&a, &b);
        assert!(
            (w1_cdf - k as f32).abs() < 1e-4,
            "shift {k}: wasserstein_1d = {w1_cdf}"
        );

        let emd = earth_mover_distance(&Array1::from(a), &Array1::from(b), &cost1);
        assert!(
            (emd - k as f32).abs() < 1e-3,
            "shift {k}: earth_mover_distance = {emd}"
        );
    }
}

/// Measured on 2026-11-04 (debug build): rel err 8.7e-4 at reg=5, 6.5e-5 at
/// reg=1, then 3e-7 to 3e-6 (the f32 floor) for reg in 0.3, 0.1, 0.03.
/// Tolerances leave at least 10x headroom.
#[test]
fn sinkhorn_log_converges_to_exact_w1_as_reg_shrinks() {
    let cost1 = grid_cost(N, 1.0);
    let a = gaussian_hist(N, 15.0, 4.0);
    let b = gaussian_hist(N, 32.0, 6.0);
    let exact = wasserstein_1d(&a, &b);
    let (aa, bb) = (Array1::from(a), Array1::from(b));

    let mut prev_err = f32::INFINITY;
    for (reg, tol) in [
        (5.0f32, 1e-2),
        (1.0, 1e-3),
        (0.3, 1e-4),
        (0.1, 1e-4),
        (0.03, 1e-4),
    ] {
        let (_, c) = sinkhorn_log(&aa, &bb, &cost1, reg, 1000);
        let rel_err = (c - exact).abs() / exact;
        assert!(
            rel_err < tol,
            "reg={reg}: sinkhorn cost {c} vs exact {exact}, rel err {rel_err} >= {tol}"
        );
        // Entropic bias shrinks with reg; allow f32 noise once both are tiny.
        assert!(
            rel_err <= prev_err + 1e-5,
            "reg={reg}: rel err {rel_err} grew from {prev_err}"
        );
        prev_err = rel_err;
    }
}
