//! Wasserstein-Fisher-Rao (Hellinger-Kantorovich) distance.
//!
//! The WFR distance (Chizat, Peyre, Schmitzer & Vialard 2018; also called
//! Hellinger-Kantorovich) interpolates between Wasserstein (pure transport)
//! and Fisher-Rao (pure mass creation/destruction).
//!
//! Unlike balanced OT, WFR handles **unnormalized** measures: the input
//! histograms need not sum to the same value.
//!
//! The parameter `rho` controls the tradeoff:
//! - Large `rho`: transport is cheap relative to mass change, approaches Wasserstein.
//! - Small `rho`: mass creation/destruction is cheap, approaches Fisher-Rao.
//!
//! ## Algorithm
//!
//! WFR can be computed via unbalanced Sinkhorn with a modified cost matrix:
//!
//! $$
//! c_{\mathrm{WFR}}(x,y) = -\log\!\bigl(\cos^2\!\bigl(\min\bigl(\tfrac{d(x,y)}{2\rho},\, \tfrac{\pi}{2}\bigr)\bigr)\bigr)
//! $$
//!
//! This is then fed into the standard KL-penalized unbalanced Sinkhorn solver
//! with marginal penalty `rho`.
//!
//! ## References
//!
//! - Chizat, Peyre, Schmitzer & Vialard (2018). "Scaling Algorithms for Unbalanced
//!   Optimal Transport Problems" (Mathematics of Computation).
//! - Liero, Mielke & Savaré (2018). "Optimal Entropy-Transport Problems and a New
//!   Hellinger-Kantorovich Distance Between Positive Measures".

use ndarray::{Array1, Array2};

use crate::{Error, Result};

/// Compute the WFR-modified cost matrix from a ground cost matrix.
///
/// Applies the transformation:
/// `c_wfr[i,j] = -log(cos^2(min(sqrt(cost[i,j]) / (2*rho), pi/2)))`
///
/// where `cost[i,j]` is the **squared** ground distance (as is standard in OT).
/// If the user provides non-squared distances, they should square them first or
/// use [`wfr_cost_from_distance`].
///
/// Entries where `d(x,y) >= pi * rho` are clamped to `cos(pi/2) = 0`, giving
/// `c_wfr = +inf` in exact arithmetic. We use a large finite value instead.
fn wfr_cost_from_sq_distance(cost_sq: &Array2<f32>, rho: f32) -> Array2<f32> {
    let half_pi = std::f32::consts::FRAC_PI_2;
    cost_sq.mapv(|c_sq| {
        let d = c_sq.max(0.0).sqrt();
        let angle = (d / (2.0 * rho)).min(half_pi);
        let cos_val = angle.cos();
        if cos_val <= 1e-12 {
            // At the boundary: mass creation/destruction is strictly cheaper than transport.
            // Use large finite value (not infinity) for numerical stability.
            30.0
        } else {
            -(cos_val * cos_val).ln()
        }
    })
}

/// Wasserstein-Fisher-Rao distance between unnormalized histograms on the same support.
///
/// Computes the WFR (Hellinger-Kantorovich) distance by transforming the ground
/// cost matrix and solving a debiased unbalanced entropic OT problem (Sinkhorn
/// divergence). The debiasing ensures `wfr_distance(a, a, ...) == 0`.
///
/// The input `cost` must be a **square** matrix (same support for both measures).
/// For different supports, transform cost matrices externally and use the
/// unbalanced Sinkhorn divergence functions directly.
///
/// # Arguments
///
/// * `a` - Source histogram (non-negative, need not sum to 1)
/// * `b` - Target histogram (non-negative, need not sum to 1, same length as `a`)
/// * `cost` - Ground **squared** distance matrix (n x n, symmetric).
/// * `rho` - Transport/creation tradeoff. Larger values favor transport over
///   mass creation/destruction.
/// * `reg` - Entropic regularization strength (epsilon > 0).
/// * `max_iter` - Maximum Sinkhorn iterations.
/// * `tol` - Convergence tolerance.
///
/// # Returns
///
/// The WFR distance (square root of the debiased objective).
///
/// # Errors
///
/// Returns an error if inputs are invalid (negative masses, bad dimensions,
/// non-positive reg/rho) or if the solver does not converge.
///
/// # Example
///
/// ```rust
/// use wass::wfr::wfr_distance;
/// use ndarray::array;
///
/// let a = array![0.5, 0.3, 0.2];
/// let b = array![0.5, 0.3, 0.2];
/// let cost = array![[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]];
///
/// let d = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
/// assert!(d < 0.05, "identical measures should have near-zero WFR distance");
/// ```
pub fn wfr_distance(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    rho: f32,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let n = a.len();
    if b.len() != n {
        return Err(Error::LengthMismatch(n, b.len()));
    }
    if cost.nrows() != n || cost.ncols() != n {
        return Err(Error::CostShapeMismatch(n, n, cost.nrows(), cost.ncols()));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(Error::InvalidMassPenalty(rho));
    }
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("WFR requires non-negative masses"));
    }
    if a.sum() <= 0.0 || b.sum() <= 0.0 {
        return Err(Error::Domain("WFR requires positive total mass"));
    }

    // Transform the ground cost matrix for WFR.
    let wfr_cost = wfr_cost_from_sq_distance(cost, rho);

    // Use debiased unbalanced Sinkhorn divergence so that d(a,a) = 0.
    let div = crate::unbalanced_sinkhorn_divergence_same_support(
        a, b, &wfr_cost, reg, rho, max_iter, tol,
    )?;

    // WFR distance is the square root of the divergence value.
    Ok(div.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    /// Build a symmetric squared-distance cost matrix from 1D positions.
    fn sq_cost_1d(positions: &[f32]) -> Array2<f32> {
        let n = positions.len();
        let mut c = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let d = positions[i] - positions[j];
                c[[i, j]] = d * d;
            }
        }
        c
    }

    // ---- Deterministic tests ----

    #[test]
    fn wfr_self_distance_is_zero() {
        let a = array![0.3, 0.5, 0.2];
        let cost = sq_cost_1d(&[0.0, 1.0, 2.0]);
        let d = wfr_distance(&a, &a, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
        assert!(d < 0.1, "self-distance should be near zero: d={d}");
    }

    #[test]
    fn wfr_symmetry() {
        let a = array![0.5, 0.3, 0.2];
        let b = array![0.2, 0.4, 0.4];
        let cost = sq_cost_1d(&[0.0, 1.0, 3.0]);
        let ab = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
        let ba = wfr_distance(&b, &a, &cost.t().to_owned(), 1.0, 0.1, 500, 1e-4).unwrap();
        assert!(
            (ab - ba).abs() < 0.1,
            "WFR should be symmetric: ab={ab} ba={ba}"
        );
    }

    #[test]
    fn wfr_different_total_mass() {
        // WFR should handle measures with different total mass without requiring normalization.
        let a = array![1.0, 1.0, 1.0]; // total mass 3
        let b = array![0.1, 0.1, 0.1]; // total mass 0.3
        let cost = sq_cost_1d(&[0.0, 1.0, 2.0]);
        let d = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
        assert!(
            d > 0.0,
            "different-mass measures should have positive distance: d={d}"
        );
    }

    #[test]
    fn wfr_identical_measures_different_rho() {
        let a = array![0.5, 0.5];
        let cost = sq_cost_1d(&[0.0, 1.0]);
        // For identical measures, WFR should be near zero regardless of rho.
        for &rho in &[0.1, 1.0, 10.0] {
            let d = wfr_distance(&a, &a, &cost, rho, 0.1, 500, 1e-4).unwrap();
            assert!(
                d < 0.15,
                "self-distance should be near zero for rho={rho}: d={d}"
            );
        }
    }

    #[test]
    fn wfr_positive_for_different_measures() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let cost = sq_cost_1d(&[0.0, 2.0]);
        let d = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
        assert!(
            d > 0.01,
            "different measures should have positive distance: d={d}"
        );
    }

    #[test]
    fn wfr_rejects_negative_mass() {
        let a = array![-0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = sq_cost_1d(&[0.0, 1.0]);
        assert!(wfr_distance(&a, &b, &cost, 1.0, 0.1, 100, 1e-4).is_err());
    }

    #[test]
    fn wfr_rejects_invalid_rho() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = sq_cost_1d(&[0.0, 1.0]);
        assert!(wfr_distance(&a, &b, &cost, -1.0, 0.1, 100, 1e-4).is_err());
        assert!(wfr_distance(&a, &b, &cost, 0.0, 0.1, 100, 1e-4).is_err());
    }

    #[test]
    fn wfr_rejects_invalid_reg() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = sq_cost_1d(&[0.0, 1.0]);
        assert!(wfr_distance(&a, &b, &cost, 1.0, 0.0, 100, 1e-4).is_err());
        assert!(wfr_distance(&a, &b, &cost, 1.0, -0.1, 100, 1e-4).is_err());
    }

    #[test]
    fn wfr_rejects_shape_mismatch() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5, 0.0];
        let cost = Array2::zeros((2, 2));
        assert!(wfr_distance(&a, &b, &cost, 1.0, 0.1, 100, 1e-4).is_err());
    }

    // ---- Property-based tests ----

    proptest! {
        #[test]
        fn prop_wfr_non_negative(
            n in 2usize..5,
        ) {
            let a_vec: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let b_vec: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.2).collect();
            let positions: Vec<f32> = (0..n).map(|i| i as f32).collect();

            let a = Array1::from_vec(a_vec);
            let b = Array1::from_vec(b_vec);
            let cost = sq_cost_1d(&positions);

            let d = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-3).unwrap();
            prop_assert!(d >= -1e-4, "d={d}");
        }

        #[test]
        fn prop_wfr_self_distance_near_zero(
            n in 2usize..5,
        ) {
            let a_vec: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let positions: Vec<f32> = (0..n).map(|i| i as f32).collect();

            let a = Array1::from_vec(a_vec);
            let cost = sq_cost_1d(&positions);

            let d = wfr_distance(&a, &a, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
            prop_assert!(d < 0.15, "self-distance too large: d={d}");
        }

        #[test]
        fn prop_wfr_symmetry(
            n in 2usize..4,
        ) {
            let a_vec: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let b_vec: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.4).collect();
            let positions: Vec<f32> = (0..n).map(|i| i as f32).collect();

            let a = Array1::from_vec(a_vec);
            let b = Array1::from_vec(b_vec);
            let cost = sq_cost_1d(&positions);

            let ab = wfr_distance(&a, &b, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
            // Symmetric cost matrix => transpose is the same.
            let ba = wfr_distance(&b, &a, &cost, 1.0, 0.1, 500, 1e-4).unwrap();
            prop_assert!((ab - ba).abs() < 0.15, "ab={ab} ba={ba}");
        }
    }
}
