//! Closed-form Wasserstein distances between Gaussian distributions.
//!
//! The 2-Wasserstein distance between Gaussians admits a closed form
//! (Dowson & Landau 1982, Givens & Shortt 1984):
//!
//! $$
//! W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2))
//! = \|\mu_1 - \mu_2\|^2
//! + \mathrm{tr}(\Sigma_1) + \mathrm{tr}(\Sigma_2)
//! - 2\,\mathrm{tr}\!\bigl((\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2})^{1/2}\bigr)
//! $$
//!
//! For diagonal covariances `diag(s)`, the matrix square root simplifies to
//! element-wise `sqrt`, and the cross-term becomes `sum(sqrt(s1_i * s2_i))`.
//!
//! For 1D Gaussians with standard deviations `sigma1`, `sigma2`:
//!
//! $$
//! W_2^2 = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2
//! $$
//!
//! # Related crates
//! - [`qig`]: Bures distance on density matrices. For centered Gaussians, W2 = Bures distance.

use crate::{Error, Result};

/// W2 distance between 1D Gaussians `N(mu1, sigma1^2)` and `N(mu2, sigma2^2)`.
///
/// Returns the 2-Wasserstein distance (not squared). Standard deviations must
/// be non-negative.
///
/// # Formula
///
/// `W2 = sqrt((mu1 - mu2)^2 + (sigma1 - sigma2)^2)`
///
/// # Example
///
/// ```rust
/// use wass::gaussian::w2_gaussian_1d;
///
/// // Same Gaussian => distance 0.
/// assert!((w2_gaussian_1d(0.0, 1.0, 0.0, 1.0)).abs() < 1e-7);
///
/// // Pure mean shift.
/// let d = w2_gaussian_1d(0.0, 1.0, 3.0, 1.0);
/// assert!((d - 3.0).abs() < 1e-6);
///
/// // Pure variance change.
/// let d = w2_gaussian_1d(0.0, 1.0, 0.0, 2.0);
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
pub fn w2_gaussian_1d(mu1: f32, sigma1: f32, mu2: f32, sigma2: f32) -> f32 {
    let mean_sq = (mu1 - mu2) * (mu1 - mu2);
    let var_sq = (sigma1 - sigma2) * (sigma1 - sigma2);
    (mean_sq + var_sq).sqrt()
}

/// W2 distance between d-dimensional Gaussians with diagonal covariance.
///
/// `sigma1` and `sigma2` are the diagonal entries of the covariance matrices
/// (i.e. the per-dimension variances, **not** standard deviations).
///
/// Returns the 2-Wasserstein distance (not squared).
///
/// # Formula
///
/// For diagonal covariances `diag(s1)` and `diag(s2)`:
///
/// `W2^2 = ||mu1 - mu2||^2 + sum(s1_i) + sum(s2_i) - 2 * sum(sqrt(s1_i * s2_i))`
///
/// The cross-term simplifies because diagonal matrices commute.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if the slices have different lengths.
/// Returns [`Error::Domain`] if any variance entry is negative.
///
/// # Example
///
/// ```rust
/// use wass::gaussian::w2_gaussian_diagonal;
///
/// // Identical Gaussians => 0.
/// let d = w2_gaussian_diagonal(&[0.0, 0.0], &[1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0]).unwrap();
/// assert!(d < 1e-6);
/// ```
pub fn w2_gaussian_diagonal(
    mu1: &[f32],
    sigma1: &[f32],
    mu2: &[f32],
    sigma2: &[f32],
) -> Result<f32> {
    let d = mu1.len();
    if mu2.len() != d {
        return Err(Error::LengthMismatch(d, mu2.len()));
    }
    if sigma1.len() != d {
        return Err(Error::LengthMismatch(d, sigma1.len()));
    }
    if sigma2.len() != d {
        return Err(Error::LengthMismatch(d, sigma2.len()));
    }

    // Validate non-negative variances.
    if sigma1.iter().any(|&s| s < 0.0) || sigma2.iter().any(|&s| s < 0.0) {
        return Err(Error::Domain(
            "covariance diagonal entries must be non-negative",
        ));
    }

    // ||mu1 - mu2||^2
    let mean_sq: f32 = mu1
        .iter()
        .zip(mu2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();

    // tr(Sigma1) + tr(Sigma2) - 2 * tr(sqrt(Sigma1^{1/2} Sigma2 Sigma1^{1/2}))
    // For diagonal: sum(s1) + sum(s2) - 2 * sum(sqrt(s1_i * s2_i))
    let trace_term: f32 = sigma1
        .iter()
        .zip(sigma2.iter())
        .map(|(&s1, &s2)| s1 + s2 - 2.0 * (s1 * s2).sqrt())
        .sum();

    let w2_sq = mean_sq + trace_term;
    // Clamp tiny negative values from floating-point drift.
    Ok(w2_sq.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---- Deterministic tests ----

    #[test]
    fn w2_1d_self_distance_is_zero() {
        assert!(w2_gaussian_1d(1.0, 2.0, 1.0, 2.0) < 1e-7);
    }

    #[test]
    fn w2_1d_pure_mean_shift() {
        let d = w2_gaussian_1d(0.0, 1.0, 5.0, 1.0);
        assert!((d - 5.0).abs() < 1e-6, "d={d}");
    }

    #[test]
    fn w2_1d_pure_variance_change() {
        // W2 = |sigma1 - sigma2| when means are equal.
        let d = w2_gaussian_1d(0.0, 1.0, 0.0, 3.0);
        assert!((d - 2.0).abs() < 1e-6, "d={d}");
    }

    #[test]
    fn w2_1d_symmetry() {
        let ab = w2_gaussian_1d(1.0, 2.0, 3.0, 4.0);
        let ba = w2_gaussian_1d(3.0, 4.0, 1.0, 2.0);
        assert!((ab - ba).abs() < 1e-7);
    }

    #[test]
    fn w2_1d_triangle_inequality() {
        let ab = w2_gaussian_1d(0.0, 1.0, 2.0, 3.0);
        let bc = w2_gaussian_1d(2.0, 3.0, 5.0, 0.5);
        let ac = w2_gaussian_1d(0.0, 1.0, 5.0, 0.5);
        assert!(ac <= ab + bc + 1e-6, "ac={ac} > ab+bc={}", ab + bc);
    }

    #[test]
    fn w2_1d_matches_formula() {
        // W2^2 = (mu1 - mu2)^2 + (sigma1 - sigma2)^2
        let d = w2_gaussian_1d(1.0, 2.0, 4.0, 6.0);
        let expected = ((3.0f32 * 3.0) + (4.0 * 4.0)).sqrt(); // sqrt(9 + 16) = 5
        assert!((d - expected).abs() < 1e-6, "d={d} expected={expected}");
    }

    #[test]
    fn w2_diagonal_self_distance_is_zero() {
        let d = w2_gaussian_diagonal(&[1.0, 2.0], &[3.0, 4.0], &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert!(d < 1e-6, "d={d}");
    }

    #[test]
    fn w2_diagonal_pure_mean_shift() {
        // Zero covariance => W2 = ||mu1 - mu2||.
        let d = w2_gaussian_diagonal(&[0.0, 0.0], &[0.0, 0.0], &[3.0, 4.0], &[0.0, 0.0]).unwrap();
        assert!((d - 5.0).abs() < 1e-6, "d={d}");
    }

    #[test]
    fn w2_diagonal_symmetry() {
        let ab = w2_gaussian_diagonal(&[1.0, 2.0], &[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]).unwrap();
        let ba = w2_gaussian_diagonal(&[3.0, 4.0], &[5.0, 6.0], &[1.0, 2.0], &[1.0, 2.0]).unwrap();
        assert!((ab - ba).abs() < 1e-6, "ab={ab} ba={ba}");
    }

    #[test]
    fn w2_diagonal_triangle_inequality() {
        let ab = w2_gaussian_diagonal(&[0.0], &[1.0], &[3.0], &[4.0]).unwrap();
        let bc = w2_gaussian_diagonal(&[3.0], &[4.0], &[7.0], &[0.5]).unwrap();
        let ac = w2_gaussian_diagonal(&[0.0], &[1.0], &[7.0], &[0.5]).unwrap();
        assert!(ac <= ab + bc + 1e-5, "ac={ac} > ab+bc={}", ab + bc);
    }

    #[test]
    fn w2_diagonal_1d_consistency() {
        // 1D diagonal with variance s^2 should match w2_gaussian_1d with sigma=s.
        // w2_gaussian_1d uses standard deviation; w2_gaussian_diagonal uses variance.
        // W2^2(1d) = (mu1-mu2)^2 + (sigma1-sigma2)^2  [std devs]
        // W2^2(diag) = (mu1-mu2)^2 + s1 + s2 - 2*sqrt(s1*s2)
        //            = (mu1-mu2)^2 + (sqrt(s1) - sqrt(s2))^2
        // So w2_gaussian_1d(mu1, sigma1, mu2, sigma2) == w2_gaussian_diagonal(mu1, sigma1^2, mu2, sigma2^2).
        let d_1d = w2_gaussian_1d(1.0, 2.0, 4.0, 5.0);
        let d_diag = w2_gaussian_diagonal(&[1.0], &[4.0], &[4.0], &[25.0]).unwrap();
        assert!((d_1d - d_diag).abs() < 1e-5, "1d={d_1d} diag={d_diag}");
    }

    #[test]
    fn w2_diagonal_rejects_negative_variance() {
        let r = w2_gaussian_diagonal(&[0.0], &[-1.0], &[0.0], &[1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn w2_diagonal_rejects_length_mismatch() {
        let r = w2_gaussian_diagonal(&[0.0, 1.0], &[1.0], &[0.0, 1.0], &[1.0, 1.0]);
        assert!(r.is_err());
    }

    // ---- Property-based tests ----

    proptest! {
        #[test]
        fn prop_w2_1d_non_negative(
            mu1 in -100.0f32..100.0,
            sigma1 in 0.0f32..50.0,
            mu2 in -100.0f32..100.0,
            sigma2 in 0.0f32..50.0,
        ) {
            let d = w2_gaussian_1d(mu1, sigma1, mu2, sigma2);
            prop_assert!(d >= 0.0, "d={d}");
        }

        #[test]
        fn prop_w2_1d_self_distance_zero(
            mu in -100.0f32..100.0,
            sigma in 0.0f32..50.0,
        ) {
            let d = w2_gaussian_1d(mu, sigma, mu, sigma);
            prop_assert!(d < 1e-6, "d={d}");
        }

        #[test]
        fn prop_w2_1d_symmetry(
            mu1 in -100.0f32..100.0,
            sigma1 in 0.0f32..50.0,
            mu2 in -100.0f32..100.0,
            sigma2 in 0.0f32..50.0,
        ) {
            let ab = w2_gaussian_1d(mu1, sigma1, mu2, sigma2);
            let ba = w2_gaussian_1d(mu2, sigma2, mu1, sigma1);
            prop_assert!((ab - ba).abs() < 1e-5, "ab={ab} ba={ba}");
        }

        #[test]
        fn prop_w2_1d_triangle_inequality(
            mu1 in -50.0f32..50.0,
            sigma1 in 0.01f32..20.0,
            mu2 in -50.0f32..50.0,
            sigma2 in 0.01f32..20.0,
            mu3 in -50.0f32..50.0,
            sigma3 in 0.01f32..20.0,
        ) {
            let ab = w2_gaussian_1d(mu1, sigma1, mu2, sigma2);
            let bc = w2_gaussian_1d(mu2, sigma2, mu3, sigma3);
            let ac = w2_gaussian_1d(mu1, sigma1, mu3, sigma3);
            prop_assert!(ac <= ab + bc + 1e-4, "ac={ac} > ab+bc={}", ab + bc);
        }

        #[test]
        fn prop_w2_diagonal_non_negative(
            d in 1usize..6,
        ) {
            // Generate random means and variances.
            let mu1: Vec<f32> = (0..d).map(|i| (i as f32) * 1.5).collect();
            let mu2: Vec<f32> = (0..d).map(|i| (i as f32) * 0.7 + 1.0).collect();
            let s1: Vec<f32> = (0..d).map(|i| (i as f32) + 0.5).collect();
            let s2: Vec<f32> = (0..d).map(|i| (i as f32) * 2.0 + 0.1).collect();
            let dist = w2_gaussian_diagonal(&mu1, &s1, &mu2, &s2).unwrap();
            prop_assert!(dist >= 0.0, "dist={dist}");
        }

        #[test]
        fn prop_w2_diagonal_self_zero(
            d in 1usize..6,
        ) {
            let mu: Vec<f32> = (0..d).map(|i| (i as f32) * 1.5).collect();
            let s: Vec<f32> = (0..d).map(|i| (i as f32) + 0.5).collect();
            let dist = w2_gaussian_diagonal(&mu, &s, &mu, &s).unwrap();
            prop_assert!(dist < 1e-6, "dist={dist}");
        }

        #[test]
        fn prop_w2_diagonal_symmetry(
            d in 1usize..4,
        ) {
            let mu1: Vec<f32> = (0..d).map(|i| (i as f32) * 1.5).collect();
            let mu2: Vec<f32> = (0..d).map(|i| (i as f32) * 0.7 + 1.0).collect();
            let s1: Vec<f32> = (0..d).map(|i| (i as f32) + 0.5).collect();
            let s2: Vec<f32> = (0..d).map(|i| (i as f32) * 2.0 + 0.1).collect();
            let ab = w2_gaussian_diagonal(&mu1, &s1, &mu2, &s2).unwrap();
            let ba = w2_gaussian_diagonal(&mu2, &s2, &mu1, &s1).unwrap();
            prop_assert!((ab - ba).abs() < 1e-5, "ab={ab} ba={ba}");
        }
    }
}
