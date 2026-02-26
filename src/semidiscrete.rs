//! Semidiscrete optimal transport (SD-OT) primitives.
//!
//! This module is intentionally small and “plumbing-level”:
//! potentials + scores + (hard) assignments, plus a deterministic SGD fitter for `g`.
//!
//! `flowmatch` uses this for SD-FM experiments.
//!
//! # References
//!
//! - Agarwal et al. (2024, NeurIPS). “A Combinatorial Algorithm for Semidiscrete OT”
//!   -- O(n log n) exact algorithm as an alternative to SGD on dual potentials
//! - Taskesen et al. (2023). “Semi-discrete OT: Hardness, Regularization and Numerical
//!   Solution” -- validates the regularized SGD approach used here
//! - Pooladian et al. (2023, ICML). “Minimax Estimation of Discontinuous OT Maps:
//!   Semidiscrete Case” -- minimax-optimal estimators; SD-OT maps are piecewise constant

use crate::{Error, Result, EPSILON};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Configuration for semidiscrete potential fitting via SGD.
#[derive(Debug, Clone)]
pub struct SemidiscreteSgdConfig {
    /// Entropic regularization. `0.0` means hard assignments (ε=0).
    pub epsilon: f32,
    /// SGD learning rate.
    pub lr: f32,
    /// Number of SGD steps.
    pub steps: usize,
    /// Batch size for sampling `x` per step.
    pub batch_size: usize,
    /// RNG seed (deterministic by default).
    pub seed: u64,
}

impl Default for SemidiscreteSgdConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.0,
            lr: 0.5,
            steps: 2_000,
            batch_size: 1_024,
            seed: 42,
        }
    }
}

/// Compute scores for a single `x` against all `y_j` using a negative dot-product cost.
///
/// Returns `g_j + <x, y_j>`.
pub fn scores_neg_dot(
    x: &ArrayView1<f32>,
    y: &ArrayView2<f32>,
    g: &ArrayView1<f32>,
) -> Array1<f32> {
    let n = y.nrows();
    debug_assert_eq!(g.len(), n);
    let mut out = Array1::zeros(n);
    for j in 0..n {
        out[j] = g[j] + y.row(j).dot(x);
    }
    out
}

/// Hard assignment `argmax(scores)`.
pub fn assign_hard_from_scores(scores: &ArrayView1<f32>) -> usize {
    let mut best = 0usize;
    let mut best_val = scores[0];
    for j in 1..scores.len() {
        let v = scores[j];
        if v > best_val {
            best = j;
            best_val = v;
        }
    }
    best
}

/// Fit potentials `g` so that induced assignments roughly match target weights `b`.
///
/// This uses the classic gradient form (soft case): `E[s(x)] - b`.
/// For `epsilon == 0`, we use hard assignments.
pub fn fit_potentials_sgd_neg_dot(
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    cfg: &SemidiscreteSgdConfig,
) -> Result<Array1<f32>> {
    let n = y.nrows();
    if b.len() != n {
        return Err(Error::LengthMismatch(n, b.len()));
    }
    if y.ncols() == 0 {
        return Err(Error::Domain("y must have positive dimension"));
    }
    if b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("b must be nonnegative"));
    }
    let bs = b.sum();
    if bs <= 0.0 {
        return Err(Error::Domain("b must have positive total mass"));
    }
    if !(cfg.lr > 0.0) || !cfg.lr.is_finite() {
        return Err(Error::Domain("lr must be positive and finite"));
    }
    if cfg.steps == 0 || cfg.batch_size == 0 {
        return Err(Error::Domain("steps and batch_size must be >= 1"));
    }
    if cfg.epsilon < 0.0 || !cfg.epsilon.is_finite() {
        return Err(Error::InvalidRegularization(cfg.epsilon));
    }

    // Normalize b for the target marginal.
    let b = b.to_owned() / (bs + EPSILON);

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    let d = y.ncols();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut g = Array1::<f32>::zeros(n);

    // Center g to reduce drift (potentials are identifiable up to a constant).
    let center = |g: &mut Array1<f32>| {
        let m = g.mean().unwrap_or(0.0);
        *g -= m;
    };

    for _ in 0..cfg.steps {
        let mut avg = vec![0.0f32; n];
        for _ in 0..cfg.batch_size {
            // Sample x ~ N(0, I).
            let mut x = Array1::<f32>::zeros(d);
            for i in 0..d {
                let v: f64 = rng.sample(StandardNormal);
                x[i] = v as f32;
            }

            let scores = scores_neg_dot(&x.view(), y, &g.view());
            if cfg.epsilon == 0.0 {
                let j = assign_hard_from_scores(&scores.view());
                avg[j] += 1.0;
            } else {
                // Soft assignment: proportional to b_j * exp(score_j / eps), normalized.
                let eps = cfg.epsilon;
                let mut maxv = f32::NEG_INFINITY;
                let mut tmp = vec![0.0f32; n];
                for j in 0..n {
                    if b[j] <= 0.0 {
                        tmp[j] = 0.0;
                        continue;
                    }
                    let v = (scores[j] / eps) + b[j].ln();
                    maxv = maxv.max(v);
                    tmp[j] = v;
                }
                let mut s = 0.0f64;
                for j in 0..n {
                    let w = (tmp[j] - maxv).exp();
                    tmp[j] = w;
                    s += w as f64;
                }
                if s > 0.0 {
                    for j in 0..n {
                        avg[j] += tmp[j] / (s as f32);
                    }
                }
            }
        }

        // avg now approximates E[s(x)] over the batch.
        let inv_bs = 1.0 / (cfg.batch_size as f32);
        for j in 0..n {
            let grad = (avg[j] * inv_bs) - b[j];
            g[j] -= cfg.lr * grad;
        }
        center(&mut g);
    }

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn scores_neg_dot_basic() {
        let x = array![1.0, 0.0];
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let g = array![0.0, 0.0];
        let s = scores_neg_dot(&x.view(), &y.view(), &g.view());
        // s[0] = 0 + <[1,0],[1,0]> = 1; s[1] = 0 + <[1,0],[0,1]> = 0
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s[1].abs() < 1e-6);
    }

    #[test]
    fn scores_neg_dot_with_potentials() {
        let x = array![1.0, 0.0];
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let g = array![-2.0, 3.0]; // bias toward y[1]
        let s = scores_neg_dot(&x.view(), &y.view(), &g.view());
        // s[0] = -2 + 1 = -1; s[1] = 3 + 0 = 3
        assert!((s[0] - (-1.0)).abs() < 1e-6);
        assert!((s[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn assign_hard_argmax() {
        let scores = array![0.1, 0.9, 0.5];
        assert_eq!(assign_hard_from_scores(&scores.view()), 1);
    }

    #[test]
    fn assign_hard_first_wins_tie() {
        // Strict > means first element wins on tie
        let scores = array![0.5, 0.5, 0.5];
        assert_eq!(assign_hard_from_scores(&scores.view()), 0);
    }

    #[test]
    fn config_default_is_sane() {
        let cfg = SemidiscreteSgdConfig::default();
        assert_eq!(cfg.epsilon, 0.0);
        assert!(cfg.lr > 0.0);
        assert!(cfg.steps > 0);
        assert!(cfg.batch_size > 0);
    }

    #[test]
    fn fit_potentials_rejects_bad_inputs() {
        let y = Array2::<f32>::zeros((3, 2));
        let b = array![0.5, 0.5]; // wrong length
        let cfg = SemidiscreteSgdConfig::default();
        assert!(fit_potentials_sgd_neg_dot(&y.view(), &b.view(), &cfg).is_err());
    }

    #[test]
    fn fit_potentials_runs_and_returns() {
        // 2 targets in 2D, uniform weights
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![0.5, 0.5];
        let cfg = SemidiscreteSgdConfig {
            steps: 100,
            batch_size: 64,
            ..Default::default()
        };
        let g = fit_potentials_sgd_neg_dot(&y.view(), &b.view(), &cfg).unwrap();
        assert_eq!(g.len(), 2);
        // Potentials are centered (mean ~0)
        assert!(g.mean().unwrap().abs() < 1e-3, "g should be centered: {:?}", g);
    }

    #[test]
    fn fit_potentials_is_deterministic() {
        let y = array![[1.0, 0.0], [-1.0, 0.0]];
        let b = array![0.5, 0.5];
        let cfg = SemidiscreteSgdConfig {
            steps: 50,
            batch_size: 32,
            seed: 123,
            ..Default::default()
        };
        let g1 = fit_potentials_sgd_neg_dot(&y.view(), &b.view(), &cfg).unwrap();
        let g2 = fit_potentials_sgd_neg_dot(&y.view(), &b.view(), &cfg).unwrap();
        assert_eq!(g1, g2, "same seed should give same result");
    }
}
