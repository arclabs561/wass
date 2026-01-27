//! Semidiscrete optimal transport (SD-OT) primitives.
//!
//! This module is intentionally small and “plumbing-level”:
//! potentials + scores + (hard) assignments, plus a deterministic SGD fitter for `g`.
//!
//! `flowmatch` uses this for SD-FM experiments.

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
pub fn scores_neg_dot(x: &ArrayView1<f32>, y: &ArrayView2<f32>, g: &ArrayView1<f32>) -> Array1<f32> {
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
    let mut center = |g: &mut Array1<f32>| {
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

