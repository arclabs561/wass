//! Entropic Wasserstein barycenters on a fixed shared support, via iterative
//! Bregman projections (Benamou et al. 2015; Cuturi & Doucet 2014).
//!
//! Given `S` distributions on the SAME `n`-point grid and a shared `n x n`
//! ground cost, [`barycenter`] returns the weighted barycenter on that grid.
//!
//! The result is the ENTROPIC (regularized) barycenter: it is slightly
//! blurred by the entropic regularizer `reg`. It is not the exact
//! unregularized Wasserstein barycenter; smaller `reg` is sharper but relies
//! on the log-domain stabilization this module uses.
//!
//! The iteration runs in the log domain on purpose. The equivalent
//! linear-domain form (`a_k = p_k / (K b_k)`, `m_k = K^T a_k`, geometric
//! mean, `b_k = bary / m_k`) silently degrades to the arithmetic
//! average-of-histograms when `reg` is small, because the kernel
//! `exp(-C/reg)` underflows off-diagonal and mass cannot cross gaps in the
//! support. The log domain reuses the same `logsumexp` machinery the Sinkhorn
//! solver uses and stays correct at small `reg`.

use crate::{logsumexp_by, Error, Result};
use ndarray::{Array1, Array2};

/// Weighted entropic Wasserstein barycenter of `dists` on a shared `n`-point
/// grid with shared `n x n` `cost`. `weights` are normalized internally.
///
/// See the [module docs](self) for the entropic-blur caveat. Errors on a
/// length/shape mismatch, non-positive `reg`, or non-positive weight sum.
pub fn barycenter(
    dists: &[Array1<f32>],
    cost: &Array2<f32>,
    weights: &[f32],
    reg: f32,
    max_iter: usize,
) -> Result<Array1<f32>> {
    barycenter_with_convergence(dists, cost, weights, reg, max_iter).map(|(b, _)| b)
}

/// Like [`barycenter`] but also returns the iteration count at which the
/// barycenter's log-marginal stopped changing (== `max_iter` if it never did).
pub fn barycenter_with_convergence(
    dists: &[Array1<f32>],
    cost: &Array2<f32>,
    weights: &[f32],
    reg: f32,
    max_iter: usize,
) -> Result<(Array1<f32>, usize)> {
    if !(reg.is_finite() && reg > 0.0) {
        return Err(Error::InvalidRegularization(reg));
    }
    if dists.is_empty() {
        return Err(Error::Domain("barycenter requires at least one distribution"));
    }
    if weights.len() != dists.len() {
        return Err(Error::Domain(
            "weights length must equal the number of distributions",
        ));
    }
    let n = dists[0].len();
    for d in dists {
        if d.len() != n {
            return Err(Error::LengthMismatch(n, d.len()));
        }
    }
    let (cr, cc) = (cost.shape()[0], cost.shape()[1]);
    if cr != n || cc != n {
        return Err(Error::CostShapeMismatch(n, n, cr, cc));
    }
    let w_sum: f32 = weights.iter().sum();
    if !(w_sum.is_finite() && w_sum > 0.0) {
        return Err(Error::Domain("weights must sum to a positive finite value"));
    }
    let lambda: Vec<f32> = weights.iter().map(|w| w / w_sum).collect();
    let s = dists.len();

    // log p_k, normalized, with -inf for zero-mass bins (hard support
    // exclusion, matching sinkhorn_log).
    let log_p: Vec<Array1<f32>> = dists
        .iter()
        .map(|d| {
            let sum = d.sum();
            d.mapv(|x| {
                let v = x / (sum + f32::EPSILON);
                if v <= 0.0 {
                    f32::NEG_INFINITY
                } else {
                    v.ln()
                }
            })
        })
        .collect();

    // log of the per-distribution scaling vectors b_k (b_k = ones => log 0).
    let mut log_b: Vec<Array1<f32>> = vec![Array1::zeros(n); s];
    let mut log_bary: Array1<f32> = Array1::zeros(n);
    let mut iters = max_iter;

    for it in 0..max_iter {
        // m_k = K^T (p_k / (K b_k)), all in log space. log_K[i,j] = -C[i,j]/reg.
        let mut log_m: Vec<Array1<f32>> = Vec::with_capacity(s);
        for k in 0..s {
            // log_a_k[i] = log_p_k[i] - logsumexp_j(-C[i,j]/reg + log_b_k[j])
            let mut log_a: Array1<f32> = Array1::zeros(n);
            for i in 0..n {
                let lse = logsumexp_by(n, |j| -cost[[i, j]] / reg + log_b[k][j]);
                log_a[i] = log_p[k][i] - lse;
            }
            // log_m_k[j] = logsumexp_i(-C[i,j]/reg + log_a_k[i])  (cost symmetric)
            let mut log_mk: Array1<f32> = Array1::zeros(n);
            for j in 0..n {
                log_mk[j] = logsumexp_by(n, |i| -cost[[i, j]] / reg + log_a[i]);
            }
            log_m.push(log_mk);
        }

        // log_bary[j] = sum_k lambda_k * log_m_k[j]  (weighted geometric mean)
        let mut new_log_bary: Array1<f32> = Array1::zeros(n);
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..s {
                acc += lambda[k] * log_m[k][j];
            }
            new_log_bary[j] = acc;
        }

        // b_k = bary / m_k  =>  log_b_k = log_bary - log_m_k
        for k in 0..s {
            for j in 0..n {
                log_b[k][j] = new_log_bary[j] - log_m[k][j];
            }
        }

        let delta = new_log_bary
            .iter()
            .zip(log_bary.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        log_bary = new_log_bary;
        if delta < 1e-6 {
            iters = it + 1;
            break;
        }
    }

    // Normalize exp(log_bary) into a probability histogram.
    let max = log_bary.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut bary = log_bary.mapv(|v| (v - max).exp());
    let sum = bary.sum();
    bary.mapv_inplace(|v| v / (sum + f32::EPSILON));
    Ok((bary, iters))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1D grid of `n` points, squared-distance ground cost (W2).
    fn sq_cost(n: usize) -> Array2<f32> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            let d = i as f32 - j as f32;
            d * d
        })
    }

    // Gaussian histogram on 0..n.
    fn gaussian(n: usize, mean: f32, std: f32) -> Array1<f32> {
        let mut v = Array1::from_shape_fn(n, |i| {
            let z = (i as f32 - mean) / std;
            (-0.5 * z * z).exp()
        });
        let s = v.sum();
        v.mapv_inplace(|x| x / s);
        v
    }

    fn mean_of(h: &Array1<f32>) -> f32 {
        h.iter().enumerate().map(|(i, &p)| i as f32 * p).sum()
    }

    // The W2 barycenter of two Gaussians has mean = weighted average of the
    // input means. The asymmetric-weight case is the discriminating test: a
    // collapse-to-average-of-histograms bug would still pass symmetric weights
    // (mean 25 by symmetry) but fail this one.
    #[test]
    fn asymmetric_weights_interpolate_the_mean() {
        let n = 51;
        let cost = sq_cost(n);
        let p = gaussian(n, 15.0, 3.0);
        let q = gaussian(n, 35.0, 3.0);
        let reg = 8.0;

        let b = barycenter(&[p.clone(), q.clone()], &cost, &[0.8, 0.2], reg, 500).unwrap();
        // expected mean = 0.8*15 + 0.2*35 = 19.0
        let m = mean_of(&b);
        assert!(
            (m - 19.0).abs() < 1.5,
            "asymmetric barycenter mean {m} not near 19.0 (would be ~25 if collapsed to histogram average)"
        );
        // it is a normalized histogram
        assert!((b.sum() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn symmetric_weights_center() {
        let n = 51;
        let cost = sq_cost(n);
        let p = gaussian(n, 15.0, 3.0);
        let q = gaussian(n, 35.0, 3.0);
        let b = barycenter(&[p, q], &cost, &[0.5, 0.5], 8.0, 500).unwrap();
        let m = mean_of(&b);
        assert!((m - 25.0).abs() < 1.0, "symmetric barycenter mean {m} not near midpoint 25");
    }

    #[test]
    fn three_inputs_average() {
        let n = 61;
        let cost = sq_cost(n);
        let ds = [gaussian(n, 10.0, 3.0), gaussian(n, 30.0, 3.0), gaussian(n, 50.0, 3.0)];
        let b = barycenter(&ds, &cost, &[1.0, 1.0, 1.0], 8.0, 500).unwrap();
        let m = mean_of(&b);
        assert!((m - 30.0).abs() < 1.0, "three-input barycenter mean {m} not near 30");
    }

    #[test]
    fn rejects_bad_inputs() {
        let n = 10;
        let cost = sq_cost(n);
        let p = gaussian(n, 3.0, 1.0);
        let q = gaussian(n, 6.0, 1.0);
        assert!(barycenter(&[p.clone(), q.clone()], &cost, &[0.5, 0.5], 0.0, 10).is_err()); // reg
        assert!(barycenter(&[], &cost, &[], 1.0, 10).is_err()); // empty
        assert!(barycenter(&[p.clone(), q.clone()], &cost, &[1.0], 1.0, 10).is_err()); // weights len
        let short = gaussian(n - 1, 3.0, 1.0);
        assert!(barycenter(&[p, short], &cost, &[0.5, 0.5], 1.0, 10).is_err()); // length
    }
}
