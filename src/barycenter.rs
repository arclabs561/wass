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

use crate::{logsumexp_by, sinkhorn_log, sq_euclidean_cost_matrix, Error, Result};
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
        return Err(Error::Domain(
            "barycenter requires at least one distribution",
        ));
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

/// Free-support (Cuturi & Doucet 2014) Wasserstein barycenter. Unlike
/// [`barycenter`], which fixes a shared grid and moves only the weights, here
/// the support POINTS move: the barycenter is `n_support` points in the same
/// `d`-dimensional space as the inputs.
///
/// Each entry of `measures` is a `(support, weights)` pair: `support` is a
/// `k_i x d` point cloud and `weights` is a length-`k_i` mass vector
/// (normalized internally). `weights` (the second argument) are the per-measure
/// mixing coefficients, also normalized internally.
///
/// The barycenter's own weights are held uniform (`1 / n_support`); only the
/// positions are optimized, by alternating entropic OT ([`sinkhorn_log`]) from
/// the current barycenter to each measure with a barycentric-projection update
/// of the positions. `init` seeds the support; when `None`, the first
/// `n_support` points of the first measure are used (cycled if it has fewer).
///
/// Returns `(support, weights)` for the barycenter: an `n_support x d` array and
/// the uniform weight vector. The same entropic-blur caveat as [`barycenter`]
/// applies: smaller `reg` is sharper but relies on the log-domain Sinkhorn, and
/// finite `reg` biases the spread of the recovered support slightly inward.
///
/// Errors on an empty input, a `weights`/`measures` length mismatch, a
/// support/weight shape mismatch, mixed dimensions, `n_support == 0`,
/// non-positive `reg`, or a non-positive weight sum.
#[allow(clippy::too_many_arguments)]
pub fn free_support_barycenter(
    measures: &[(Array2<f32>, Array1<f32>)],
    weights: &[f32],
    n_support: usize,
    reg: f32,
    sinkhorn_iter: usize,
    outer_iter: usize,
    init: Option<Array2<f32>>,
) -> Result<(Array2<f32>, Array1<f32>)> {
    if !(reg.is_finite() && reg > 0.0) {
        return Err(Error::InvalidRegularization(reg));
    }
    if measures.is_empty() {
        return Err(Error::Domain(
            "free_support_barycenter requires at least one measure",
        ));
    }
    if weights.len() != measures.len() {
        return Err(Error::Domain(
            "weights length must equal the number of measures",
        ));
    }
    if n_support == 0 {
        return Err(Error::Domain("n_support must be positive"));
    }
    let d = measures[0].0.ncols();
    for (x, a) in measures {
        if x.ncols() != d {
            return Err(Error::Domain("all measures must share the same dimension"));
        }
        if x.nrows() != a.len() {
            return Err(Error::LengthMismatch(x.nrows(), a.len()));
        }
        if x.nrows() == 0 {
            return Err(Error::Domain(
                "each measure needs at least one support point",
            ));
        }
    }
    let w_sum: f32 = weights.iter().sum();
    if !(w_sum.is_finite() && w_sum > 0.0) {
        return Err(Error::Domain("weights must sum to a positive finite value"));
    }
    let lambda: Vec<f32> = weights.iter().map(|w| w / w_sum).collect();

    // Seed the support. Default: first n_support points of measure 0 (cycled).
    let mut y: Array2<f32> = match init {
        Some(y0) => {
            if y0.nrows() != n_support || y0.ncols() != d {
                return Err(Error::Domain("init must be n_support x d"));
            }
            y0
        }
        None => {
            let src = &measures[0].0;
            Array2::from_shape_fn((n_support, d), |(j, c)| src[[j % src.nrows(), c]])
        }
    };

    // Uniform barycenter weights, held fixed (positions-only optimization).
    let b: Array1<f32> = Array1::from_elem(n_support, 1.0 / n_support as f32);

    for _ in 0..outer_iter {
        // Y_new[j] = (1/b_j) * sum_i lambda_i * (T_i @ X_i)[j], where T_i is the
        // OT plan from the barycenter (row marginal b) to measure i.
        let mut y_new: Array2<f32> = Array2::zeros((n_support, d));
        for (i, (x_i, a_i)) in measures.iter().enumerate() {
            let cost = sq_euclidean_cost_matrix(&y, x_i); // n_support x k_i
            let (plan, _) = sinkhorn_log(&b, a_i, &cost, reg, sinkhorn_iter);
            let tx = plan.dot(x_i); // n_support x d (row j sums plan over X_i)
            for j in 0..n_support {
                let inv_bj = 1.0 / b[j];
                for c in 0..d {
                    y_new[[j, c]] += lambda[i] * tx[[j, c]] * inv_bj;
                }
            }
        }
        let delta = y_new
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        y = y_new;
        if delta < 1e-5 {
            break;
        }
    }

    Ok((y, b))
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
        assert!(
            (m - 25.0).abs() < 1.0,
            "symmetric barycenter mean {m} not near midpoint 25"
        );
    }

    #[test]
    fn three_inputs_average() {
        let n = 61;
        let cost = sq_cost(n);
        let ds = [
            gaussian(n, 10.0, 3.0),
            gaussian(n, 30.0, 3.0),
            gaussian(n, 50.0, 3.0),
        ];
        let b = barycenter(&ds, &cost, &[1.0, 1.0, 1.0], 8.0, 500).unwrap();
        let m = mean_of(&b);
        assert!(
            (m - 30.0).abs() < 1.0,
            "three-input barycenter mean {m} not near 30"
        );
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

    // ── free-support barycenter ──────────────────────────────────────────────

    // A 1D point cloud: n points at mean + std * z, z evenly spaced in [-2, 2].
    // Returned as a k x 1 array (the free-support API takes d-dimensional points).
    fn cloud_1d(n: usize, mean: f32, std: f32) -> Array2<f32> {
        Array2::from_shape_fn((n, 1), |(i, _)| {
            let z = -2.0 + 4.0 * (i as f32) / ((n - 1) as f32);
            mean + std * z
        })
    }

    fn col_mean(y: &Array2<f32>) -> f32 {
        y.column(0).mean().unwrap()
    }
    fn col_std(y: &Array2<f32>) -> f32 {
        let m = col_mean(y);
        let var = y.column(0).iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / y.nrows() as f32;
        var.sqrt()
    }

    // The 1D free-support W2 barycenter of two point clouds has, in closed form
    // (1D Bures): mean = sum lambda_i * mean_i, std = sum lambda_i * std_i. The
    // ASYMMETRIC-weight case is discriminating: a "just pool the points" bug
    // would land the mean at the midpoint (20) regardless of weights, not at 17.
    #[test]
    fn free_support_asymmetric_mean_interpolates() {
        let k = 21;
        let x1 = cloud_1d(k, 10.0, 2.0);
        let x2 = cloud_1d(k, 30.0, 2.0);
        let a1 = Array1::from_elem(k, 1.0 / k as f32);
        let a2 = Array1::from_elem(k, 1.0 / k as f32);

        let (m1, s1) = (col_mean(&x1), col_std(&x1));
        let (m2, s2) = (col_mean(&x2), col_std(&x2));
        let (l1, l2) = (0.7f32, 0.3f32);

        let (y, b) =
            free_support_barycenter(&[(x1, a1), (x2, a2)], &[l1, l2], k, 0.5, 300, 60, None)
                .unwrap();

        let my = col_mean(&y);
        let expected_mean = l1 * m1 + l2 * m2; // = 17.0
        assert!(
            (my - expected_mean).abs() < 1.0,
            "free-support mean {my} not near {expected_mean} (would be ~20 if points were just pooled)"
        );
        // Bures std oracle (entropic blur shrinks the spread inward, so a one-
        // sided lower tolerance: never wider than the closed form, not far below).
        let sy = col_std(&y);
        let expected_std = l1 * s1 + l2 * s2;
        assert!(
            sy <= expected_std + 0.2 && sy > 0.4 * expected_std,
            "free-support std {sy} not a plausibly-shrunk version of Bures std {expected_std}"
        );
        // uniform weights, normalized
        assert!((b.sum() - 1.0).abs() < 1e-5);
        assert!((b[0] - 1.0 / k as f32).abs() < 1e-6);
    }

    #[test]
    fn free_support_rejects_bad_inputs() {
        let k = 5;
        let x = cloud_1d(k, 1.0, 1.0);
        let a = Array1::from_elem(k, 1.0 / k as f32);
        let m = vec![(x.clone(), a.clone())];
        // bad reg
        assert!(free_support_barycenter(&m, &[1.0], k, 0.0, 10, 5, None).is_err());
        // empty
        assert!(free_support_barycenter(&[], &[], k, 1.0, 10, 5, None).is_err());
        // weights len mismatch
        assert!(free_support_barycenter(&m, &[0.5, 0.5], k, 1.0, 10, 5, None).is_err());
        // n_support == 0
        assert!(free_support_barycenter(&m, &[1.0], 0, 1.0, 10, 5, None).is_err());
        // support/weight shape mismatch
        let bad = vec![(x, Array1::from_elem(k - 1, 1.0))];
        assert!(free_support_barycenter(&bad, &[1.0], k, 1.0, 10, 5, None).is_err());
    }

    // Apply a 2D rigid motion p -> R(theta) p + (tx, ty) to every row.
    fn rigid(points: &Array2<f32>, theta: f32, tx: f32, ty: f32) -> Array2<f32> {
        let (c, s) = (theta.cos(), theta.sin());
        Array2::from_shape_fn((points.nrows(), 2), |(i, col)| {
            let (x, y) = (points[[i, 0]], points[[i, 1]]);
            if col == 0 {
                c * x - s * y + tx
            } else {
                s * x + c * y + ty
            }
        })
    }

    use proptest::prelude::*;

    proptest! {
        // Rotation/translation EQUIVARIANCE: the squared-Euclidean ground cost
        // is invariant under rigid motions, so the free-support barycenter is
        // equivariant. Rigidly transforming every input must rigidly transform
        // the barycenter by the same motion. Default-seeded init is derived from
        // measure 0, so it transforms consistently and the whole iteration
        // commutes with the motion. This is the principled "rotation test" for OT.
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn free_support_is_rigid_equivariant(
            theta in 0.0f32..std::f32::consts::TAU,
            tx in -4.0f32..4.0,
            ty in -4.0f32..4.0,
        ) {
            // Two fixed 2D clouds (4 points each).
            let x1 = ndarray::array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
            let x2 = ndarray::array![[3.0f32, 3.0], [4.0, 3.0], [3.0, 4.0], [4.0, 4.0]];
            let a = Array1::from_elem(4, 0.25f32);
            let lam = [0.6f32, 0.4];

            let (y_base, _) = free_support_barycenter(
                &[(x1.clone(), a.clone()), (x2.clone(), a.clone())],
                &lam, 4, 1.0, 200, 50, None,
            ).unwrap();

            let (y_moved, _) = free_support_barycenter(
                &[
                    (rigid(&x1, theta, tx, ty), a.clone()),
                    (rigid(&x2, theta, tx, ty), a.clone()),
                ],
                &lam, 4, 1.0, 200, 50, None,
            ).unwrap();

            let y_base_moved = rigid(&y_base, theta, tx, ty);
            for j in 0..4 {
                for col in 0..2 {
                    prop_assert!(
                        (y_moved[[j, col]] - y_base_moved[[j, col]]).abs() < 0.05,
                        "equivariance broke at [{j},{col}]: {} vs {}",
                        y_moved[[j, col]], y_base_moved[[j, col]]
                    );
                }
            }
        }
    }
}
