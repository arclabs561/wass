//! Entropic Gromov-Wasserstein Optimal Transport.
//!
//! Matches two metric spaces (X, C1) and (Y, C2) by finding a transport plan P
//! that minimizes the distortion between their structures.
//!
//! Metric: $GW(C_1, C_2) = \min_P \sum_{ijkl} |C_1(i,k) - C_2(j,l)|^2 P_{ij} P_{kl} - \epsilon H(P)$

use ndarray::{Array1, Array2};
use crate::{sinkhorn_log, Error, Result};

/// Compute the Entropic Gromov-Wasserstein discrepancy and transport plan.
pub fn gromov_wasserstein(
    c1: &Array2<f64>,
    c2: &Array2<f64>,
    p: &Array1<f64>,
    q: &Array1<f64>,
    epsilon: f64,
    max_iter: usize,
    sinkhorn_iter: usize,
) -> Result<(Array2<f64>, f64)> {
    let m = c1.nrows();
    let n = c2.nrows();

    if c1.ncols() != m || c2.ncols() != n {
        return Err(Error::CostShapeMismatch(m, m, c1.nrows(), c1.ncols()));
    }
    if p.len() != m || q.len() != n {
        return Err(Error::LengthMismatch(m, p.len()));
    }

    // Gradient G(P) = const_1 + const_2 - 2 * C1 * P * C2^T
    let c1_sq = c1.mapv(|x| x.powi(2));
    let c2_sq = c2.mapv(|x| x.powi(2));

    let mu_c1_sq = c1_sq.dot(p);
    let nu_c2_sq = c2_sq.t().dot(q);

    let mut plan = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            plan[[i, j]] = p[i] * q[j];
        }
    }

    let mut gw_dist = 0.0;

    for _iter in 0..max_iter {
        let c1_p = c1.dot(&plan);
        let c1_p_c2t = c1_p.dot(&c2.t());

        let mut g = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                g[[i, j]] = mu_c1_sq[i] + nu_c2_sq[j] - 2.0 * c1_p_c2t[[i, j]];
            }
        }

        let (new_plan, _dist) = sinkhorn_log(p, q, &g, epsilon, sinkhorn_iter);
        plan = new_plan;
        gw_dist = g.iter().zip(plan.iter()).map(|(gi, pi)| gi * pi).sum();
    }

    Ok((plan, gw_dist))
}
