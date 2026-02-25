//! Entropic Gromov-Wasserstein Optimal Transport.
//!
//! Matches two metric spaces (X, C1) and (Y, C2) by finding a transport plan P
//! that minimizes the distortion between their structures.
//!
//! Metric: $GW(C_1, C_2) = \min_P \sum_{ijkl} |C_1(i,k) - C_2(j,l)|^2 P_{ij} P_{kl} - \epsilon H(P)$

use crate::{sinkhorn_log, Error, Result};
use ndarray::{Array1, Array2};

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

        // `sinkhorn_log` currently operates on f32; keep the public GW API in f64, but
        // do the Sinkhorn substep in f32.
        let p32 = p.mapv(|x| x as f32);
        let q32 = q.mapv(|x| x as f32);
        let g32 = g.mapv(|x| x as f32);
        let (new_plan32, _dist) = sinkhorn_log(&p32, &q32, &g32, epsilon as f32, sinkhorn_iter);
        plan = new_plan32.mapv(|x| x as f64);
        gw_dist = g.iter().zip(plan.iter()).map(|(gi, pi)| gi * pi).sum();
    }

    Ok((plan, gw_dist))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gw_identical_spaces_low_cost() {
        // Two identical metric spaces should have low GW cost
        let c = array![[0.0, 1.0], [1.0, 0.0]];
        let p = array![0.5, 0.5];
        let (plan, dist) = gromov_wasserstein(&c, &c, &p, &p, 0.1, 10, 50).unwrap();
        // GW with entropic regularization doesn't reach exactly 0;
        // the plan should be close to identity-like (diagonal dominant)
        assert!(dist < 1.0, "identical spaces should have low dist: dist={}", dist);
        let sum: f64 = plan.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "plan should sum to ~1: sum={}", sum);
    }

    #[test]
    fn gw_plan_has_correct_shape() {
        let c1 = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
        let c2 = array![[0.0, 3.0], [3.0, 0.0]];
        let p = array![0.33, 0.34, 0.33];
        let q = array![0.5, 0.5];
        let (plan, _) = gromov_wasserstein(&c1, &c2, &p, &q, 0.1, 5, 50).unwrap();
        assert_eq!(plan.shape(), &[3, 2]);
    }

    #[test]
    fn gw_rejects_non_square_cost() {
        let c1 = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]; // 2x3, not square
        let c2 = array![[0.0, 1.0], [1.0, 0.0]];
        let p = array![0.5, 0.5];
        let q = array![0.5, 0.5];
        assert!(gromov_wasserstein(&c1, &c2, &p, &q, 0.1, 5, 50).is_err());
    }

    #[test]
    fn gw_rejects_length_mismatch() {
        let c = array![[0.0, 1.0], [1.0, 0.0]];
        let p = array![0.5, 0.5];
        let q_bad = array![0.33, 0.34, 0.33]; // wrong length for c
        assert!(gromov_wasserstein(&c, &c, &p, &q_bad, 0.1, 5, 50).is_err());
    }
}
