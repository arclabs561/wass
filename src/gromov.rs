//! Entropic Gromov-Wasserstein optimal transport.
//!
//! Gromov-Wasserstein (Memoli, 2011) matches two **metric spaces** \((X, C_1)\) and
//! \((Y, C_2)\) without requiring them to share an ambient space. It finds a transport
//! plan \(P\) that minimizes the quadratic distortion:
//!
//! \[
//! \mathrm{GW}(C_1, C_2) = \min_{P \in U(p,q)} \sum_{i,j,k,l} |C_1(i,k) - C_2(j,l)|^2 P_{ij} P_{kl} + \varepsilon H(P)
//! \]
//!
//! **Intuition**: GW measures how much the *internal structure* of two spaces differs.
//! If two graphs have the same topology but different node labels, GW will still find
//! the correct alignment -- it only cares about pairwise distances, not coordinates.
//!
//! **Applications**: shape matching, cross-lingual word embedding alignment,
//! graph comparison, protein structure alignment.
//!
//! **Algorithm**: projected gradient descent -- at each outer iteration, linearize the
//! quadratic objective to get a linear cost matrix \(G\), then solve the linearized
//! problem with Sinkhorn. This is a.k.a. the "Frank-Wolfe" or "conditional gradient" scheme.
//!
//! ## References
//!
//! - Memoli (2011). "Gromov-Wasserstein Distances and Metric Measure Spaces"
//! - Peyre, Cuturi, Solomon (2016). "Gromov-Wasserstein Averaging"
//! - Peyre & Cuturi (2019). "Computational Optimal Transport", Ch. 10
//! - Rioux, Goldfeld, Kato (2023). "Entropic Gromov-Wasserstein Distances:
//!   Stability and Algorithms" -- convergence rates for the Frank-Wolfe/Sinkhorn scheme
//! - Zhang et al. (2024). "Fast Gradient Computation for Gromov-Wasserstein Distance"
//!   -- accelerated gradient for the C1*P*C2^T inner loop (potential future optimization)
//! - Beier et al. (2021). "On a Linear Gromov-Wasserstein Distance" -- O(n^2)
//!   linear approximation as a cheaper alternative for large problems

use crate::{sinkhorn_log, Error, Result};
use ndarray::{Array1, Array2};

/// Compute the entropic Gromov-Wasserstein discrepancy and transport plan.
///
/// Given intra-space cost matrices \(C_1 \in \mathbb{R}^{m \times m}\) and
/// \(C_2 \in \mathbb{R}^{n \times n}\), marginals \(p \in \Delta^m\) and
/// \(q \in \Delta^n\), and entropic regularization \(\varepsilon\), returns
/// the GW transport plan \(P^*\) and the associated distortion cost.
///
/// # Arguments
///
/// * `c1` - Intra-space distance matrix for space \(X\) (\(m \times m\), symmetric)
/// * `c2` - Intra-space distance matrix for space \(Y\) (\(n \times n\), symmetric)
/// * `p` - Source marginal (length \(m\), sums to 1)
/// * `q` - Target marginal (length \(n\), sums to 1)
/// * `epsilon` - Entropic regularization \(\varepsilon > 0\)
/// * `max_iter` - Outer (Frank-Wolfe) iterations
/// * `sinkhorn_iter` - Inner Sinkhorn iterations per linearization
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
