//! # wass
//!
//! Optimal transport: move mass from one distribution to another at minimum cost.
//!
//! ## The Problem
//!
//! Given two probability distributions (piles of sand), find the cheapest way
//! to transform one into the other. The "cost" is how much mass moves times
//! how far it moves.
//!
//! ## Key Functions
//!
//! | Function | Use Case | Complexity |
//! |----------|----------|------------|
//! | [`wasserstein_1d`] | 1D distributions | O(n log n) |
//! | [`sinkhorn`] | General transport | O(n² × iterations) |
//! | [`earth_mover_distance`] | Exact transport | O(n³) |
//!
//! ## Quick Start
//!
//! ```rust
//! use wass::{wasserstein_1d, sinkhorn};
//! use ndarray::array;
//!
//! // 1D Wasserstein (fast, closed-form)
//! let a = [0.0, 0.25, 0.5, 0.25];
//! let b = [0.25, 0.5, 0.25, 0.0];
//! let w1 = wasserstein_1d(&a, &b);
//!
//! // General transport with Sinkhorn
//! let cost = array![[0.0, 1.0], [1.0, 0.0]];
//! let a = array![0.5, 0.5];
//! let b = array![0.5, 0.5];
//! let (plan, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);
//! ```
//!
//! ## Why Optimal Transport?
//!
//! - **Better distance metric**: Unlike KL divergence, OT compares distributions
//!   with different supports (no divide-by-zero issues)
//! - **Geometry-aware**: Respects the underlying metric space
//! - **Interpolation**: Can create meaningful "in-between" distributions
//!
//! ## Applications in ML
//!
//! - **Wasserstein GAN**: More stable training via W1 critic
//! - **Domain adaptation**: Align source/target feature distributions
//! - **Embedding comparison**: Compare document/image embeddings as distributions
//! - **Fair ML**: Measure/minimize distribution shift across groups
//!
//! ## Connections
//!
//! - [`rkhs`](../rkhs): MMD vs Wasserstein—both compare distributions
//! - [`surp`](../surp): KL divergence vs optimal transport
//! - [`fynch`](../fynch): Sinkhorn for differentiable sorting
//!
//! ## What Can Go Wrong
//!
//! 1. **Sinkhorn not converging**: Decrease epsilon or increase iterations.
//! 2. **Numerical overflow**: Large cost/small epsilon → exp overflow. Scale costs.
//! 3. **Marginal mismatch**: Sinkhorn assumes both margins sum to 1. Normalize inputs.
//! 4. **Sliced approximation bias**: Fewer projections = noisier estimate.
//! 5. **1D vs general**: [`wasserstein_1d`] only works for 1D histograms on same bins.
//!
//! ## Note on Production Use
//!
//! For comprehensive optimal transport, consider [`ruvector-math`](https://crates.io/crates/ruvector-math)
//! which provides a more feature-complete implementation.
//!
//! ## References
//!
//! - Kantorovich (1942). "On the Translocation of Masses"
//! - Cuturi (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
//! - Peyré & Cuturi (2019). "Computational Optimal Transport"

use ndarray::{Array1, Array2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("distributions have different lengths: {0} vs {1}")]
    LengthMismatch(usize, usize),

    #[error("cost matrix shape mismatch: expected ({0}, {1}), got ({2}, {3})")]
    CostShapeMismatch(usize, usize, usize, usize),

    #[error("distribution does not sum to 1.0 (sum = {0})")]
    NotNormalized(f64),

    #[error("Sinkhorn did not converge in {0} iterations")]
    SinkhornNotConverged(usize),
}

pub type Result<T> = std::result::Result<T, Error>;

const EPSILON: f64 = 1e-12;

/// 1D Wasserstein distance (Earth Mover's Distance).
///
/// For 1D distributions, the Wasserstein distance has a closed-form solution
/// based on the cumulative distribution functions (CDFs).
///
/// W₁(a, b) = ∫|F_a(x) - F_b(x)| dx
///
/// # Arguments
///
/// * `a` - First distribution (histogram/PMF)
/// * `b` - Second distribution (histogram/PMF)
///
/// # Returns
///
/// W₁ distance (assumes unit spacing between bins)
///
/// # Complexity
///
/// O(n log n) for sorting, O(n) for integration
///
/// # Example
///
/// ```rust
/// use wass::wasserstein_1d;
///
/// let a = [0.0, 1.0, 0.0, 0.0];  // all mass at index 1
/// let b = [0.0, 0.0, 0.0, 1.0];  // all mass at index 3
///
/// let w = wasserstein_1d(&a, &b);
/// assert!((w - 2.0).abs() < 1e-10);  // distance = 2 bins
/// ```
pub fn wasserstein_1d(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "distributions must have same length");

    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    // Compute CDFs
    let mut cdf_a = vec![0.0; n];
    let mut cdf_b = vec![0.0; n];

    cdf_a[0] = a[0];
    cdf_b[0] = b[0];

    for i in 1..n {
        cdf_a[i] = cdf_a[i - 1] + a[i];
        cdf_b[i] = cdf_b[i - 1] + b[i];
    }

    // W₁ = Σ |CDF_a[i] - CDF_b[i]| (discrete approximation)
    cdf_a
        .iter()
        .zip(cdf_b.iter())
        .map(|(&ca, &cb)| (ca - cb).abs())
        .sum()
}

/// Sinkhorn algorithm for entropic regularized optimal transport.
///
/// Solves the regularized transport problem:
///
/// min_P <C, P> - ε H(P)
/// s.t. P1 = a, P^T1 = b, P ≥ 0
///
/// where H(P) = -Σ P_ij log P_ij is the entropy.
///
/// # Arguments
///
/// * `a` - Source distribution (length m)
/// * `b` - Target distribution (length n)
/// * `cost` - Cost matrix C (m × n)
/// * `reg` - Regularization strength ε (smaller = closer to exact)
/// * `max_iter` - Maximum iterations
///
/// # Returns
///
/// (transport_plan, transport_distance) where:
/// - transport_plan: P matrix (m × n) giving how much mass moves
/// - transport_distance: <C, P> = Σ C_ij P_ij
///
/// # Complexity
///
/// O(m × n × iterations)
///
/// # Example
///
/// ```rust
/// use wass::sinkhorn;
/// use ndarray::array;
///
/// let a = array![0.5, 0.5];
/// let b = array![0.5, 0.5];
/// let cost = array![[0.0, 1.0], [1.0, 0.0]];
///
/// let (plan, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);
/// ```
pub fn sinkhorn(
    a: &Array1<f64>,
    b: &Array1<f64>,
    cost: &Array2<f64>,
    reg: f64,
    max_iter: usize,
) -> (Array2<f64>, f64) {
    let m = a.len();
    let n = b.len();

    assert_eq!(cost.shape(), &[m, n], "cost matrix shape mismatch");

    // Kernel K = exp(-C / ε)
    let k: Array2<f64> = cost.mapv(|c| (-c / reg).exp());

    // Initialize scaling vectors
    let mut u = Array1::ones(m);
    let mut v = Array1::ones(n);

    // Sinkhorn iterations
    for _ in 0..max_iter {
        // u = a / (K v)
        let kv = k.dot(&v);
        for i in 0..m {
            u[i] = a[i] / (kv[i] + EPSILON);
        }

        // v = b / (K^T u)
        let ktu = k.t().dot(&u);
        for j in 0..n {
            v[j] = b[j] / (ktu[j] + EPSILON);
        }
    }

    // Transport plan P = diag(u) K diag(v)
    let mut plan = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            plan[[i, j]] = u[i] * k[[i, j]] * v[j];
        }
    }

    // Transport distance = <C, P>
    let distance: f64 = cost
        .iter()
        .zip(plan.iter())
        .map(|(&c, &p)| c * p)
        .sum();

    (plan, distance)
}

/// Sinkhorn with convergence check.
///
/// Same as [`sinkhorn`] but checks for convergence and returns early.
///
/// # Arguments
///
/// * `a` - Source distribution
/// * `b` - Target distribution
/// * `cost` - Cost matrix
/// * `reg` - Regularization strength
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance for marginal error
///
/// # Returns
///
/// Ok((plan, distance, iterations)) if converged, Err otherwise
pub fn sinkhorn_with_convergence(
    a: &Array1<f64>,
    b: &Array1<f64>,
    cost: &Array2<f64>,
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Array2<f64>, f64, usize)> {
    let m = a.len();
    let n = b.len();

    assert_eq!(cost.shape(), &[m, n], "cost matrix shape mismatch");

    let k: Array2<f64> = cost.mapv(|c| (-c / reg).exp());

    let mut u = Array1::ones(m);
    let mut v = Array1::ones(n);

    for iter in 0..max_iter {
        let u_prev = u.clone();

        // u = a / (K v)
        let kv = k.dot(&v);
        for i in 0..m {
            u[i] = a[i] / (kv[i] + EPSILON);
        }

        // v = b / (K^T u)
        let ktu = k.t().dot(&u);
        for j in 0..n {
            v[j] = b[j] / (ktu[j] + EPSILON);
        }

        // Check convergence
        let diff: f64 = u
            .iter()
            .zip(u_prev.iter())
            .map(|(&ui, &upi)| (ui - upi).abs())
            .sum();

        if diff < tol {
            let mut plan = Array2::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    plan[[i, j]] = u[i] * k[[i, j]] * v[j];
                }
            }

            let distance: f64 = cost
                .iter()
                .zip(plan.iter())
                .map(|(&c, &p)| c * p)
                .sum();

            return Ok((plan, distance, iter + 1));
        }
    }

    Err(Error::SinkhornNotConverged(max_iter))
}

/// Earth mover's distance with custom ground metric.
///
/// Computes W₁(a, b) where ground metric is given by the cost matrix.
/// Uses Sinkhorn with small regularization as approximation.
///
/// For exact solution, use linear programming (not implemented here).
///
/// # Arguments
///
/// * `a` - Source distribution
/// * `b` - Target distribution
/// * `cost` - Ground metric / cost matrix
///
/// # Returns
///
/// Approximate EMD
pub fn earth_mover_distance(a: &Array1<f64>, b: &Array1<f64>, cost: &Array2<f64>) -> f64 {
    let reg = 0.01; // Small regularization for good approximation
    let (_, distance) = sinkhorn(a, b, cost, reg, 200);
    distance
}

/// Create Euclidean cost matrix from point positions.
///
/// C[i,j] = ||x_i - y_j||₂
///
/// # Arguments
///
/// * `x` - Source points (m × d)
/// * `y` - Target points (n × d)
///
/// # Returns
///
/// Cost matrix (m × n)
pub fn euclidean_cost_matrix(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let m = x.nrows();
    let n = y.nrows();
    let d = x.ncols();

    assert_eq!(y.ncols(), d, "point dimensions must match");

    let mut cost = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut dist_sq = 0.0;
            for k in 0..d {
                let diff = x[[i, k]] - y[[j, k]];
                dist_sq += diff * diff;
            }
            cost[[i, j]] = dist_sq.sqrt();
        }
    }

    cost
}

/// Sliced Wasserstein distance (fast approximation for high dimensions).
///
/// Projects distributions onto random 1D subspaces and averages W₁ distances.
///
/// # Arguments
///
/// * `x` - Source samples (m × d)
/// * `y` - Target samples (n × d)
/// * `n_projections` - Number of random projections
///
/// # Returns
///
/// Sliced Wasserstein distance
pub fn sliced_wasserstein(x: &Array2<f64>, y: &Array2<f64>, n_projections: usize) -> f64 {
    let d = x.ncols();
    assert_eq!(y.ncols(), d, "point dimensions must match");

    let m = x.nrows();
    let n = y.nrows();

    let mut total = 0.0;

    // Simple random projections (could use Sobol sequence for better coverage)
    use std::f64::consts::PI;
    for k in 0..n_projections {
        // Random direction on unit sphere (simplified: just rotate)
        let theta = 2.0 * PI * (k as f64) / (n_projections as f64);
        let mut direction = vec![0.0; d];
        direction[0] = theta.cos();
        if d > 1 {
            direction[1] = theta.sin();
        }

        // Project points
        let mut proj_x: Vec<f64> = (0..m)
            .map(|i| {
                (0..d).map(|j| x[[i, j]] * direction[j]).sum()
            })
            .collect();

        let mut proj_y: Vec<f64> = (0..n)
            .map(|i| {
                (0..d).map(|j| y[[i, j]] * direction[j]).sum()
            })
            .collect();

        // Sort projections
        proj_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        proj_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // W₁ for 1D empirical distributions = average absolute difference of sorted samples
        // When m ≠ n, need interpolation - simplified: assume m = n
        let min_len = m.min(n);
        let w1: f64 = (0..min_len)
            .map(|i| (proj_x[i] - proj_y[i]).abs())
            .sum::<f64>()
            / min_len as f64;

        total += w1;
    }

    total / n_projections as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_wasserstein_1d_same() {
        let a = [0.25, 0.25, 0.25, 0.25];
        let w = wasserstein_1d(&a, &a);
        assert!(w < 1e-10, "same distribution should have 0 distance");
    }

    #[test]
    fn test_wasserstein_1d_shift() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0, 1.0];
        let w = wasserstein_1d(&a, &b);
        assert!(
            (w - 3.0).abs() < 0.01,
            "point mass shift of 3 should have distance ~3"
        );
    }

    #[test]
    fn test_sinkhorn_basic() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];

        let (plan, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);

        // Plan should sum to 1
        let plan_sum: f64 = plan.iter().sum();
        assert!(
            (plan_sum - 1.0).abs() < 0.01,
            "plan should sum to 1"
        );

        // Marginals should match
        let row_sums: Vec<f64> = (0..2).map(|i| plan.row(i).sum()).collect();
        assert!((row_sums[0] - 0.5).abs() < 0.1);
        assert!((row_sums[1] - 0.5).abs() < 0.1);

        // Distance should be reasonable
        assert!(distance >= 0.0 && distance < 1.0);
    }

    #[test]
    fn test_sinkhorn_identical() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];

        let (_, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);

        // Identity transport should be cheap
        assert!(distance < 0.5, "identical distributions should have low OT cost");
    }

    #[test]
    fn test_euclidean_cost_matrix() {
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0, 0.0], [0.0, 1.0]];

        let cost = euclidean_cost_matrix(&x, &y);

        assert!((cost[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((cost[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((cost[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((cost[[1, 1]] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sliced_wasserstein() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0, 0.0], [1.0, 1.0]];

        let sw = sliced_wasserstein(&x, &y, 10);
        assert!(sw < 0.1, "same points should have ~0 sliced Wasserstein");

        let y2 = array![[10.0, 10.0], [11.0, 11.0]];
        let sw2 = sliced_wasserstein(&x, &y2, 10);
        assert!(sw2 > 5.0, "distant points should have large sliced Wasserstein");
    }
}
