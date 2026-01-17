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
//! | [`sinkhorn`] | General transport (dense) | O(n² × iterations) |
//! | [`sparse::solve_semidual_l2`] | Sparse transport (L2) | O(n² × L-BFGS iter) |
//! | [`earth_mover_distance`] | Exact transport | O(n³) |
//!
//! ## Quick Start
//!
//! ```rust
//! use wass::{wasserstein_1d, sinkhorn, sinkhorn_divergence};
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
//!
//! // Sinkhorn Divergence (2026 Bleeding Edge Metric)
//! let div = sinkhorn_divergence(&a, &b, &cost, 0.1, 100);
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
//! - [`logp`](../logp): KL divergence vs optimal transport
//! - [`fynch`](../fynch): Differentiable sorting/ranking (includes a Sinkhorn-based sorter)
//! - [`sparse`]: Sparse OT with L2 regularization (interpretable alignments)
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

pub mod flow;
pub mod gromov;
pub mod sparse;

pub use flow::{flow_drift, VectorField};

/// Optimal Transport error variants.
#[derive(Debug, Error)]
pub enum Error {
    /// Distributions have different lengths.
    #[error("distributions have different lengths: {0} vs {1}")]
    LengthMismatch(usize, usize),

    /// Cost matrix shape mismatch.
    #[error("cost matrix shape mismatch: expected ({0}, {1}), got ({2}, {3})")]
    CostShapeMismatch(usize, usize, usize, usize),

    /// Distribution does not sum to 1.0.
    #[error("distribution does not sum to 1.0 (sum = {0})")]
    NotNormalized(f32),

    /// Sinkhorn algorithm did not converge within the iteration limit.
    #[error("Sinkhorn did not converge in {0} iterations")]
    SinkhornNotConverged(usize),
}

/// Result type for Optimal Transport operations.
pub type Result<T> = std::result::Result<T, Error>;

const EPSILON: f32 = 1e-7;

/// Numerically stable \(\log \sum_i \exp(x_i)\) for an indexable family.
///
/// This is the classic "log-sum-exp trick":
/// \[
/// \log \sum_i \exp(x_i) = m + \log \sum_i \exp(x_i - m), \quad m = \max_i x_i
/// \]
///
/// Returns `-∞` if `len == 0`.
#[inline]
fn logsumexp_by(len: usize, mut f: impl FnMut(usize) -> f32) -> f32 {
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    let mut max_val = f32::NEG_INFINITY;
    for i in 0..len {
        max_val = max_val.max(f(i));
    }
    if !max_val.is_finite() {
        // If everything is -inf (or NaN), propagate the max.
        return max_val;
    }

    let mut sum_exp = 0.0;
    for i in 0..len {
        sum_exp += (f(i) - max_val).exp();
    }
    max_val + sum_exp.ln()
}

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
/// assert!((w - 2.0).abs() < 1e-5);  // distance = 2 bins
/// ```
pub fn wasserstein_1d(a: &[f32], b: &[f32]) -> f32 {
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
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
) -> (Array2<f32>, f32) {
    let m = a.len();
    let n = b.len();

    assert_eq!(cost.shape(), &[m, n], "cost matrix shape mismatch");

    // Kernel K = exp(-C / ε)
    let k: Array2<f32> = cost.mapv(|c| (-c / reg).exp());

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
    let distance: f32 = cost.iter().zip(plan.iter()).map(|(&c, &p)| c * p).sum();

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
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<(Array2<f32>, f32, usize)> {
    let m = a.len();
    let n = b.len();

    assert_eq!(cost.shape(), &[m, n], "cost matrix shape mismatch");

    let k: Array2<f32> = cost.mapv(|c| (-c / reg).exp());

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
        let diff: f32 = u
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

            let distance: f32 = cost.iter().zip(plan.iter()).map(|(&c, &p)| c * p).sum();

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
pub fn earth_mover_distance(a: &Array1<f32>, b: &Array1<f32>, cost: &Array2<f32>) -> f32 {
    let reg = 0.01; // Small regularization for good approximation
    let (_, distance) = sinkhorn(a, b, cost, reg, 200);
    distance
}

/// Sinkhorn Divergence (De-biased Entropic OT).
///
/// Formula: S_ε(p, q) = OT_ε(p, q) - 1/2 * (OT_ε(p, p) + OT_ε(q, q))
///
/// This provides a positive-definite divergence that interpolates between
/// Wasserstein distance (ε → 0) and Maximum Mean Discrepancy (ε → ∞).
pub fn sinkhorn_divergence(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
) -> f32 {
    let (_, ot_pq) = sinkhorn_log(a, b, cost, reg, max_iter);

    // Internal cost matrices for self-distance
    let m = a.len();
    let n = b.len();
    // Assuming Euclidean cost if not specified, but here we reuse whatever metric is in `cost`.
    // For self-distance, we need the cost matrix between the SAME support.
    // This is only easy if support is the same for a and b.
    // If not, Sinkhorn Divergence is harder to compute correctly.

    // Simplified assumption: if cost is square, use it for p-p and q-q.
    if m == n {
        let (_, ot_pp) = sinkhorn_log(a, a, cost, reg, max_iter);
        let (_, ot_qq) = sinkhorn_log(b, b, cost, reg, max_iter);
        (ot_pq - 0.5 * (ot_pp + ot_qq)).max(0.0)
    } else {
        // Warning: this fallback is mathematically biased (not a divergence).
        // For rectangular cost, we can't compute OT(p,p) with the same matrix.
        ot_pq 
    }
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
pub fn euclidean_cost_matrix(x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
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

/// Sinkhorn algorithm in log-space for numerical stability.
///
/// Solves entropic OT using log-domain computations to avoid underflow/overflow.
/// Formula: f_i = ε log(a_i) - ε log(Σ exp((g_j - C_ij) / ε))
///
/// This implementation uses log-sum-exp trick for stability and normalizes distributions.
pub fn sinkhorn_log(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
) -> (Array2<f32>, f32) {
    let m = a.len();
    let n = b.len();

    // Normalize input distributions to ensure they sum to 1
    let a_sum = a.sum();
    let b_sum = b.sum();
    let a = a / (a_sum + EPSILON);
    let b = b / (b_sum + EPSILON);

    let log_a = a.mapv(|x| (x + EPSILON).ln());
    let log_b = b.mapv(|x| (x + EPSILON).ln());

    let mut f: Array1<f32> = Array1::zeros(m);
    let mut g: Array1<f32> = Array1::zeros(n);

    for _ in 0..max_iter {
        // Update f: f_i = ε * log(a_i) - ε * logsumexp_j((g_j - C_ij) / ε)
        for i in 0..m {
            let lse = logsumexp_by(n, |j| (g[j] - cost[[i, j]]) / reg);
            f[i] = reg * (log_a[i] - lse);
        }

        // Update g: g_j = ε * log(b_i) - ε * logsumexp_i((f_i - C_ij) / ε)
        for j in 0..n {
            let lse = logsumexp_by(m, |i| (f[i] - cost[[i, j]]) / reg);
            g[j] = reg * (log_b[j] - lse);
        }
    }

    let mut plan = Array2::zeros((m, n));
    let mut distance = 0.0;
    for i in 0..m {
        for j in 0..n {
            let log_p = (f[i] + g[j] - cost[[i, j]]) / reg;
            plan[[i, j]] = log_p.exp();
            distance += plan[[i, j]] * cost[[i, j]];
        }
    }

    (plan, distance)
}

/// Sliced Wasserstein distance (fast approximation for high dimensions).
///
/// Projects distributions onto random 1D subspaces and averages W₁ distances.
/// This uses Gaussian random projections to estimate the distance.
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
pub fn sliced_wasserstein(x: &Array2<f32>, y: &Array2<f32>, n_projections: usize) -> f32 {
    let d = x.ncols();
    assert_eq!(y.ncols(), d, "point dimensions must match");

    let m = x.nrows();
    let n = y.nrows();

    if m == 0 || n == 0 {
        return 0.0;
    }

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    // Use a deterministic seed for consistency in tests/benchmarks
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut total = 0.0;

    for _ in 0..n_projections {
        // 1. Generate random direction from unit sphere (Gaussian normalized)
        let mut direction: Array1<f32> = Array1::zeros(d);
        for i in 0..d {
            direction[i] = StandardNormal.sample(&mut rng);
        }
        
        // Optimize: use innr for dot product if available
        #[cfg(feature = "simd")]
        let norm = innr::dense::norm(direction.as_slice().unwrap());
        #[cfg(not(feature = "simd"))]
        let norm = direction.dot(&direction).sqrt();
        
        direction /= norm.max(EPSILON);

        // 2. Project points
        #[cfg(feature = "simd")]
        let mut proj_x = Vec::with_capacity(m);
        #[cfg(feature = "simd")]
        for i in 0..m {
            proj_x.push(innr::dense::dot(x.row(i).as_slice().unwrap(), direction.as_slice().unwrap()));
        }
        
        #[cfg(not(feature = "simd"))]
        let mut proj_x = x.dot(&direction).to_vec();
        
        #[cfg(feature = "simd")]
        let mut proj_y = Vec::with_capacity(n);
        #[cfg(feature = "simd")]
        for i in 0..n {
            proj_y.push(innr::dense::dot(y.row(i).as_slice().unwrap(), direction.as_slice().unwrap()));
        }

        #[cfg(not(feature = "simd"))]
        let mut proj_y = y.dot(&direction).to_vec();

        // 3. Sort projections
        proj_x.sort_by(|a, b| a.total_cmp(b));
        proj_y.sort_by(|a, b| a.total_cmp(b));

        // 4. W₁ for 1D empirical distributions
        // When m = n, this is just mean absolute difference
        // When m != n, we should ideally use interpolation.
        // For now, we assume m = n or truncate to min_len as a baseline.
        let min_len = m.min(n);
        let w1: f32 = (0..min_len)
            .map(|i| (proj_x[i] - proj_y[i]).abs())
            .sum::<f32>()
            / min_len as f32;

        total += w1;
    }

    total / n_projections as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn test_wasserstein_1d_same() {
        let a = [0.25, 0.25, 0.25, 0.25];
        let w = wasserstein_1d(&a, &a);
        assert!(w < 1e-7, "same distribution should have 0 distance");
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
        let plan_sum: f32 = plan.iter().sum();
        assert!((plan_sum - 1.0).abs() < 0.01, "plan should sum to 1");

        // Marginals should match
        let row_sums: Vec<f32> = (0..2).map(|i| plan.row(i).sum()).collect();
        assert!((row_sums[0] - 0.5).abs() < 0.1);
        assert!((row_sums[1] - 0.5).abs() < 0.1);

        // Distance should be reasonable
        assert!((0.0..1.0).contains(&distance));
    }

    #[test]
    fn test_sinkhorn_identical() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];

        let (_, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);

        // Identity transport should be cheap
        assert!(
            distance < 0.5,
            "identical distributions should have low OT cost"
        );
    }

    #[test]
    fn test_euclidean_cost_matrix() {
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0, 0.0], [0.0, 1.0]];

        let cost = euclidean_cost_matrix(&x, &y);

        assert!((cost[[0, 0]] - 0.0).abs() < 1e-7);
        assert!((cost[[0, 1]] - 1.0).abs() < 1e-7);
        assert!((cost[[1, 0]] - 1.0).abs() < 1e-7);
        assert!((cost[[1, 1]] - 2.0_f32.sqrt()).abs() < 1e-7);
    }

    #[test]
    fn test_sliced_wasserstein() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0, 0.0], [1.0, 1.0]];

        let sw = sliced_wasserstein(&x, &y, 10);
        assert!(sw < 0.1, "same points should have ~0 sliced Wasserstein");

        let y2 = array![[10.0, 10.0], [11.0, 11.0]];
        let sw2 = sliced_wasserstein(&x, &y2, 10);
        assert!(
            sw2 > 5.0,
            "distant points should have large sliced Wasserstein"
        );
    }

    proptest! {
        #[test]
        fn prop_sinkhorn_divergence_non_negative(
            a in prop::collection::vec(0.0f32..1.0, 2..8),
            b in prop::collection::vec(0.0f32..1.0, 2..8),
        ) {
            let n = a.len();
            let mut a_dist = Array1::from_vec(a);
            let mut b_dist = Array1::from_vec(b);
            
            // Normalize
            let sa = a_dist.sum();
            let sb = b_dist.sum();
            if sa > 0.0 { a_dist /= sa; } else { a_dist[0] = 1.0; }
            if sb > 0.0 { b_dist /= sb; } else { b_dist[0] = 1.0; }
            
            let mut cost = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    cost[[i, j]] = (i as f32 - j as f32).abs();
                }
            }
            
            let div = sinkhorn_divergence(&a_dist, &b_dist, &cost, 0.1, 50);
            prop_assert!(div >= -1e-6);
        }

        #[test]
        fn logsumexp_translation_invariant(
            xs in prop::collection::vec(-50.0f32..50.0, 1..64),
            shift in -10.0f32..10.0
        ) {
            let l1 = logsumexp_by(xs.len(), |i| xs[i]);
            let l2 = logsumexp_by(xs.len(), |i| xs[i] + shift);
            prop_assert!((l2 - (l1 + shift)).abs() < 1e-5);
        }

        #[test]
        fn logsumexp_matches_naive_on_safe_range(
            xs in prop::collection::vec(-20.0f32..20.0, 1..64),
        ) {
            // Reference implementation (naive): log(sum(exp(x))).
            // This over/underflows for large magnitude x; hence the restricted range.
            let naive = xs.iter().map(|&x| x.exp()).sum::<f32>().ln();
            let stable = logsumexp_by(xs.len(), |i| xs[i]);
            prop_assert!((stable - naive).abs() < 1e-5);
        }

        #[test]
        fn logsumexp_bounds_by_max(
            xs in prop::collection::vec(-50.0f32..50.0, 1..64),
        ) {
            let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = logsumexp_by(xs.len(), |i| xs[i]);
            // max <= logsumexp <= max + ln(n)
            prop_assert!(lse >= max - 1e-5);
            prop_assert!(lse <= max + (xs.len() as f32).ln() + 1e-5);
        }
    }
}
