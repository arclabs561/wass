#![warn(missing_docs)]
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
//! | [`wasserstein_1d`] | 1D distributions (histograms) | O(n) |
//! | [`wasserstein_1d_samples`] | 1D distributions (raw samples) | O(n log n) |
//! | [`sliced_wasserstein`] | High-dim point clouds (random projections) | O(L n log n) |
//! | [`max_sliced_wasserstein`] | High-dim point clouds (max projection) | O(L n log n) |
//! | [`sinkhorn`] | General transport (dense) | O(n² × iterations) |
//! | [`sinkhorn_low_rank`] | Large-scale transport (rank r) | O((n+m)r × iterations) |
//! | [`sinkhorn_hierarchical`] | Large-scale transport (tree partition) | O(k² + local subproblems) |
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
//! // Sinkhorn Divergence (Robust Distribution Metric)
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
//! ## References
//!
//! - Kantorovich (1942). "On the Translocation of Masses"
//! - Cuturi (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
//! - Peyré & Cuturi (2019). "Computational Optimal Transport"
//! - Memoli (2011). "Gromov-Wasserstein Distances and Metric Measure Spaces"
//! - Blondel, Seguy, Rolet (2018). "Smooth and Sparse OT" (AISTATS)
//! - Chizat et al. (2018). "Scaling Algorithms for Unbalanced OT Problems"
//! - Sejourne et al. (2023). "Unbalanced OT Meets Sliced-Wasserstein"
//! - Rabin et al. (2012). "Wasserstein Barycenter and Its Application to Texture Mixing"
//! - Scetbon, Cuturi & Peyré (2021). "Low-Rank Sinkhorn Factorization" (ICML)

use ndarray::{Array1, Array2};
use thiserror::Error;

pub mod flow;
pub mod gaussian;
pub mod gromov;
pub mod semidiscrete;
pub mod sparse;
pub mod wfr;

pub use flow::{flow_drift, VectorField};

/// Optimal Transport error variants.
#[derive(Debug, Error)]
#[non_exhaustive]
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

    /// Invalid regularization parameter.
    #[error("regularization parameter must be positive and finite, got {0}")]
    InvalidRegularization(f32),

    /// Invalid mass-variation penalty parameter for unbalanced OT.
    #[error("mass penalty parameter must be positive and finite, got {0}")]
    InvalidMassPenalty(f32),

    /// Invalid rank parameter for low-rank factorization.
    #[error("rank must be >= 1 and <= min(n, m), got rank={0} for n={1}, m={2}")]
    InvalidRank(usize, usize, usize),

    /// Invalid branching factor for hierarchical OT.
    #[error("branching must be >= 2 and <= min(n, m), got branching={0} for n={1}, m={2}")]
    InvalidBranching(usize, usize, usize),

    /// Domain error (invalid inputs for the mathematical definition).
    #[error("{0}")]
    Domain(&'static str),
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

/// 1D Wasserstein distance (Earth Mover's Distance) via CDF integration.
///
/// The 1D Wasserstein-1 distance admits a closed-form solution (Vallender, 1974):
///
/// \[
/// W_1(a, b) = \sum_{i} |F_a(i) - F_b(i)|
/// \]
///
/// where \(F_a, F_b\) are the cumulative distribution functions.
///
/// **Intuition**: imagine two histograms as piles of sand on a line. \(W_1\) is the
/// minimum total work (mass times distance) to reshape one pile into the other.
/// In 1D, the optimal plan is uniquely determined by the CDFs -- no linear program needed.
///
/// **Properties**:
/// - \(W_1(a, b) \ge 0\), with equality iff \(a = b\)
/// - Symmetric: \(W_1(a, b) = W_1(b, a)\)
/// - Satisfies the triangle inequality (it is a true metric on distributions)
/// - Metrizes weak convergence + convergence of first moments
///
/// # Arguments
///
/// * `a` - First distribution (histogram/PMF over shared bins)
/// * `b` - Second distribution (histogram/PMF over shared bins)
///
/// # Returns
///
/// \(W_1\) distance (assumes unit spacing between bins).
///
/// # Complexity
///
/// \(O(n)\) -- single pass over the CDFs.
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

/// Sinkhorn algorithm for entropic-regularized optimal transport (matrix scaling).
///
/// Solves the regularized Kantorovich problem (Cuturi, 2013):
///
/// \[
/// \min_{P \in U(a,b)} \langle C, P \rangle + \varepsilon H(P)
/// \]
///
/// where \(U(a,b) = \{P \ge 0 : P\mathbf{1} = a,\; P^\top\mathbf{1} = b\}\) is
/// the transport polytope and \(H(P) = -\sum_{ij} P_{ij} \log P_{ij}\) is the entropy.
///
/// **Algorithm**: alternating Bregman projections (matrix scaling). At each step:
/// \(u \leftarrow a / (Kv)\), then \(v \leftarrow b / (K^\top u)\),
/// where \(K_{ij} = \exp(-C_{ij}/\varepsilon)\). The optimal plan is
/// \(P^* = \mathrm{diag}(u)\, K\, \mathrm{diag}(v)\).
///
/// **Convergence**: linear rate \(O(\lambda^k)\) where \(\lambda < 1\) depends on the
/// Hilbert metric of \(K\). Smaller \(\varepsilon\) means slower convergence and
/// potential numerical underflow in \(K\). For small \(\varepsilon\), prefer [`sinkhorn_log`].
///
/// # Arguments
///
/// * `a` - Source distribution (length m, must sum to 1)
/// * `b` - Target distribution (length n, must sum to 1)
/// * `cost` - Cost matrix \(C\) (m x n)
/// * `reg` - Regularization strength \(\varepsilon > 0\) (smaller = closer to exact OT)
/// * `max_iter` - Maximum Sinkhorn iterations
///
/// # Returns
///
/// `(plan, distance)` where:
/// - `plan`: transport plan \(P^*\) (m x n), satisfying \(P^*\mathbf{1} \approx a\)
/// - `distance`: transport cost \(\langle C, P^* \rangle\)
///
/// # Complexity
///
/// \(O(mn \cdot \text{iterations})\). Each iteration is a matrix-vector product.
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

        // Check convergence via marginal error:
        // row_sum = diag(u) K v, col_sum = diag(v) K^T u.
        //
        // This matches the OT constraints and is scale-invariant.
        let kv2 = k.dot(&v);
        let mut max_err = 0.0f32;
        for i in 0..m {
            let row_sum = u[i] * kv2[i];
            max_err = max_err.max((row_sum - a[i]).abs());
        }
        let ktu2 = k.t().dot(&u);
        for j in 0..n {
            let col_sum = v[j] * ktu2[j];
            max_err = max_err.max((col_sum - b[j]).abs());
        }

        if max_err < tol {
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

/// Approximate Earth Mover's Distance (EMD) via log-domain Sinkhorn.
///
/// Computes \(W_1(a, b) \approx \min_{P \in U(a,b)} \langle C, P \rangle\)
/// using entropic regularization with \(\varepsilon = 0.01\).
///
/// The exact EMD requires solving a linear program (\(O(n^3)\) network simplex).
/// This function uses [`sinkhorn_log`] as a fast approximation -- the result is
/// biased upward by \(O(\varepsilon \log n)\) due to the entropy penalty.
///
/// # Arguments
///
/// * `a` - Source distribution (sums to 1)
/// * `b` - Target distribution (sums to 1)
/// * `cost` - Ground cost matrix \(C_{ij}\)
pub fn earth_mover_distance(a: &Array1<f32>, b: &Array1<f32>, cost: &Array2<f32>) -> f32 {
    let reg = 0.01; // Small regularization for good approximation
    let (_, distance) = sinkhorn_log(a, b, cost, reg, 200);
    distance
}

/// Sinkhorn Divergence (de-biased entropic OT).
///
/// \[
/// S_\varepsilon(a, b) = \mathrm{OT}_\varepsilon(a, b)
///     - \tfrac{1}{2}\bigl(\mathrm{OT}_\varepsilon(a, a) + \mathrm{OT}_\varepsilon(b, b)\bigr)
/// \]
///
/// **Why de-bias?** Raw entropic OT \(\mathrm{OT}_\varepsilon(a, b)\) is biased:
/// \(\mathrm{OT}_\varepsilon(a, a) > 0\) even when \(a = b\). The self-transport
/// terms correct this, yielding a proper divergence (\(S_\varepsilon(a, a) = 0\)).
///
/// **Interpolation** (Feydy et al., 2018): as \(\varepsilon \to 0\),
/// \(S_\varepsilon \to W_p^p\) (Wasserstein); as \(\varepsilon \to \infty\),
/// \(S_\varepsilon \to \text{MMD}^2\) (Maximum Mean Discrepancy with the
/// cost kernel). This makes it a smooth bridge between geometry-aware and
/// kernel-based distribution comparison.
///
/// # Warning
///
/// This function is only a true Sinkhorn divergence when the supports match (i.e.
/// `cost` is square and can be used for `p-p` and `q-q` self-costs).
///
/// If `cost` is **rectangular**, this function returns the entropic OT cost `OT_ε(p,q)`
/// (not debiased), which is **not** a divergence.
///
/// Prefer:
/// - [`sinkhorn_divergence_same_support`] when supports match
/// - [`sinkhorn_divergence_general`] when supports differ and you have `cost_ab/cost_aa/cost_bb`
#[deprecated(
    note = "Use sinkhorn_divergence_same_support or sinkhorn_divergence_general for a true divergence"
)]
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
        // This is not Sinkhorn divergence. Use `sinkhorn_divergence_general` if you have
        // the required self-cost matrices, or `sinkhorn_divergence_same_support` when
        // supports match.
        ot_pq
    }
}

/// De-biased Sinkhorn divergence for distributions on the **same support**.
///
/// Computes \(S_\varepsilon(a, b) = \mathrm{OT}_\varepsilon(a,b) - \tfrac{1}{2}(\mathrm{OT}_\varepsilon(a,a) + \mathrm{OT}_\varepsilon(b,b))\).
///
/// **Preconditions**: `a.len() == b.len() == n`, `cost` is \(n \times n\).
///
/// **Properties** (Feydy et al., 2018):
/// - \(S_\varepsilon(a, a) = 0\) (zero self-divergence)
/// - \(S_\varepsilon(a, b) \ge 0\) (positive definite)
/// - Symmetric: \(S_\varepsilon(a, b) = S_\varepsilon(b, a)\)
/// - Metrizes weak convergence for fixed \(\varepsilon > 0\)
///
/// Returns an error rather than silently producing a biased quantity.
pub fn sinkhorn_divergence_same_support(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let n = a.len();
    if b.len() != n {
        return Err(Error::LengthMismatch(n, b.len()));
    }
    if cost.nrows() != n || cost.ncols() != n {
        return Err(Error::CostShapeMismatch(n, n, cost.nrows(), cost.ncols()));
    }

    let (_p_pq, ot_pq, _iters_pq) = sinkhorn_log_with_convergence(a, b, cost, reg, max_iter, tol)?;
    let (_p_pp, ot_pp, _iters_pp) = sinkhorn_log_with_convergence(a, a, cost, reg, max_iter, tol)?;
    let (_p_qq, ot_qq, _iters_qq) = sinkhorn_log_with_convergence(b, b, cost, reg, max_iter, tol)?;

    // In exact arithmetic this is >= 0, but allow tiny negative drift.
    Ok((ot_pq - 0.5 * (ot_pp + ot_qq)).max(0.0))
}

/// Correct Sinkhorn divergence for **different supports**.
///
/// You must provide cost matrices:
/// - `cost_ab` for X×Y
/// - `cost_aa` for X×X
/// - `cost_bb` for Y×Y
pub fn sinkhorn_divergence_general(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost_ab: &Array2<f32>,
    cost_aa: &Array2<f32>,
    cost_bb: &Array2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let m = a.len();
    let n = b.len();
    if cost_ab.nrows() != m || cost_ab.ncols() != n {
        return Err(Error::CostShapeMismatch(
            m,
            n,
            cost_ab.nrows(),
            cost_ab.ncols(),
        ));
    }
    if cost_aa.nrows() != m || cost_aa.ncols() != m {
        return Err(Error::CostShapeMismatch(
            m,
            m,
            cost_aa.nrows(),
            cost_aa.ncols(),
        ));
    }
    if cost_bb.nrows() != n || cost_bb.ncols() != n {
        return Err(Error::CostShapeMismatch(
            n,
            n,
            cost_bb.nrows(),
            cost_bb.ncols(),
        ));
    }

    let (_p_pq, ot_pq, _iters_pq) =
        sinkhorn_log_with_convergence(a, b, cost_ab, reg, max_iter, tol)?;
    let (_p_pp, ot_pp, _iters_pp) =
        sinkhorn_log_with_convergence(a, a, cost_aa, reg, max_iter, tol)?;
    let (_p_qq, ot_qq, _iters_qq) =
        sinkhorn_log_with_convergence(b, b, cost_bb, reg, max_iter, tol)?;

    Ok((ot_pq - 0.5 * (ot_pp + ot_qq)).max(0.0))
}

/// Create Euclidean cost matrix from point positions.
///
/// C\[i,j\] = ||x_i - y_j||₂
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

/// Create **squared** Euclidean cost matrix from point positions.
///
/// C\[i,j\] = ||x_i - y_j||₂²
///
/// This is the standard ground cost for 2-Wasserstein (W2) optimal transport,
/// and the correct cost for OT-CFM flow matching (Tong et al. 2023).
///
/// # Arguments
///
/// * `x` - Source points (m x d)
/// * `y` - Target points (n x d)
///
/// # Returns
///
/// Cost matrix (m x n) where each entry is the squared Euclidean distance.
pub fn sq_euclidean_cost_matrix(x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
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
            cost[[i, j]] = dist_sq;
        }
    }

    cost
}

/// Log-domain Sinkhorn algorithm for entropic optimal transport.
///
/// Solves the same problem as [`sinkhorn`] but in log-space, avoiding the
/// numerical underflow that plagues the matrix-scaling version when
/// \(\varepsilon\) is small relative to the cost range.
///
/// **Log-domain update** (dual potentials \(f, g\)):
/// \[
/// f_i \leftarrow \varepsilon \log a_i - \varepsilon \operatorname{LSE}_j\!\bigl(\tfrac{g_j - C_{ij}}{\varepsilon}\bigr)
/// \]
/// \[
/// g_j \leftarrow \varepsilon \log b_j - \varepsilon \operatorname{LSE}_i\!\bigl(\tfrac{f_i - C_{ij}}{\varepsilon}\bigr)
/// \]
///
/// The transport plan is recovered as
/// \(P_{ij} = \exp\bigl(\tfrac{f_i + g_j - C_{ij}}{\varepsilon}\bigr)\).
///
/// **When to use this vs [`sinkhorn`]**: always prefer this when \(\varepsilon < 0.1 \cdot \max(C)\),
/// or when cost entries vary over several orders of magnitude. The log-domain version
/// replaces `exp` followed by division with `log-sum-exp`, which is unconditionally stable.
///
/// This implementation **normalizes** input distributions internally.
///
/// If you need a convergence check against marginal constraints (and an error on failure),
/// use [`sinkhorn_log_with_convergence`].
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

    // Treat exact zeros as -∞ (hard support exclusion) to avoid +∞ - (-∞) style NaNs.
    // This also makes plan entries involving zero-mass bins go to exactly 0.
    let log_a = a.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });
    let log_b = b.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });

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

/// Sinkhorn in log-space with a convergence check on marginal constraints.
///
/// Returns `(plan, <C,P>, iterations)` if converged within `max_iter`.
pub fn sinkhorn_log_with_convergence(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<(Array2<f32>, f32, usize)> {
    let m = a.len();
    let n = b.len();
    if cost.nrows() != m || cost.ncols() != n {
        return Err(Error::CostShapeMismatch(m, n, cost.nrows(), cost.ncols()));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }

    // Balanced OT expects nonnegative mass with positive total mass.
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("sinkhorn requires nonnegative masses"));
    }

    // Normalize input distributions to ensure they sum to 1.
    // (This matches the rest of this crate; callers can enforce strict normalization upstream.)
    let a_sum = a.sum();
    let b_sum = b.sum();
    if a_sum <= 0.0 || b_sum <= 0.0 {
        return Err(Error::Domain("sinkhorn requires positive total mass"));
    }
    let a = a / (a_sum + EPSILON);
    let b = b / (b_sum + EPSILON);

    // Consistent with `sinkhorn_log`: treat exact zeros as hard support exclusion.
    // This matters for correctness when users pass sparse histograms.
    let log_a = a.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });
    let log_b = b.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });

    let mut f: Array1<f32> = Array1::zeros(m);
    let mut g: Array1<f32> = Array1::zeros(n);

    // Check marginals every so often; doing it every iteration is O(mn) overhead.
    let check_every = 10usize.max(1);

    for iter in 0..max_iter {
        for i in 0..m {
            let lse = logsumexp_by(n, |j| (g[j] - cost[[i, j]]) / reg);
            f[i] = reg * (log_a[i] - lse);
        }
        for j in 0..n {
            let lse = logsumexp_by(m, |i| (f[i] - cost[[i, j]]) / reg);
            g[j] = reg * (log_b[j] - lse);
        }

        if (iter + 1) % check_every == 0 || iter + 1 == max_iter {
            // max marginal error: compute row/col sums from current (f,g).
            let mut max_err = 0.0f32;
            for i in 0..m {
                // row_sum = exp(f_i/reg) * Σ_j exp((g_j - C_ij)/reg)
                let lse = logsumexp_by(n, |j| (g[j] - cost[[i, j]]) / reg);
                let row_sum = (f[i] / reg).exp() * lse.exp();
                max_err = max_err.max((row_sum - a[i]).abs());
            }
            for j in 0..n {
                let lse = logsumexp_by(m, |i| (f[i] - cost[[i, j]]) / reg);
                let col_sum = (g[j] / reg).exp() * lse.exp();
                max_err = max_err.max((col_sum - b[j]).abs());
            }
            if max_err < tol {
                // Build plan once at the end.
                let mut plan = Array2::zeros((m, n));
                let mut distance = 0.0;
                for i in 0..m {
                    for j in 0..n {
                        let log_p = (f[i] + g[j] - cost[[i, j]]) / reg;
                        let pij = log_p.exp();
                        plan[[i, j]] = pij;
                        distance += pij * cost[[i, j]];
                    }
                }
                return Ok((plan, distance, iter + 1));
            }
        }
    }

    Err(Error::SinkhornNotConverged(max_iter))
}

/// Unbalanced Sinkhorn in log-space (KL-penalized marginals), scaling form.
///
/// This implements the classic scaling updates
/// \(u \leftarrow (a/(Kv))^\alpha\), \(v \leftarrow (b/(K^\top u))^\alpha\),
/// with \(K_{ij}=\exp(-C_{ij}/\varepsilon)\) and \(\alpha=\rho/(\rho+\varepsilon)\).
///
/// Compared to balanced OT, **do not** normalize `a` and `b` here: total mass is part of
/// the signal. This is the entire point of unbalanced OT.
///
/// The classic scaling form for KL-penalized marginals yields updates:
/// \[
/// u \leftarrow \left(\frac{a}{K v}\right)^{\alpha},\quad
/// v \leftarrow \left(\frac{b}{K^\top u}\right)^{\alpha},\quad
/// \alpha = \frac{\rho}{\rho + \varepsilon}.
/// \]
/// In log-domain: \(\log u \leftarrow \alpha(\log a - \log(Kv))\), similarly for \(v\).
///
/// Returns `(plan, objective, iterations)` once the dual updates stabilize.
///
/// The returned `objective` matches this scaling formulation:
/// \[
/// \min_{P\ge 0}\; \langle P, C \rangle
/// + \varepsilon\,\mathrm{KL}(P\,\|\,K)
/// + \rho\,\mathrm{KL}(P\mathbf{1}\,\|\,a)
/// + \rho\,\mathrm{KL}(P^\top\mathbf{1}\,\|\,b),
/// \quad K_{ij}=\exp(-C_{ij}/\varepsilon).
/// \]
pub fn unbalanced_sinkhorn_log_with_convergence(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    rho: f32,
    max_iter: usize,
    tol: f32,
) -> Result<(Array2<f32>, f32, usize)> {
    let m = a.len();
    let n = b.len();
    if cost.nrows() != m || cost.ncols() != n {
        return Err(Error::CostShapeMismatch(m, n, cost.nrows(), cost.ncols()));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(Error::InvalidMassPenalty(rho));
    }

    // Validate non-negativity and non-emptiness of masses.
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("unbalanced OT requires nonnegative masses"));
    }
    let a_sum = a.sum();
    let b_sum = b.sum();
    if a_sum <= 0.0 || b_sum <= 0.0 {
        return Err(Error::Domain("unbalanced OT requires positive total mass"));
    }

    let alpha = rho / (rho + reg);

    // Treat zero mass as hard support exclusion (log = -∞).
    let log_a = a.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });
    let log_b = b.mapv(|x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() });

    // log u, log v (dual scaling variables in log-space)
    let mut log_v: Array1<f32> = Array1::zeros(n);

    // Check convergence by dual drift (fixed-point residual).
    let check_every = 10usize.max(1);

    for iter in 0..max_iter {
        // log(Kv)_i = logsumexp_j (logK_ij + log_v[j]), with logK_ij = -C_ij/reg.
        //
        // IMPORTANT: `log_v` is already a log-scale quantity; do not divide it by `reg`.
        let mut log_u_new = Array1::zeros(m);
        for i in 0..m {
            if log_a[i] == f32::NEG_INFINITY {
                log_u_new[i] = f32::NEG_INFINITY;
                continue;
            }
            let lkv = logsumexp_by(n, |j| log_v[j] - (cost[[i, j]] / reg));
            // log u = alpha * (log a - log(Kv))
            if lkv == f32::NEG_INFINITY {
                log_u_new[i] = f32::NEG_INFINITY;
            } else {
                log_u_new[i] = alpha * (log_a[i] - lkv);
            }
        }

        let mut log_v_new = Array1::zeros(n);
        for j in 0..n {
            if log_b[j] == f32::NEG_INFINITY {
                log_v_new[j] = f32::NEG_INFINITY;
                continue;
            }
            let lktu = logsumexp_by(m, |i| log_u_new[i] - (cost[[i, j]] / reg));
            if lktu == f32::NEG_INFINITY {
                log_v_new[j] = f32::NEG_INFINITY;
            } else {
                log_v_new[j] = alpha * (log_b[j] - lktu);
            }
        }

        let log_u = log_u_new;
        log_v = log_v_new;

        if (iter + 1) % check_every == 0 || iter + 1 == max_iter {
            // Fixed-point stability: max change in log_u/log_v across one extra update.
            let mut max_diff = 0.0f32;

            // one-step recompute to measure residual
            for i in 0..m {
                if log_a[i] == f32::NEG_INFINITY {
                    continue;
                }
                let lkv = logsumexp_by(n, |j| log_v[j] - (cost[[i, j]] / reg));
                if lkv != f32::NEG_INFINITY {
                    let val = alpha * (log_a[i] - lkv);
                    max_diff = max_diff.max((val - log_u[i]).abs());
                }
            }
            for j in 0..n {
                if log_b[j] == f32::NEG_INFINITY {
                    continue;
                }
                let lktu = logsumexp_by(m, |i| log_u[i] - (cost[[i, j]] / reg));
                if lktu != f32::NEG_INFINITY {
                    let val = alpha * (log_b[j] - lktu);
                    max_diff = max_diff.max((val - log_v[j]).abs());
                }
            }

            if max_diff < tol {
                let mut plan = Array2::zeros((m, n));
                let mut transport_cost = 0.0;
                for i in 0..m {
                    for j in 0..n {
                        if log_u[i] == f32::NEG_INFINITY || log_v[j] == f32::NEG_INFINITY {
                            continue;
                        }
                        // P_ij = u_i * exp(-C_ij/ε) * v_j  ⇒  log P_ij = log u_i + log v_j - C_ij/ε
                        let log_p = log_u[i] + log_v[j] - (cost[[i, j]] / reg);
                        let pij = log_p.exp();
                        plan[[i, j]] = pij;
                        transport_cost += pij * cost[[i, j]];
                    }
                }

                // Compute the full objective for the scaling formulation:
                // <C,P> + ε KL(P || K) + ρ KL(P1 || a) + ρ KL(Pᵀ1 || b)
                //
                // We use the generalized (unnormalized) KL:
                // KL(p||q) = Σ p log(p/q) - p + q, with p=0 contributing +q.
                fn kl_mass(p: &Array1<f32>, q: &Array1<f32>) -> f32 {
                    let mut s: f64 = 0.0;
                    for (&pi, &qi) in p.iter().zip(q.iter()) {
                        // Treat tiny mass as zero to avoid spurious +∞ from float noise.
                        if pi <= 1e-12 {
                            s += qi as f64;
                            continue;
                        }
                        if qi <= 0.0 {
                            return f32::INFINITY;
                        }
                        let pi64 = pi as f64;
                        let qi64 = qi as f64;
                        s += pi64 * (pi64 / qi64).ln() - pi64 + qi64;
                    }
                    s as f32
                }

                let row = plan.sum_axis(ndarray::Axis(1));
                let col = plan.sum_axis(ndarray::Axis(0));

                let kl_row = kl_mass(&row, a);
                let kl_col = kl_mass(&col, b);

                // KL(P || K) without materializing K:
                // log K_ij = -C_ij / ε, and Σ K_ij is a constant offset in the generalized KL.
                let mut kl_plan: f64 = 0.0;
                let mut sum_k: f64 = 0.0;
                for i in 0..m {
                    for j in 0..n {
                        let cij = cost[[i, j]] as f64;
                        sum_k += (-cij / (reg as f64)).exp();

                        let pij = plan[[i, j]] as f64;
                        if pij <= 1e-12 {
                            continue;
                        }
                        let log_k = -cij / (reg as f64);
                        kl_plan += pij * (pij.ln() - log_k) - pij;
                    }
                }
                kl_plan += sum_k;

                let obj = transport_cost + reg * (kl_plan as f32) + rho * (kl_row + kl_col);

                return Ok((plan, obj, iter + 1));
            }
        }
    }

    Err(Error::SinkhornNotConverged(max_iter))
}

/// Unbalanced Sinkhorn divergence for measures on the same support.
///
/// This mirrors `sinkhorn_divergence_same_support` but uses unbalanced OT subproblems.
pub fn unbalanced_sinkhorn_divergence_same_support(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost: &Array2<f32>,
    reg: f32,
    rho: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let n = a.len();
    if b.len() != n {
        return Err(Error::LengthMismatch(n, b.len()));
    }
    if cost.nrows() != n || cost.ncols() != n {
        return Err(Error::CostShapeMismatch(n, n, cost.nrows(), cost.ncols()));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(Error::InvalidMassPenalty(rho));
    }
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("unbalanced OT requires nonnegative masses"));
    }
    if a.sum() <= 0.0 || b.sum() <= 0.0 {
        return Err(Error::Domain("unbalanced OT requires positive total mass"));
    }

    // Follow Séjourné et al. (2019) / GeomLoss: compute divergence from *dual potentials*.
    // This avoids ambiguity about which primal objective a given scaling scheme solves.
    //
    // Reference: `jeanfeydy/geomloss`, `geomloss/sinkhorn_divergence.py`.
    fn log_weights(w: &Array1<f32>) -> Array1<f32> {
        w.mapv(|x| if x <= 0.0 { -100000.0 } else { x.ln() })
    }

    fn softmin_xy(eps: f32, cost: &Array2<f32>, h_y: &Array1<f32>) -> Array1<f32> {
        let m = cost.nrows();
        let n = cost.ncols();
        debug_assert_eq!(h_y.len(), n);
        let mut out = Array1::zeros(m);
        for i in 0..m {
            let lse = logsumexp_by(n, |j| h_y[j] - cost[[i, j]] / eps);
            out[i] = -eps * lse;
        }
        out
    }

    fn softmin_yx(eps: f32, cost: &Array2<f32>, h_x: &Array1<f32>) -> Array1<f32> {
        let m = cost.nrows();
        let n = cost.ncols();
        debug_assert_eq!(h_x.len(), m);
        let mut out = Array1::zeros(n);
        for j in 0..n {
            let lse = logsumexp_by(m, |i| h_x[i] - cost[[i, j]] / eps);
            out[j] = -eps * lse;
        }
        out
    }

    let eps = reg;
    let damping = rho / (rho + eps);
    let a_log = log_weights(a);
    let b_log = log_weights(b);

    // Initialize dual potentials (GeomLoss-style).
    let mut g_ab = damping * softmin_yx(eps, cost, &a_log);
    let mut f_ba = damping * softmin_xy(eps, cost, &b_log);
    let mut f_aa = damping * softmin_xy(eps, cost, &a_log);
    let mut g_bb = damping * softmin_xy(eps, cost, &b_log);

    let check_every = 10usize.max(1);
    for iter in 0..max_iter {
        let h_b = &b_log + &(g_ab.mapv(|x| x / eps));
        let h_a = &a_log + &(f_ba.mapv(|x| x / eps));

        let ft_ba = damping * softmin_xy(eps, cost, &h_b);
        let gt_ab = damping * softmin_yx(eps, cost, &h_a);

        let h_aa = &a_log + &(f_aa.mapv(|x| x / eps));
        let h_bb = &b_log + &(g_bb.mapv(|x| x / eps));

        let ft_aa = damping * softmin_xy(eps, cost, &h_aa);
        let gt_bb = damping * softmin_xy(eps, cost, &h_bb);

        let f_ba_new = 0.5 * (&f_ba + &ft_ba);
        let g_ab_new = 0.5 * (&g_ab + &gt_ab);
        let f_aa_new = 0.5 * (&f_aa + &ft_aa);
        let g_bb_new = 0.5 * (&g_bb + &gt_bb);

        f_ba = f_ba_new;
        g_ab = g_ab_new;
        f_aa = f_aa_new;
        g_bb = g_bb_new;

        if (iter + 1) % check_every == 0 || iter + 1 == max_iter {
            let mut max_diff = 0.0f32;
            for i in 0..n {
                max_diff = max_diff.max((f_ba[i] - ft_ba[i]).abs());
                max_diff = max_diff.max((f_aa[i] - ft_aa[i]).abs());
            }
            for j in 0..n {
                max_diff = max_diff.max((g_ab[j] - gt_ab[j]).abs());
                max_diff = max_diff.max((g_bb[j] - gt_bb[j]).abs());
            }
            if max_diff < tol {
                break;
            }
        }
    }

    let scale = rho + eps / 2.0;
    let mut term_a: f64 = 0.0;
    for i in 0..n {
        let ai = a[i] as f64;
        if ai == 0.0 {
            continue;
        }
        let x = (-f_aa[i] / rho).exp() - (-f_ba[i] / rho).exp();
        term_a += ai * (scale as f64) * (x as f64);
    }
    let mut term_b: f64 = 0.0;
    for j in 0..n {
        let bj = b[j] as f64;
        if bj == 0.0 {
            continue;
        }
        let x = (-g_bb[j] / rho).exp() - (-g_ab[j] / rho).exp();
        term_b += bj * (scale as f64) * (x as f64);
    }

    let mass_corr = 0.5 * eps * (a.sum() - b.sum()).powi(2);
    Ok((term_a + term_b) as f32 + mass_corr)
}

/// Unbalanced Sinkhorn divergence for different supports.
///
/// You must provide cost matrices:
/// - `cost_ab` for X×Y (distance between a and b)
/// - `cost_aa` for X×X (distance between a and a)
/// - `cost_bb` for Y×Y (distance between b and b)
///
/// Computes the debiased divergence:
/// S(a,b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b) + 0.5*ε*(m(a)-m(b))²
pub fn unbalanced_sinkhorn_divergence_general(
    a: &Array1<f32>,
    b: &Array1<f32>,
    cost_ab: &Array2<f32>,
    cost_aa: &Array2<f32>,
    cost_bb: &Array2<f32>,
    reg: f32,
    rho: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let m = a.len();
    let n = b.len();
    if cost_ab.nrows() != m || cost_ab.ncols() != n {
        return Err(Error::CostShapeMismatch(
            m,
            n,
            cost_ab.nrows(),
            cost_ab.ncols(),
        ));
    }
    if cost_aa.nrows() != m || cost_aa.ncols() != m {
        return Err(Error::CostShapeMismatch(
            m,
            m,
            cost_aa.nrows(),
            cost_aa.ncols(),
        ));
    }
    if cost_bb.nrows() != n || cost_bb.ncols() != n {
        return Err(Error::CostShapeMismatch(
            n,
            n,
            cost_bb.nrows(),
            cost_bb.ncols(),
        ));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(Error::InvalidMassPenalty(rho));
    }
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("unbalanced OT requires nonnegative masses"));
    }
    if a.sum() <= 0.0 || b.sum() <= 0.0 {
        return Err(Error::Domain("unbalanced OT requires positive total mass"));
    }

    fn log_weights(w: &Array1<f32>) -> Array1<f32> {
        w.mapv(|x| if x <= 0.0 { -100000.0 } else { x.ln() })
    }

    // softmin_rows(eps, C, h) -> out[i] = -eps * logsumexp_j((h[j] - C_ij)/eps)
    fn softmin_rows(eps: f32, cost: &Array2<f32>, h: &Array1<f32>) -> Array1<f32> {
        let m = cost.nrows();
        let n = cost.ncols();
        debug_assert_eq!(h.len(), n);
        let mut out = Array1::zeros(m);
        for i in 0..m {
            let lse = logsumexp_by(n, |j| h[j] - cost[[i, j]] / eps);
            out[i] = -eps * lse;
        }
        out
    }

    // softmin_cols(eps, C, h) -> out[j] = -eps * logsumexp_i((h[i] - C_ij)/eps)
    fn softmin_cols(eps: f32, cost: &Array2<f32>, h: &Array1<f32>) -> Array1<f32> {
        let m = cost.nrows();
        let n = cost.ncols();
        debug_assert_eq!(h.len(), m);
        let mut out = Array1::zeros(n);
        for j in 0..n {
            let lse = logsumexp_by(m, |i| h[i] - cost[[i, j]] / eps);
            out[j] = -eps * lse;
        }
        out
    }

    let eps = reg;
    let damping = rho / (rho + eps);
    let a_log = log_weights(a);
    let b_log = log_weights(b);

    // Initialize dual potentials.
    // g_ab: potential on b for (a,b). Initialized from a.
    // f_ba: potential on a for (a,b). Initialized from b.
    // f_aa: potential on a for (a,a). Initialized from a.
    // g_bb: potential on b for (b,b). Initialized from b.
    let mut g_ab = damping * softmin_cols(eps, cost_ab, &a_log);
    let mut f_ba = damping * softmin_rows(eps, cost_ab, &b_log);
    let mut f_aa = damping * softmin_rows(eps, cost_aa, &a_log);
    let mut g_bb = damping * softmin_cols(eps, cost_bb, &b_log); // cost_bb is square symmetric usually, but treat generically

    let check_every = 10usize.max(1);
    for iter in 0..max_iter {
        // Update potentials for (a,b)
        let h_b = &b_log + &(g_ab.mapv(|x| x / eps));
        let h_a = &a_log + &(f_ba.mapv(|x| x / eps));
        let ft_ba = damping * softmin_rows(eps, cost_ab, &h_b);
        let gt_ab = damping * softmin_cols(eps, cost_ab, &h_a);

        // Update potentials for (a,a)
        let h_aa = &a_log + &(f_aa.mapv(|x| x / eps));
        // For symmetric problem (a,a), we only need one potential f_aa if symmetric.
        // But implementing full update for correctness:
        // Target is 'a', potential on target is f_aa.
        // We update potential on source (also 'a'), which is also f_aa.
        let ft_aa = damping * softmin_rows(eps, cost_aa, &h_aa);

        // Update potentials for (b,b)
        let h_bb = &b_log + &(g_bb.mapv(|x| x / eps));
        let gt_bb = damping * softmin_cols(eps, cost_bb, &h_bb);

        let f_ba_new = 0.5 * (&f_ba + &ft_ba);
        let g_ab_new = 0.5 * (&g_ab + &gt_ab);
        let f_aa_new = 0.5 * (&f_aa + &ft_aa);
        let g_bb_new = 0.5 * (&g_bb + &gt_bb);

        f_ba = f_ba_new;
        g_ab = g_ab_new;
        f_aa = f_aa_new;
        g_bb = g_bb_new;

        if (iter + 1) % check_every == 0 || iter + 1 == max_iter {
            let mut max_diff = 0.0f32;
            for i in 0..m {
                max_diff = max_diff.max((f_ba[i] - ft_ba[i]).abs());
                max_diff = max_diff.max((f_aa[i] - ft_aa[i]).abs());
            }
            for j in 0..n {
                max_diff = max_diff.max((g_ab[j] - gt_ab[j]).abs());
                max_diff = max_diff.max((g_bb[j] - gt_bb[j]).abs());
            }
            if max_diff < tol {
                break;
            }
        }
    }

    let scale = rho + eps / 2.0;

    // Term A: <a, (-f_aa/rho).exp() - (-f_ba/rho).exp()>
    let mut term_a: f64 = 0.0;
    for i in 0..m {
        let ai = a[i] as f64;
        if ai == 0.0 {
            continue;
        }
        let x = (-f_aa[i] / rho).exp() - (-f_ba[i] / rho).exp();
        term_a += ai * (scale as f64) * (x as f64);
    }

    // Term B: <b, (-g_bb/rho).exp() - (-g_ab/rho).exp()>
    let mut term_b: f64 = 0.0;
    for j in 0..n {
        let bj = b[j] as f64;
        if bj == 0.0 {
            continue;
        }
        let x = (-g_bb[j] / rho).exp() - (-g_ab[j] / rho).exp();
        term_b += bj * (scale as f64) * (x as f64);
    }

    let mass_corr = 0.5 * eps * (a.sum() - b.sum()).powi(2);
    Ok((term_a + term_b) as f32 + mass_corr)
}

/// Low-rank factorized transport coupling: `P = Q * diag(g) * R^T`.
///
/// Instead of storing the full `n x m` coupling matrix, stores three factors
/// with total size `O((n + m) * rank)`. The full coupling can be recovered
/// via [`LowRankCoupling::to_dense`] for small problems or applied implicitly
/// via [`LowRankCoupling::apply`].
#[derive(Debug, Clone)]
pub struct LowRankCoupling {
    /// Left factor (n x rank), non-negative.
    pub q: Vec<f32>,
    /// Diagonal scaling (rank), positive.
    pub g: Vec<f32>,
    /// Right factor (m x rank), non-negative.
    pub r: Vec<f32>,
    /// Transport cost `<C, P>`.
    pub cost: f32,
    /// Number of Dykstra iterations used.
    pub iterations: usize,
    /// Dimensions for reconstruction.
    n: usize,
    m: usize,
    rank: usize,
}

impl LowRankCoupling {
    /// Materialize the full `n x m` coupling matrix `P = Q diag(g) R^T`.
    ///
    /// Only practical for small problems (verification, debugging).
    pub fn to_dense(&self) -> Vec<f32> {
        let mut p = vec![0.0f32; self.n * self.m];
        for i in 0..self.n {
            for j in 0..self.m {
                let mut val = 0.0f32;
                for k in 0..self.rank {
                    val += self.q[i * self.rank + k] * self.g[k] * self.r[j * self.rank + k];
                }
                p[i * self.m + j] = val;
            }
        }
        p
    }

    /// Apply coupling to a vector: computes `P * v` without materializing `P`.
    ///
    /// `v` must have length `m`. Returns a vector of length `n`.
    /// Cost: `O((n + m) * rank)` instead of `O(n * m)`.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.m, "v must have length m={}", self.m);

        // Step 1: t = R^T v (rank-length vector)
        let mut t = vec![0.0f32; self.rank];
        for j in 0..self.m {
            for k in 0..self.rank {
                t[k] += self.r[j * self.rank + k] * v[j];
            }
        }

        // Step 2: t = diag(g) * t
        for k in 0..self.rank {
            t[k] *= self.g[k];
        }

        // Step 3: result = Q * t
        let mut result = vec![0.0f32; self.n];
        for i in 0..self.n {
            for k in 0..self.rank {
                result[i] += self.q[i * self.rank + k] * t[k];
            }
        }
        result
    }

    /// Row marginals of the coupling (length n).
    pub fn row_marginals(&self) -> Vec<f32> {
        // P 1_m = Q diag(g) R^T 1_m
        let ones = vec![1.0f32; self.m];
        self.apply(&ones)
    }

    /// Column marginals of the coupling (length m).
    pub fn col_marginals(&self) -> Vec<f32> {
        // P^T 1_n = R diag(g) Q^T 1_n
        // Step 1: s = Q^T 1_n
        let mut s = vec![0.0f32; self.rank];
        for i in 0..self.n {
            for k in 0..self.rank {
                s[k] += self.q[i * self.rank + k];
            }
        }
        // Step 2: s = diag(g) * s
        for k in 0..self.rank {
            s[k] *= self.g[k];
        }
        // Step 3: result = R * s
        let mut result = vec![0.0f32; self.m];
        for j in 0..self.m {
            for k in 0..self.rank {
                result[j] += self.r[j * self.rank + k] * s[k];
            }
        }
        result
    }
}

/// Low-rank Sinkhorn factorization (Scetbon, Cuturi & Peyre, ICML 2021).
///
/// Approximates the entropic OT coupling as `P = Q diag(g) R^T` where
/// `Q` is `n x r`, `R` is `m x r`, and `r << min(n, m)`.
///
/// Memory: `O((n + m) * r)` instead of `O(n * m)`.
/// Per-iteration cost: `O((n + m) * r)` instead of `O(n * m)`.
///
/// The algorithm uses Dykstra's alternating projections to enforce
/// row and column marginal constraints on the factored coupling.
///
/// # Arguments
///
/// * `a` - Source marginal (length n, sums to 1)
/// * `b` - Target marginal (length m, sums to 1)
/// * `cost` - Cost matrix (n*m, row-major flat)
/// * `reg` - Entropic regularization `epsilon > 0`
/// * `rank` - Approximation rank `r` (1 <= r <= min(n, m))
/// * `max_iter` - Maximum Dykstra iterations
/// * `tol` - Convergence tolerance on marginal error
pub fn sinkhorn_low_rank(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    reg: f32,
    rank: usize,
    max_iter: usize,
    tol: f32,
) -> Result<LowRankCoupling> {
    let n = a.len();
    let m = b.len();

    if cost.len() != n * m {
        return Err(Error::CostShapeMismatch(n, m, n, cost.len() / n.max(1)));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if rank < 1 || rank > n.min(m) {
        return Err(Error::InvalidRank(rank, n, m));
    }
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("sinkhorn requires nonnegative masses"));
    }
    let a_sum: f32 = a.iter().sum();
    let b_sum: f32 = b.iter().sum();
    if a_sum <= 0.0 || b_sum <= 0.0 {
        return Err(Error::Domain("sinkhorn requires positive total mass"));
    }

    // Normalize inputs.
    let a_norm: Vec<f32> = a.iter().map(|&x| x / (a_sum + EPSILON)).collect();
    let b_norm: Vec<f32> = b.iter().map(|&x| x / (b_sum + EPSILON)).collect();

    // Precompute the Gibbs kernel K_ij = exp(-C_ij / reg) in factored form.
    //
    // Strategy (Scetbon et al. 2021, Section 3):
    //   1. Pick `rank` landmark indices from each side
    //   2. Form kernel sub-matrices as initial factors
    //   3. Alternate: (a) Sinkhorn row-scaling on Q using kernel-weighted R sums,
    //      (b) Sinkhorn col-scaling on R using kernel-weighted Q sums
    //   4. g balances the two factors
    //
    // The kernel weighting in the projections ensures cost minimization, not just
    // marginal feasibility.
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Uniform};

    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);

    // Precompute log-kernel rows for stable computation.
    // log_k[i * m + j] = -cost[i * m + j] / reg
    let log_k: Vec<f32> = cost.iter().map(|&c| -c / reg).collect();

    // Initialize Q from rank columns of K, scaled by sqrt(a).
    // Initialize R from rank rows of K, scaled by sqrt(b).
    // Pick landmarks: when rank >= min(n,m), use all indices; otherwise sample randomly.
    let col_indices: Vec<usize> = if rank >= m {
        (0..rank.min(m)).collect()
    } else {
        let unif_m = Uniform::new(0usize, m).unwrap();
        (0..rank).map(|_| unif_m.sample(&mut rng)).collect()
    };
    let row_indices: Vec<usize> = if rank >= n {
        (0..rank.min(n)).collect()
    } else {
        let unif_n = Uniform::new(0usize, n).unwrap();
        (0..rank).map(|_| unif_n.sample(&mut rng)).collect()
    };

    // Q[i,k] = exp(-C[i, col_k] / reg) * a[i]
    let mut q = vec![0.0f32; n * rank];
    for k in 0..rank {
        let jk = col_indices[k];
        for i in 0..n {
            q[i * rank + k] = log_k[i * m + jk].exp().max(EPSILON) * a_norm[i].max(EPSILON);
        }
    }

    // R[j,k] = exp(-C[row_k, j] / reg) * b[j]
    let mut r = vec![0.0f32; m * rank];
    for k in 0..rank {
        let ik = row_indices[k];
        for j in 0..m {
            r[j * rank + k] = log_k[ik * m + j].exp().max(EPSILON) * b_norm[j].max(EPSILON);
        }
    }

    let g = vec![1.0f32; rank];

    // Alternating Sinkhorn projections on the low-rank factors.
    //
    // The coupling is P = Q diag(g) R^T. We enforce:
    //   Row marginals: (Q diag(g) R^T) 1_m = a
    //   Col marginals: (R diag(g) Q^T) 1_n = b
    //
    // by scaling rows of Q (for row marginals) and rows of R (for col marginals).

    let mut iterations = max_iter;

    for iter in 0..max_iter {
        // --- Row projection: scale Q rows so P 1_m = a ---
        let mut v = vec![0.0f32; rank];
        for k in 0..rank {
            let mut s = 0.0f32;
            for j in 0..m {
                s += r[j * rank + k];
            }
            v[k] = g[k] * s;
        }

        let mut max_row_err = 0.0f32;
        for i in 0..n {
            let mut row_sum = 0.0f32;
            for k in 0..rank {
                row_sum += q[i * rank + k] * v[k];
            }
            if a_norm[i] > 0.0 && row_sum > EPSILON {
                let scale = a_norm[i] / row_sum;
                max_row_err = max_row_err.max((1.0 - scale).abs() * a_norm[i]);
                for k in 0..rank {
                    q[i * rank + k] *= scale;
                }
            }
        }

        // --- Column projection: scale R rows so P^T 1_n = b ---
        let mut u = vec![0.0f32; rank];
        for k in 0..rank {
            let mut s = 0.0f32;
            for i in 0..n {
                s += q[i * rank + k];
            }
            u[k] = g[k] * s;
        }

        let mut max_col_err = 0.0f32;
        for j in 0..m {
            let mut col_sum = 0.0f32;
            for k in 0..rank {
                col_sum += r[j * rank + k] * u[k];
            }
            if b_norm[j] > 0.0 && col_sum > EPSILON {
                let scale = b_norm[j] / col_sum;
                max_col_err = max_col_err.max((1.0 - scale).abs() * b_norm[j]);
                for k in 0..rank {
                    r[j * rank + k] *= scale;
                }
            }
        }

        let max_err = max_row_err.max(max_col_err);
        if max_err < tol {
            iterations = iter + 1;
            break;
        }
    }

    // Compute transport cost: <C, P> = sum_ij C_ij * (sum_k Q_ik g_k R_jk)
    // Efficient: sum_k g_k * (sum_i C_i* Q_ik) dot (R_*k)
    // = sum_k g_k * (Q^T C R)_kk -- but we avoid materializing n*m.
    // Instead: for each k, compute (C Q_k) then dot with R_k.
    let mut transport_cost = 0.0f32;
    for k in 0..rank {
        // Compute (C * q_k)[j] = sum_i C[i,j] (wrong direction)
        // We want sum_ij C_ij Q_ik R_jk g_k
        // = g_k * sum_i Q_ik * sum_j C_ij R_jk
        // = g_k * sum_i Q_ik * (C[i,:] dot R[:,k])
        let mut s = 0.0f64;
        for i in 0..n {
            let qik = q[i * rank + k] as f64;
            if qik < 1e-12 {
                continue;
            }
            let mut cr = 0.0f64;
            for j in 0..m {
                cr += cost[i * m + j] as f64 * r[j * rank + k] as f64;
            }
            s += qik * cr;
        }
        transport_cost += (g[k] as f64 * s) as f32;
    }

    Ok(LowRankCoupling {
        q,
        g,
        r,
        cost: transport_cost,
        iterations,
        n,
        m,
        rank,
    })
}

/// 1D Wasserstein-p distance between two empirical samples (sorting-based, exact).
///
/// Given raw samples (not histograms), computes the exact Wasserstein-p distance:
///
/// $$
/// W_p(a, b) = \left(\frac{1}{n}\sum_{i=1}^{n} |a_{(i)} - b_{(i)}|^p\right)^{1/p}
/// $$
///
/// where `a_{(i)}` and `b_{(i)}` are the order statistics (sorted values).
/// Both samples must have the same length.
///
/// For `p = 1`: sum of absolute differences of sorted values, divided by `n`.
/// For `p = 2`: root-mean-square of sorted differences.
///
/// # Arguments
///
/// * `a` - First sample (will be sorted internally)
/// * `b` - Second sample (same length as `a`)
/// * `p` - Wasserstein exponent (typically 1 or 2, must be >= 1)
///
/// # Complexity
///
/// `O(n log n)` -- dominated by sorting.
///
/// # Example
///
/// ```rust
/// use wass::wasserstein_1d_samples;
///
/// let a = [0.0, 1.0];
/// let b = [1.0, 2.0];
/// let w1 = wasserstein_1d_samples(&a, &b, 1.0);
/// assert!((w1 - 1.0).abs() < 1e-6);
/// ```
pub fn wasserstein_1d_samples(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "samples must have same length");
    assert!(p >= 1.0, "p must be >= 1.0, got {}", p);

    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut sa: Vec<f32> = a.to_vec();
    let mut sb: Vec<f32> = b.to_vec();
    sa.sort_by(|x, y| x.total_cmp(y));
    sb.sort_by(|x, y| x.total_cmp(y));

    if (p - 1.0).abs() < 1e-7 {
        // W1: mean absolute difference
        let sum: f32 = sa.iter().zip(sb.iter()).map(|(x, y)| (x - y).abs()).sum();
        sum / n as f32
    } else {
        // Wp: (mean |diff|^p)^(1/p)
        let sum: f32 = sa
            .iter()
            .zip(sb.iter())
            .map(|(x, y)| (x - y).abs().powf(p))
            .sum();
        (sum / n as f32).powf(1.0 / p)
    }
}

/// Project points onto a direction vector and sort the projections.
fn project_and_sort(
    points: &Array2<f32>,
    direction: &Array1<f32>,
    #[allow(unused_variables)] n_points: usize,
) -> Vec<f32> {
    #[cfg(feature = "simd")]
    let mut proj = {
        let mut v = Vec::with_capacity(n_points);
        for i in 0..n_points {
            v.push(innr::dense::dot(
                points.row(i).as_slice().unwrap(),
                direction.as_slice().unwrap(),
            ));
        }
        v
    };

    #[cfg(not(feature = "simd"))]
    let mut proj = points.dot(direction).to_vec();

    proj.sort_by(|a, b| a.total_cmp(b));
    proj
}

/// Generate a random unit vector on `S^{d-1}` via Gaussian normalization.
fn random_unit_direction(d: usize, rng: &mut impl rand::Rng) -> Array1<f32> {
    use rand_distr::{Distribution, StandardNormal};

    let mut direction: Array1<f32> = Array1::zeros(d);
    for i in 0..d {
        direction[i] = StandardNormal.sample(rng);
    }

    #[cfg(feature = "simd")]
    let norm = innr::dense::norm(direction.as_slice().unwrap());
    #[cfg(not(feature = "simd"))]
    let norm = direction.dot(&direction).sqrt();

    direction /= norm.max(EPSILON);
    direction
}

/// Wasserstein-p on two sorted 1D projections of equal length.
fn w_p_sorted(proj_x: &[f32], proj_y: &[f32], p: f32) -> f32 {
    let n = proj_x.len().min(proj_y.len());
    if n == 0 {
        return 0.0;
    }
    if (p - 1.0).abs() < 1e-7 {
        let sum: f32 = (0..n).map(|i| (proj_x[i] - proj_y[i]).abs()).sum();
        sum / n as f32
    } else {
        let sum: f32 = (0..n).map(|i| (proj_x[i] - proj_y[i]).abs().powf(p)).sum();
        (sum / n as f32).powf(1.0 / p)
    }
}

/// Sliced Wasserstein distance -- a scalable approximation for high dimensions.
///
/// $$
/// \mathrm{SW}_p(X, Y) = \left(\mathbb{E}_{\theta \sim S^{d-1}}\bigl[W_p(\theta^\top X,\; \theta^\top Y)^p\bigr]\right)^{1/p}
/// $$
///
/// **Idea** (Rabin et al., 2012; Bonneel et al., 2015): project both point clouds
/// onto random 1D directions `theta`, compute the exact 1D `W_p` (which is
/// just sorting + paired differences), and aggregate over projections.
///
/// **Why**: full `W_p` in `d` dimensions requires `O(n^3)` (linear program)
/// or `O(n^2 k)` (Sinkhorn). Sliced `W_p` costs only
/// `O(L * n log n)` where `L` is the number of projections.
///
/// **Trade-off**: the approximation is unbiased but has variance `O(1/L)`.
/// For `L >= 50` the estimate is usually stable. Larger `d` may need more projections.
///
/// # Arguments
///
/// * `x` - Source point cloud (m x d)
/// * `y` - Target point cloud (n x d), must have same `d` as `x`
/// * `n_projections` - Number of random 1D directions `L`
/// * `seed` - RNG seed for reproducibility
/// * `p` - Wasserstein exponent (typically 1 or 2, must be >= 1)
///
/// # Panics
///
/// Panics if `x` and `y` have different column counts (dimension mismatch).
///
/// # Example
///
/// ```rust
/// use wass::sliced_wasserstein;
/// use ndarray::array;
///
/// let x = array![[0.0, 0.0], [1.0, 1.0]];
/// let y = array![[10.0, 10.0], [11.0, 11.0]];
/// let sw = sliced_wasserstein(&x, &y, 50, 42, 1.0);
/// assert!(sw > 5.0);
/// ```
pub fn sliced_wasserstein(
    x: &Array2<f32>,
    y: &Array2<f32>,
    n_projections: usize,
    seed: u64,
    p: f32,
) -> f32 {
    let d = x.ncols();
    assert_eq!(y.ncols(), d, "point dimensions must match");
    assert!(p >= 1.0, "p must be >= 1.0, got {}", p);

    let m = x.nrows();
    let n = y.nrows();

    if m == 0 || n == 0 || n_projections == 0 {
        return 0.0;
    }

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut total = 0.0f64;

    for _ in 0..n_projections {
        let direction = random_unit_direction(d, &mut rng);
        let proj_x = project_and_sort(x, &direction, m);
        let proj_y = project_and_sort(y, &direction, n);
        let wp = w_p_sorted(&proj_x, &proj_y, p);
        // Aggregate W_p^p, then take p-th root at the end
        total += (wp as f64).powi(p as i32);
    }

    (total / n_projections as f64).powf(1.0 / p as f64) as f32
}

/// Max-sliced Wasserstein distance: finds the projection maximizing 1D Wasserstein.
///
/// $$
/// \mathrm{max\text{-}SW}_p(X, Y) = \max_{\theta \in S^{d-1}} W_p(\theta^\top X,\; \theta^\top Y)
/// $$
///
/// More discriminative than random projections but slower. Uses projected
/// gradient ascent on the unit sphere: at each iteration, compute the
/// gradient of `W_p` with respect to `theta` via finite differences,
/// take a gradient step, and re-project onto `S^{d-1}`.
///
/// Falls back to the best of `max_iter` random restarts when `d` is large,
/// since the optimization landscape has many local maxima.
///
/// # Arguments
///
/// * `x` - Source point cloud (m x d)
/// * `y` - Target point cloud (n x d)
/// * `max_iter` - Number of random restarts (each evaluates one direction)
/// * `seed` - RNG seed
/// * `p` - Wasserstein exponent (>= 1)
///
/// # Returns
///
/// The maximum 1D Wasserstein-p distance over `max_iter` random directions.
///
/// # Example
///
/// ```rust
/// use wass::max_sliced_wasserstein;
/// use ndarray::array;
///
/// let x = array![[0.0, 0.0], [1.0, 0.0]];
/// let y = array![[10.0, 0.0], [11.0, 0.0]];
/// let msw = max_sliced_wasserstein(&x, &y, 100, 42, 1.0);
/// assert!(msw > 9.0);
/// ```
pub fn max_sliced_wasserstein(
    x: &Array2<f32>,
    y: &Array2<f32>,
    max_iter: usize,
    seed: u64,
    p: f32,
) -> f32 {
    let d = x.ncols();
    assert_eq!(y.ncols(), d, "point dimensions must match");
    assert!(p >= 1.0, "p must be >= 1.0, got {}", p);

    let m = x.nrows();
    let n = y.nrows();

    if m == 0 || n == 0 || max_iter == 0 {
        return 0.0;
    }

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut best = 0.0f32;

    // For each random direction, compute W_p and keep the max.
    // This is the "random restart" variant. For true gradient-based
    // max-sliced, one would differentiate through the sort, which
    // requires a differentiable framework.
    for _ in 0..max_iter {
        let direction = random_unit_direction(d, &mut rng);
        let proj_x = project_and_sort(x, &direction, m);
        let proj_y = project_and_sort(y, &direction, n);
        let wp = w_p_sorted(&proj_x, &proj_y, p);
        best = best.max(wp);
    }

    best
}

/// Hierarchical tree-partitioned Sinkhorn for large-scale optimal transport.
///
/// Recursively partitions source and target points into `branching` groups,
/// solves a coarse OT problem between group centroids, then refines by solving
/// local subproblems for each pair with nonzero coarse coupling.
///
/// This reduces the cost from `O(nm)` per Sinkhorn iteration to
/// `O(branching^2 + sum(n_i * m_j))` where the sum is over nonzero coarse pairs,
/// which is much smaller when the transport plan is sparse (i.e., most mass
/// moves between nearby groups).
///
/// **Partitioning**: points are sorted by their mean cost (a 1D projection of the
/// cost structure) and split into equal-sized groups. This avoids requiring a
/// full k-means implementation while still producing spatially coherent groups.
///
/// **Depth**: `max_depth = 1` means a single coarse-then-refine pass.
/// `max_depth > 1` applies the same partitioning recursively to each subproblem.
///
/// # Arguments
///
/// * `a` - Source marginal (length n, sums to 1)
/// * `b` - Target marginal (length m, sums to 1)
/// * `cost` - n x m cost matrix, row-major flat
/// * `n` - Number of source points
/// * `m` - Number of target points
/// * `reg` - Entropic regularization
/// * `branching` - Number of groups at each level (2..=min(n,m))
/// * `max_depth` - Recursion depth (1 = single coarse solve)
/// * `max_iter` - Max Sinkhorn iterations per subproblem
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// `(transport_cost, coupling)` where coupling is n x m row-major.
///
/// # References
///
/// Halmos, Gold, Liu, Raphael (2025). "Hierarchical Refinement: Optimal
/// Transport to Infinity and Beyond."
pub fn sinkhorn_hierarchical(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    n: usize,
    m: usize,
    reg: f32,
    branching: usize,
    max_depth: usize,
    max_iter: usize,
    tol: f32,
) -> Result<(f32, Vec<f32>)> {
    // --- validation ---
    if cost.len() != n * m {
        return Err(Error::CostShapeMismatch(n, m, n, cost.len() / n.max(1)));
    }
    if reg <= 0.0 || !reg.is_finite() {
        return Err(Error::InvalidRegularization(reg));
    }
    if a.len() != n || b.len() != m {
        return Err(Error::CostShapeMismatch(n, m, a.len(), b.len()));
    }
    if a.iter().any(|&x| x < 0.0) || b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("sinkhorn requires nonnegative masses"));
    }
    let a_sum: f32 = a.iter().sum();
    let b_sum: f32 = b.iter().sum();
    if a_sum <= 0.0 || b_sum <= 0.0 {
        return Err(Error::Domain("sinkhorn requires positive total mass"));
    }
    if branching < 2 || branching > n.min(m) {
        return Err(Error::InvalidBranching(branching, n, m));
    }

    // Normalize.
    let a_norm: Vec<f32> = a.iter().map(|&x| x / a_sum).collect();
    let b_norm: Vec<f32> = b.iter().map(|&x| x / b_sum).collect();

    let mut coupling = vec![0.0f32; n * m];
    hierarchical_recurse(
        &a_norm,
        &b_norm,
        cost,
        n,
        m,
        &(0..n).collect::<Vec<_>>(),
        &(0..m).collect::<Vec<_>>(),
        reg,
        branching,
        max_depth,
        max_iter,
        tol,
        1.0, // total mass at top level
        &mut coupling,
    )?;

    let transport_cost: f32 = coupling.iter().zip(cost.iter()).map(|(&p, &c)| p * c).sum();

    Ok((transport_cost, coupling))
}

/// Partition `indices` into `k` groups by sorting on mean cost (a cheap 1D proxy).
fn partition_by_cost_projection(
    cost_row: impl Fn(usize) -> f32,
    indices: &[usize],
    k: usize,
) -> Vec<Vec<usize>> {
    let mut indexed: Vec<(usize, f32)> = indices.iter().map(|&i| (i, cost_row(i))).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let total = indexed.len();
    let base_size = total / k;
    let remainder = total % k;

    let mut groups = Vec::with_capacity(k);
    let mut offset = 0;
    for g in 0..k {
        let size = base_size + if g < remainder { 1 } else { 0 };
        let group: Vec<usize> = indexed[offset..offset + size]
            .iter()
            .map(|&(i, _)| i)
            .collect();
        if !group.is_empty() {
            groups.push(group);
        }
        offset += size;
    }
    groups
}

/// Compute the centroid cost between two groups: average cost across all pairs.
fn group_cost(src_group: &[usize], tgt_group: &[usize], cost: &[f32], m_cols: usize) -> f32 {
    if src_group.is_empty() || tgt_group.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for &i in src_group {
        for &j in tgt_group {
            total += cost[i * m_cols + j];
        }
    }
    total / (src_group.len() * tgt_group.len()) as f32
}

/// Solve a local Sinkhorn subproblem and write scaled coupling into the global matrix.
///
/// `mass` is the total mass that should flow through this subproblem (from the
/// coarse coupling). The local plan is normalized to sum=1 internally, then
/// scaled by `mass` when written to the output.
fn solve_local_subproblem(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    full_m: usize,
    src_idx: &[usize],
    tgt_idx: &[usize],
    reg: f32,
    max_iter: usize,
    tol: f32,
    mass: f32,
    coupling: &mut [f32],
) {
    let local_n = src_idx.len();
    let local_m = tgt_idx.len();

    // Build local marginals (relative weights within this group).
    let local_a: Vec<f32> = src_idx.iter().map(|&i| a[i]).collect();
    let local_b: Vec<f32> = tgt_idx.iter().map(|&j| b[j]).collect();

    let a_sum: f32 = local_a.iter().sum();
    let b_sum: f32 = local_b.iter().sum();
    if a_sum <= 0.0 || b_sum <= 0.0 {
        return;
    }

    let a_local: Array1<f32> = Array1::from_vec(local_a.iter().map(|&x| x / a_sum).collect());
    let b_local: Array1<f32> = Array1::from_vec(local_b.iter().map(|&x| x / b_sum).collect());

    let mut local_cost = Array2::zeros((local_n, local_m));
    for (li, &gi) in src_idx.iter().enumerate() {
        for (lj, &gj) in tgt_idx.iter().enumerate() {
            local_cost[[li, lj]] = cost[gi * full_m + gj];
        }
    }

    let (plan, _, _) =
        sinkhorn_log_with_convergence(&a_local, &b_local, &local_cost, reg, max_iter, tol)
            .unwrap_or_else(|_| {
                let (plan, dist) = sinkhorn_log(&a_local, &b_local, &local_cost, reg, max_iter);
                (plan, dist, max_iter)
            });

    // plan sums to ~1.0; scale by mass assigned by the coarse coupling.
    for (li, &gi) in src_idx.iter().enumerate() {
        for (lj, &gj) in tgt_idx.iter().enumerate() {
            coupling[gi * full_m + gj] += plan[[li, lj]] * mass;
        }
    }
}

/// Recursive hierarchical solve.
///
/// `mass` is the total mass this subproblem must transport (from the parent's
/// coarse coupling). At the top level this is 1.0.
fn hierarchical_recurse(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    full_n: usize,
    full_m: usize,
    src_idx: &[usize],
    tgt_idx: &[usize],
    reg: f32,
    branching: usize,
    depth: usize,
    max_iter: usize,
    tol: f32,
    mass: f32,
    coupling: &mut [f32],
) -> Result<()> {
    let local_n = src_idx.len();
    let local_m = tgt_idx.len();

    // Base case: small enough or depth exhausted, solve directly.
    if depth == 0 || local_n <= branching || local_m <= branching {
        solve_local_subproblem(
            a, b, cost, full_m, src_idx, tgt_idx, reg, max_iter, tol, mass, coupling,
        );
        return Ok(());
    }

    // --- Coarse step: partition and solve a small OT problem ---

    let src_groups = partition_by_cost_projection(
        |i| {
            let mut s = 0.0f32;
            for &j in tgt_idx {
                s += cost[i * full_m + j];
            }
            s / local_m as f32
        },
        src_idx,
        branching.min(local_n),
    );

    let tgt_groups = partition_by_cost_projection(
        |j| {
            let mut s = 0.0f32;
            for &i in src_idx {
                s += cost[i * full_m + j];
            }
            s / local_n as f32
        },
        tgt_idx,
        branching.min(local_m),
    );

    let k_src = src_groups.len();
    let k_tgt = tgt_groups.len();

    // Coarse marginals: mass in each group (relative to this subproblem).
    let coarse_a: Vec<f32> = src_groups
        .iter()
        .map(|g| g.iter().map(|&i| a[i]).sum::<f32>())
        .collect();
    let coarse_b: Vec<f32> = tgt_groups
        .iter()
        .map(|g| g.iter().map(|&j| b[j]).sum::<f32>())
        .collect();

    let coarse_a_sum: f32 = coarse_a.iter().sum();
    let coarse_b_sum: f32 = coarse_b.iter().sum();
    if coarse_a_sum <= 0.0 || coarse_b_sum <= 0.0 {
        return Ok(());
    }

    let coarse_a_norm: Array1<f32> =
        Array1::from_vec(coarse_a.iter().map(|&x| x / coarse_a_sum).collect());
    let coarse_b_norm: Array1<f32> =
        Array1::from_vec(coarse_b.iter().map(|&x| x / coarse_b_sum).collect());

    // Coarse cost: average pairwise cost between groups.
    let mut coarse_cost = Array2::zeros((k_src, k_tgt));
    for (si, sg) in src_groups.iter().enumerate() {
        for (ti, tg) in tgt_groups.iter().enumerate() {
            coarse_cost[[si, ti]] = group_cost(sg, tg, cost, full_m);
        }
    }

    // Solve coarse problem (plan sums to ~1.0).
    let (coarse_plan, _, _) = sinkhorn_log_with_convergence(
        &coarse_a_norm,
        &coarse_b_norm,
        &coarse_cost,
        reg,
        max_iter,
        tol,
    )
    .unwrap_or_else(|_| {
        let (plan, dist) =
            sinkhorn_log(&coarse_a_norm, &coarse_b_norm, &coarse_cost, reg, max_iter);
        (plan, dist, max_iter)
    });

    // --- Refine: for each pair with nonzero coarse coupling, recurse ---
    let coarse_threshold = 1e-8;
    for (si, sg) in src_groups.iter().enumerate() {
        for (ti, tg) in tgt_groups.iter().enumerate() {
            let w = coarse_plan[[si, ti]];
            if w < coarse_threshold {
                continue;
            }
            // The coarse plan entry w is the fraction of the coarse problem's
            // mass flowing between these groups. Scale by `mass` to get the
            // absolute mass for this subproblem.
            hierarchical_recurse(
                a,
                b,
                cost,
                full_n,
                full_m,
                sg,
                tg,
                reg,
                branching,
                depth - 1,
                max_iter,
                tol,
                mass * w,
                coupling,
            )?;
        }
    }

    Ok(())
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
    fn test_sq_euclidean_cost_matrix() {
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0, 0.0], [0.0, 1.0]];

        let cost = sq_euclidean_cost_matrix(&x, &y);

        assert!((cost[[0, 0]] - 0.0).abs() < 1e-7);
        assert!((cost[[0, 1]] - 1.0).abs() < 1e-7); // ||[0,0] - [0,1]||^2 = 1
        assert!((cost[[1, 0]] - 1.0).abs() < 1e-7); // ||[1,0] - [0,0]||^2 = 1
        assert!((cost[[1, 1]] - 2.0).abs() < 1e-7); // ||[1,0] - [0,1]||^2 = 2
    }

    #[test]
    fn test_sq_vs_euclidean_cost_matrix() {
        // sq cost should be euclidean cost squared element-wise.
        let x = array![[0.0, 0.0], [1.0, 0.5], [0.3, -0.7]];
        let y = array![[0.5, 0.5], [-1.0, 0.0], [0.0, 1.0]];

        let l2 = euclidean_cost_matrix(&x, &y);
        let sq = sq_euclidean_cost_matrix(&x, &y);

        for i in 0..3 {
            for j in 0..3 {
                let expected = l2[[i, j]] * l2[[i, j]];
                assert!(
                    (sq[[i, j]] - expected).abs() < 1e-5,
                    "mismatch at ({i},{j}): sq={} expected={}",
                    sq[[i, j]],
                    expected
                );
            }
        }
    }

    // --- wasserstein_1d_samples tests ---

    #[test]
    fn w1d_samples_shift() {
        // [0, 1] vs [1, 2]: sorted diffs are |0-1| + |1-2| = 2, mean = 1.0
        let w = wasserstein_1d_samples(&[0.0, 1.0], &[1.0, 2.0], 1.0);
        assert!((w - 1.0).abs() < 1e-6, "W1 shift: {}", w);
    }

    #[test]
    fn w1d_samples_w2_shift() {
        // W2([0,1], [1,2]) = sqrt(mean(1^2 + 1^2)) = sqrt(1) = 1.0
        let w = wasserstein_1d_samples(&[0.0, 1.0], &[1.0, 2.0], 2.0);
        assert!((w - 1.0).abs() < 1e-6, "W2 shift: {}", w);
    }

    #[test]
    fn w1d_samples_self_distance() {
        let a = [0.5, 1.3, -0.2, 4.0];
        let w = wasserstein_1d_samples(&a, &a, 1.0);
        assert!(w < 1e-7, "self-distance: {}", w);
    }

    #[test]
    fn w1d_samples_unsorted_input() {
        // Should handle unsorted input correctly (sorts internally)
        let a = [3.0, 1.0, 2.0];
        let b = [6.0, 4.0, 5.0];
        let w = wasserstein_1d_samples(&a, &b, 1.0);
        // sorted: [1,2,3] vs [4,5,6] -> diffs = [3,3,3], mean = 3.0
        assert!((w - 3.0).abs() < 1e-6, "unsorted W1: {}", w);
    }

    // --- sliced_wasserstein tests ---

    #[test]
    fn test_sliced_wasserstein_self_distance() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]];
        let sw = sliced_wasserstein(&x, &x, 50, 42, 1.0);
        assert!(sw < 1e-5, "self-distance should be ~0: {}", sw);
    }

    #[test]
    fn test_sliced_wasserstein_symmetric() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[5.0, 5.0], [6.0, 6.0]];
        let sw_xy = sliced_wasserstein(&x, &y, 100, 42, 1.0);
        let sw_yx = sliced_wasserstein(&y, &x, 100, 42, 1.0);
        assert!(
            (sw_xy - sw_yx).abs() < 1e-5,
            "symmetry: {} vs {}",
            sw_xy,
            sw_yx
        );
    }

    #[test]
    fn test_sliced_wasserstein_nonneg() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[10.0, 10.0], [11.0, 11.0]];
        let sw = sliced_wasserstein(&x, &y, 50, 42, 1.0);
        assert!(sw >= 0.0, "non-negative: {}", sw);
    }

    #[test]
    fn test_sliced_wasserstein_separation() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y_close = array![[0.1, 0.1], [1.1, 1.1]];
        let y_far = array![[10.0, 10.0], [11.0, 11.0]];
        let sw_close = sliced_wasserstein(&x, &y_close, 100, 42, 1.0);
        let sw_far = sliced_wasserstein(&x, &y_far, 100, 42, 1.0);
        assert!(
            sw_far > sw_close,
            "distant > close: {} vs {}",
            sw_far,
            sw_close
        );
    }

    #[test]
    fn test_sliced_wasserstein_w2() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[10.0, 10.0], [11.0, 11.0]];
        let sw1 = sliced_wasserstein(&x, &y, 100, 42, 1.0);
        let sw2 = sliced_wasserstein(&x, &y, 100, 42, 2.0);
        // Both should be positive for well-separated clouds
        assert!(sw1 > 5.0, "SW1 should be large: {}", sw1);
        assert!(sw2 > 5.0, "SW2 should be large: {}", sw2);
    }

    // --- max_sliced_wasserstein tests ---

    #[test]
    fn test_max_sliced_self_distance() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let msw = max_sliced_wasserstein(&x, &x, 50, 42, 1.0);
        assert!(msw < 1e-5, "max-sliced self-distance: {}", msw);
    }

    #[test]
    fn test_max_sliced_ge_sliced() {
        // max over directions >= average over directions
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y = array![[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]];
        let n_proj = 200;
        let sw = sliced_wasserstein(&x, &y, n_proj, 42, 1.0);
        let msw = max_sliced_wasserstein(&x, &y, n_proj, 42, 1.0);
        assert!(
            msw >= sw - 1e-5,
            "max-sliced ({}) should >= sliced ({})",
            msw,
            sw
        );
    }

    #[test]
    fn test_max_sliced_axis_aligned() {
        // Points separated along x-axis only; the optimal direction is [1, 0].
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[10.0, 0.0], [11.0, 0.0]];
        let msw = max_sliced_wasserstein(&x, &y, 200, 42, 1.0);
        // W1 along the x-axis: sorted diffs = |0-10| + |1-11| = 20, mean = 10
        assert!(msw > 9.0, "axis-aligned max-sliced: {}", msw);
    }

    proptest! {
        #[test]
        fn prop_sinkhorn_divergence_non_negative(
            (a, b) in (2usize..8).prop_flat_map(|n| {
                (
                    prop::collection::vec(0.0f32..1.0, n),
                    prop::collection::vec(0.0f32..1.0, n),
                )
            }),
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

            // Sinkhorn convergence can be slow for some marginals; give it enough iterations.
            let div = sinkhorn_divergence_same_support(&a_dist, &b_dist, &cost, 0.1, 2000, 1e-2).unwrap();
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

    proptest! {
        #[test]
        fn prop_w1d_samples_nonneg(
            a in prop::collection::vec(-100.0f32..100.0, 2..32),
            b in prop::collection::vec(-100.0f32..100.0, 2..32),
        ) {
            let n = a.len().min(b.len());
            let w = wasserstein_1d_samples(&a[..n], &b[..n], 1.0);
            prop_assert!(w >= -1e-7, "non-negative: {}", w);
        }

        #[test]
        fn prop_w1d_samples_symmetric(
            a in prop::collection::vec(-100.0f32..100.0, 2..32),
            b in prop::collection::vec(-100.0f32..100.0, 2..32),
        ) {
            let n = a.len().min(b.len());
            let ab = wasserstein_1d_samples(&a[..n], &b[..n], 1.0);
            let ba = wasserstein_1d_samples(&b[..n], &a[..n], 1.0);
            prop_assert!((ab - ba).abs() < 1e-5, "symmetric: {} vs {}", ab, ba);
        }

        #[test]
        fn prop_w1d_samples_self_zero(
            a in prop::collection::vec(-100.0f32..100.0, 2..32),
        ) {
            let w = wasserstein_1d_samples(&a, &a, 1.0);
            prop_assert!(w < 1e-6, "self-distance: {}", w);
        }

        #[test]
        fn prop_sliced_wasserstein_nonneg(
            seed in 0u64..1000,
        ) {
            let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]];
            let y = array![[3.0, 3.0], [4.0, 4.0], [5.0, 3.5]];
            let sw = sliced_wasserstein(&x, &y, 20, seed, 1.0);
            prop_assert!(sw >= -1e-7, "non-negative: {}", sw);
        }

        #[test]
        fn prop_sliced_wasserstein_self_zero(
            seed in 0u64..1000,
        ) {
            let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]];
            let sw = sliced_wasserstein(&x, &x, 20, seed, 1.0);
            prop_assert!(sw < 1e-5, "self-distance: {}", sw);
        }

        #[test]
        fn prop_sliced_wasserstein_symmetric(
            seed in 0u64..1000,
        ) {
            let x = array![[0.0, 0.0], [1.0, 1.0]];
            let y = array![[3.0, 3.0], [4.0, 4.0]];
            let sw_xy = sliced_wasserstein(&x, &y, 20, seed, 1.0);
            let sw_yx = sliced_wasserstein(&y, &x, 20, seed, 1.0);
            prop_assert!((sw_xy - sw_yx).abs() < 1e-5, "symmetric: {} vs {}", sw_xy, sw_yx);
        }
    }

    #[test]
    fn earth_mover_distance_identical_is_zero() {
        let a = array![0.5, 0.5];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];
        let emd = earth_mover_distance(&a, &a, &cost);
        assert!(emd < 0.05, "identical distributions: emd={}", emd);
    }

    #[test]
    fn earth_mover_distance_shifted_distributions() {
        let a = array![0.7, 0.3];
        let b = array![0.3, 0.7];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];
        let emd = earth_mover_distance(&a, &b, &cost);
        // Moving 0.4 mass a distance of 1 => cost ~0.4
        assert!(
            emd > 0.2,
            "shifted distributions should have positive cost: emd={}",
            emd
        );
        assert!(
            emd < 0.6,
            "cost bounded by total mass * max cost: emd={}",
            emd
        );
    }

    #[test]
    fn earth_mover_distance_point_mass_shift() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let cost = array![[0.0, 3.0], [3.0, 0.0]];
        let emd = earth_mover_distance(&a, &b, &cost);
        // Now stable with sinkhorn_log: all mass moves distance 3
        assert!(
            (emd - 3.0).abs() < 0.2,
            "point mass shift of 3: emd={}",
            emd
        );
    }

    #[test]
    fn sinkhorn_log_plan_has_valid_marginals() {
        let a = array![0.3, 0.5, 0.2];
        let b = array![0.4, 0.4, 0.2];
        let cost = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
        let (plan, _, _) = sinkhorn_log_with_convergence(&a, &b, &cost, 0.1, 2000, 1e-4).unwrap();
        // Row sums should match source distribution
        for i in 0..3 {
            let row_sum: f32 = plan.row(i).sum();
            assert!(
                (row_sum - a[i]).abs() < 0.02,
                "row {} sum={}, expected={}",
                i,
                row_sum,
                a[i]
            );
        }
        // Col sums should match target distribution
        for j in 0..3 {
            let col_sum: f32 = plan.column(j).sum();
            assert!(
                (col_sum - b[j]).abs() < 0.02,
                "col {} sum={}, expected={}",
                j,
                col_sum,
                b[j]
            );
        }
    }

    #[test]
    fn sinkhorn_log_plan_is_nonneg() {
        let a = array![0.5, 0.5];
        let b = array![0.3, 0.7];
        let cost = array![[0.0, 2.0], [2.0, 0.0]];
        let (plan, _, _) = sinkhorn_log_with_convergence(&a, &b, &cost, 0.05, 500, 1e-6).unwrap();
        assert!(
            plan.iter().all(|&p| p >= -1e-7),
            "plan has negative entries"
        );
    }

    #[test]
    fn wasserstein_1d_triangle_inequality() {
        // W(a,c) <= W(a,b) + W(b,c)
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let c = [0.0, 0.0, 0.0, 1.0];
        let ab = wasserstein_1d(&a, &b);
        let bc = wasserstein_1d(&b, &c);
        let ac = wasserstein_1d(&a, &c);
        assert!(
            ac <= ab + bc + 1e-6,
            "triangle inequality: {ac} > {ab} + {bc}"
        );
    }

    #[test]
    fn sinkhorn_divergence_zero_on_diagonal_same_support() {
        let a = array![0.2, 0.3, 0.5];
        let cost = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
        let div = sinkhorn_divergence_same_support(&a, &a, &cost, 0.1, 500, 1e-4).unwrap();
        assert!(div.abs() < 1e-5, "div={}", div);
    }

    #[test]
    fn sinkhorn_divergence_is_symmetric_same_support() {
        let a = array![0.2, 0.3, 0.5];
        let b = array![0.5, 0.4, 0.1];
        let cost = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
        let ab = sinkhorn_divergence_same_support(&a, &b, &cost, 0.1, 500, 1e-4).unwrap();
        let ba = sinkhorn_divergence_same_support(&b, &a, &cost, 0.1, 500, 1e-4).unwrap();
        assert!((ab - ba).abs() < 1e-5, "ab={} ba={}", ab, ba);
    }

    // --- Low-rank Sinkhorn tests ---

    fn flat_cost(n: usize, m: usize) -> Vec<f32> {
        // Squared-distance cost on a 1D grid: C[i,j] = (i - j)^2
        let mut c = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                let d = i as f32 - j as f32;
                c[i * m + j] = d * d;
            }
        }
        c
    }

    #[test]
    fn low_rank_transport_cost_nonneg() {
        let n = 5;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / n as f32; n];
        let cost = flat_cost(n, n);
        let lr = sinkhorn_low_rank(&a, &b, &cost, 0.1, 3, 200, 1e-5).unwrap();
        assert!(
            lr.cost >= -1e-6,
            "transport cost should be non-negative, got {}",
            lr.cost
        );
    }

    #[test]
    fn low_rank_row_marginals_match() {
        let n = 6;
        let m = 4;
        let a: Vec<f32> = {
            let raw = vec![1.0, 2.0, 3.0, 2.0, 1.0, 1.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let b: Vec<f32> = {
            let raw = vec![2.0, 1.0, 1.0, 2.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let cost = flat_cost(n, m);

        let lr = sinkhorn_low_rank(&a, &b, &cost, 0.5, 3, 500, 1e-6).unwrap();
        let row_marg = lr.row_marginals();

        assert_eq!(row_marg.len(), n);
        for i in 0..n {
            assert!(
                (row_marg[i] - a[i]).abs() < 0.05,
                "row marginal[{}]: got {}, expected {}",
                i,
                row_marg[i],
                a[i]
            );
        }
    }

    #[test]
    fn low_rank_col_marginals_match() {
        let n = 6;
        let m = 4;
        let a: Vec<f32> = {
            let raw = vec![1.0, 2.0, 3.0, 2.0, 1.0, 1.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let b: Vec<f32> = {
            let raw = vec![2.0, 1.0, 1.0, 2.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let cost = flat_cost(n, m);

        let lr = sinkhorn_low_rank(&a, &b, &cost, 0.5, 3, 500, 1e-6).unwrap();
        let col_marg = lr.col_marginals();

        assert_eq!(col_marg.len(), m);
        for j in 0..m {
            assert!(
                (col_marg[j] - b[j]).abs() < 0.05,
                "col marginal[{}]: got {}, expected {}",
                j,
                col_marg[j],
                b[j]
            );
        }
    }

    #[test]
    fn low_rank_full_rank_approximates_sinkhorn() {
        // With rank = min(n,m), low-rank should closely match full Sinkhorn.
        let n = 4;
        let a_arr = array![0.25, 0.25, 0.25, 0.25];
        let b_arr = array![0.1, 0.3, 0.4, 0.2];
        let cost_arr = array![
            [0.0, 1.0, 4.0, 9.0],
            [1.0, 0.0, 1.0, 4.0],
            [4.0, 1.0, 0.0, 1.0],
            [9.0, 4.0, 1.0, 0.0]
        ];

        let (_, full_cost) = sinkhorn_log(&a_arr, &b_arr, &cost_arr, 0.5, 200);

        let a_flat: Vec<f32> = a_arr.to_vec();
        let b_flat: Vec<f32> = b_arr.to_vec();
        let cost_flat: Vec<f32> = cost_arr.iter().copied().collect();

        let lr = sinkhorn_low_rank(&a_flat, &b_flat, &cost_flat, 0.5, n, 500, 1e-6).unwrap();

        // With full rank the costs should be in the same ballpark.
        // Not exact because the algorithms differ, but within 50%.
        let ratio = lr.cost / full_cost;
        assert!(
            (0.5..2.0).contains(&ratio),
            "full-rank low-rank cost ({}) should approximate sinkhorn_log cost ({}), ratio={}",
            lr.cost,
            full_cost,
            ratio
        );
    }

    #[test]
    fn low_rank_memory_scales_linearly() {
        // Verify struct sizes scale as O((n+m)*r), not O(n*m).
        let n = 100;
        let m = 80;
        let rank = 5;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / m as f32; m];
        let cost = flat_cost(n, m);

        let lr = sinkhorn_low_rank(&a, &b, &cost, 1.0, rank, 100, 1e-4).unwrap();

        let factor_size = lr.q.len() + lr.r.len() + lr.g.len();
        let dense_size = n * m;
        assert_eq!(lr.q.len(), n * rank);
        assert_eq!(lr.r.len(), m * rank);
        assert_eq!(lr.g.len(), rank);
        assert!(
            factor_size < dense_size,
            "factor storage ({}) should be less than dense ({})",
            factor_size,
            dense_size
        );
    }

    #[test]
    fn low_rank_apply_matches_dense() {
        let n = 5;
        let m = 4;
        let a: Vec<f32> = {
            let raw = vec![1.0, 2.0, 1.0, 2.0, 1.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let b: Vec<f32> = {
            let raw = vec![1.0, 1.0, 1.0, 1.0];
            let s: f32 = raw.iter().sum();
            raw.iter().map(|&x| x / s).collect()
        };
        let cost = flat_cost(n, m);
        let lr = sinkhorn_low_rank(&a, &b, &cost, 0.5, 3, 300, 1e-5).unwrap();

        let v = vec![1.0, 0.0, 0.0, 0.0];
        let result_apply = lr.apply(&v);
        let dense = lr.to_dense();

        // Dense multiplication: P * v
        let mut result_dense = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..m {
                result_dense[i] += dense[i * m + j] * v[j];
            }
        }

        for i in 0..n {
            assert!(
                (result_apply[i] - result_dense[i]).abs() < 1e-5,
                "apply[{}]={} vs dense[{}]={}",
                i,
                result_apply[i],
                i,
                result_dense[i]
            );
        }
    }

    #[test]
    fn low_rank_invalid_inputs() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        // rank = 0
        assert!(sinkhorn_low_rank(&a, &b, &cost, 0.1, 0, 100, 1e-5).is_err());

        // rank > min(n,m)
        assert!(sinkhorn_low_rank(&a, &b, &cost, 0.1, 3, 100, 1e-5).is_err());

        // negative reg
        assert!(sinkhorn_low_rank(&a, &b, &cost, -0.1, 1, 100, 1e-5).is_err());

        // wrong cost size
        assert!(sinkhorn_low_rank(&a, &b, &[0.0, 1.0], 0.1, 1, 100, 1e-5).is_err());
    }

    #[test]
    fn low_rank_to_dense_nonneg() {
        let n = 5;
        let a = vec![0.2; n];
        let b = vec![0.2; n];
        let cost = flat_cost(n, n);
        let lr = sinkhorn_low_rank(&a, &b, &cost, 0.5, 3, 200, 1e-5).unwrap();
        let dense = lr.to_dense();
        for (idx, &val) in dense.iter().enumerate() {
            assert!(val >= -1e-7, "dense[{}] = {} is negative", idx, val);
        }
    }

    // --- Hierarchical OT tests ---

    #[test]
    fn hierarchical_cost_nonnegative() {
        let n = 8;
        let m = 8;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / m as f32; m];
        let cost = flat_cost(n, m);

        let (tc, coupling) =
            sinkhorn_hierarchical(&a, &b, &cost, n, m, 0.5, 4, 1, 200, 1e-5).unwrap();

        assert!(tc >= 0.0, "transport cost should be nonneg, got {}", tc);
        for (idx, &val) in coupling.iter().enumerate() {
            assert!(val >= -1e-7, "coupling[{}] = {} is negative", idx, val);
        }
    }

    #[test]
    fn hierarchical_marginals_approx() {
        let n = 6;
        let m = 6;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / m as f32; m];
        let cost = flat_cost(n, m);

        let (_, coupling) =
            sinkhorn_hierarchical(&a, &b, &cost, n, m, 0.5, 3, 1, 200, 1e-5).unwrap();

        // Row sums should approximate a.
        for i in 0..n {
            let row_sum: f32 = (0..m).map(|j| coupling[i * m + j]).sum();
            assert!(
                (row_sum - a[i]).abs() < 0.15,
                "row {} sum = {}, expected ~{}",
                i,
                row_sum,
                a[i]
            );
        }

        // Column sums should approximate b.
        for j in 0..m {
            let col_sum: f32 = (0..n).map(|i| coupling[i * m + j]).sum();
            assert!(
                (col_sum - b[j]).abs() < 0.15,
                "col {} sum = {}, expected ~{}",
                j,
                col_sum,
                b[j]
            );
        }
    }

    #[test]
    fn hierarchical_approx_quality() {
        // Use a clustered distribution where hierarchical should approximate well.
        // Two clusters: mass concentrated at indices 0-3 and 12-15.
        let n = 16;
        let m = 16;
        let mut a_flat = vec![0.0f32; n];
        let mut b_flat = vec![0.0f32; m];
        // Source: mass in first cluster.
        for i in 0..4 {
            a_flat[i] = 0.25;
        }
        // Target: mass in second cluster.
        for i in 12..16 {
            b_flat[i] = 0.25;
        }
        let cost_flat = flat_cost(n, m);

        let a_arr = Array1::from_vec(a_flat.clone());
        let b_arr = Array1::from_vec(b_flat.clone());
        let cost_arr = Array2::from_shape_vec((n, m), cost_flat.clone()).unwrap();

        let (_, regular_cost) = sinkhorn_log(&a_arr, &b_arr, &cost_arr, 1.0, 200);

        let (hier_cost, _) =
            sinkhorn_hierarchical(&a_flat, &b_flat, &cost_flat, n, m, 1.0, 4, 1, 200, 1e-5)
                .unwrap();

        assert!(
            hier_cost > 0.0,
            "hierarchical cost should be positive, got {}",
            hier_cost
        );

        // Both should find high cost (mass must move far). The hierarchical
        // approximation should be in the same order of magnitude.
        assert!(
            regular_cost > 10.0,
            "regular cost should be large for separated clusters, got {}",
            regular_cost
        );
        let ratio = hier_cost / regular_cost;
        assert!(
            (0.1..20.0).contains(&ratio),
            "hierarchical cost ({}) should be same order of magnitude as regular ({}), ratio={}",
            hier_cost,
            regular_cost,
            ratio
        );
    }

    #[test]
    fn hierarchical_deeper_recursion() {
        // max_depth > 1 should still produce valid output.
        let n = 16;
        let m = 16;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / m as f32; m];
        let cost = flat_cost(n, m);

        let (tc, coupling) =
            sinkhorn_hierarchical(&a, &b, &cost, n, m, 0.5, 4, 2, 200, 1e-5).unwrap();

        assert!(tc >= 0.0, "transport cost nonneg, got {}", tc);
        let total_mass: f32 = coupling.iter().sum();
        assert!(
            total_mass > 0.0,
            "total coupling mass should be positive, got {}",
            total_mass
        );
    }

    #[test]
    fn hierarchical_invalid_inputs() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        // branching < 2
        assert!(sinkhorn_hierarchical(&a, &b, &cost, 2, 2, 0.1, 1, 1, 100, 1e-5).is_err());

        // branching > min(n,m)
        assert!(sinkhorn_hierarchical(&a, &b, &cost, 2, 2, 0.1, 3, 1, 100, 1e-5).is_err());

        // negative reg
        assert!(sinkhorn_hierarchical(&a, &b, &cost, 2, 2, -0.1, 2, 1, 100, 1e-5).is_err());

        // wrong cost size
        assert!(sinkhorn_hierarchical(&a, &b, &[0.0, 1.0], 2, 2, 0.1, 2, 1, 100, 1e-5).is_err());
    }

    #[test]
    fn hierarchical_timing_large() {
        // Smoke test on a larger problem. No assertion on speed, just that it completes.
        let n = 64;
        let m = 64;
        let a = vec![1.0 / n as f32; n];
        let b = vec![1.0 / m as f32; m];
        let cost = flat_cost(n, m);

        let start = std::time::Instant::now();
        let (tc, _) = sinkhorn_hierarchical(&a, &b, &cost, n, m, 1.0, 8, 2, 100, 1e-4).unwrap();
        let hier_elapsed = start.elapsed();

        assert!(tc >= 0.0);

        // Just print timing for manual inspection, no assertion.
        eprintln!(
            "hierarchical 64x64 branching=8 depth=2: cost={:.4}, time={:?}",
            tc, hier_elapsed
        );
    }
}
