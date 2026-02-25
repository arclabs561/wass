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

use ndarray::{Array1, Array2};
use thiserror::Error;

pub mod flow;
pub mod gromov;
pub mod semidiscrete;
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

    /// Invalid regularization parameter.
    #[error("regularization parameter must be positive and finite, got {0}")]
    InvalidRegularization(f32),

    /// Invalid mass-variation penalty parameter for unbalanced OT.
    #[error("mass penalty parameter must be positive and finite, got {0}")]
    InvalidMassPenalty(f32),

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

/// Sliced Wasserstein distance -- a scalable approximation for high dimensions.
///
/// \[
/// \mathrm{SW}_1(X, Y) = \mathbb{E}_{\theta \sim S^{d-1}}\bigl[W_1(\theta^\top X,\; \theta^\top Y)\bigr]
/// \]
///
/// **Idea** (Rabin et al., 2012; Bonneel et al., 2015): project both point clouds
/// onto random 1D directions \(\theta\), compute the exact 1D \(W_1\) (which is
/// just sorting + CDF integration), and average over projections.
///
/// **Why**: full \(W_1\) in \(d\) dimensions requires \(O(n^3)\) (linear program)
/// or \(O(n^2 k)\) (Sinkhorn). Sliced \(W_1\) costs only
/// \(O(L \cdot n \log n)\) where \(L\) is the number of projections.
///
/// **Trade-off**: the approximation is unbiased but has variance \(O(1/L)\).
/// For \(L \ge 50\) the estimate is usually stable. Larger \(d\) may need more projections.
///
/// # Arguments
///
/// * `x` - Source point cloud (\(m \times d\))
/// * `y` - Target point cloud (\(n \times d\))
/// * `n_projections` - Number of random 1D directions \(L\)
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
            proj_x.push(innr::dense::dot(
                x.row(i).as_slice().unwrap(),
                direction.as_slice().unwrap(),
            ));
        }

        #[cfg(not(feature = "simd"))]
        let mut proj_x = x.dot(&direction).to_vec();

        #[cfg(feature = "simd")]
        let mut proj_y = Vec::with_capacity(n);
        #[cfg(feature = "simd")]
        for i in 0..n {
            proj_y.push(innr::dense::dot(
                y.row(i).as_slice().unwrap(),
                direction.as_slice().unwrap(),
            ));
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
        assert!(emd > 0.2, "shifted distributions should have positive cost: emd={}", emd);
        assert!(emd < 0.6, "cost bounded by total mass * max cost: emd={}", emd);
    }

    #[test]
    fn earth_mover_distance_point_mass_shift() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let cost = array![[0.0, 3.0], [3.0, 0.0]];
        let emd = earth_mover_distance(&a, &b, &cost);
        // Now stable with sinkhorn_log: all mass moves distance 3
        assert!((emd - 3.0).abs() < 0.2, "point mass shift of 3: emd={}", emd);
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
            assert!((row_sum - a[i]).abs() < 0.02, "row {} sum={}, expected={}", i, row_sum, a[i]);
        }
        // Col sums should match target distribution
        for j in 0..3 {
            let col_sum: f32 = plan.column(j).sum();
            assert!((col_sum - b[j]).abs() < 0.02, "col {} sum={}, expected={}", j, col_sum, b[j]);
        }
    }

    #[test]
    fn sinkhorn_log_plan_is_nonneg() {
        let a = array![0.5, 0.5];
        let b = array![0.3, 0.7];
        let cost = array![[0.0, 2.0], [2.0, 0.0]];
        let (plan, _, _) = sinkhorn_log_with_convergence(&a, &b, &cost, 0.05, 500, 1e-6).unwrap();
        assert!(plan.iter().all(|&p| p >= -1e-7), "plan has negative entries");
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
        assert!(ac <= ab + bc + 1e-6, "triangle inequality: {ac} > {ab} + {bc}");
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
}
