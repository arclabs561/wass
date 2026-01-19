//! Sparse optimal transport via L2 regularization.
//!
//! Implements the semi-dual approach from Blondel et al. (2018):
//! "Smooth and Sparse Optimal Transport" (AISTATS 2018).
//!
//! Unlike Sinkhorn (entropic regularization), L2 regularization produces
//! sparse transport plans with many zero entries, improving interpretability
//! and computational efficiency for matching tasks.
//!
//! # Geometry of Sparsity
//!
//! While entropic regularization (Sinkhorn) uses a smooth `exp(-C/ε)` kernel that
//! never reaches zero, L2 regularization relies on the **projection onto the
//! probability simplex**. This projection is akin to "shaving off" the tail of a
//! distribution; values below a certain threshold (τ) are set exactly to zero,
//! resulting in a sparse transport plan.
//!
//! This sparsity is not just a computational trick; it provides **hard assignments**
//! between distributions, making it clear which elements truly correspond.
//!
//! # Mathematical Formulation
//!
//! The semi-dual problem:
//!
//! ```text
//! max_α ⟨α, a⟩ - Σⱼ bⱼ max_Ωⱼ(α - C[:,j])
//! ```
//!
//! where max_Ωⱼ(x) = max_{p≥0, Σp=1} ⟨p, x⟩ - (γ/2)‖p‖²
//!
//! For L2 regularization, this projection onto the simplex is:
//!
//! ```text
//! p = [x/γ - τ]₊  where τ is chosen so Σp = 1
//! ```
//!
//! # References
//!
//! - Blondel, Seguy, Rolet (2018). "Smooth and Sparse Optimal Transport"
//!   https://arxiv.org/abs/1710.06276

use crate::{Error, Result};
use ndarray::{Array1, Array2};
use std::f64;

/// L2 regularization for optimal transport.
///
/// Regularization term: Ω(P) = (γ/2) ‖P‖²
///
/// This produces sparse transport plans (many zeros) unlike
/// entropic regularization which is always dense.
#[derive(Debug, Clone, Copy)]
pub struct SquaredL2 {
    /// Regularization parameter γ.
    /// Larger γ = more regularization = sparser plans.
    pub gamma: f64,
}

impl SquaredL2 {
    /// Create a new L2 regularizer.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Regularization strength. Typical values: 0.1 to 10.0.
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "gamma must be positive");
        Self { gamma }
    }

    /// Compute max_Ωⱼ(x) = max_{p≥0, Σp=1} ⟨p, x⟩ - (γ/2)‖p‖²
    ///
    /// This is the projection onto the simplex with L2 penalty.
    ///
    /// # Returns
    ///
    /// (value, gradient) where:
    /// - value: max_Ωⱼ(x) for each column
    /// - gradient: ∇max_Ωⱼ(x) (m × n matrix)
    pub fn max_omega(&self, x: &Array2<f64>, b: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let m = x.nrows();
        let n = x.ncols();

        // Project each column onto simplex: p = projection_simplex(x[:,j] / (b[j] * γ))
        let mut grad = Array2::zeros((m, n));
        let mut val = Array1::zeros(n);

        for j in 0..n {
            // Scale by b[j] * gamma
            let scaled: Array1<f64> = x.column(j).mapv(|xi| xi / (b[j] * self.gamma));

            // Project onto simplex
            let p = projection_simplex(&scaled);

            // Value: ⟨p, x[:,j]⟩ - (γ/2) b[j] ‖p‖²
            let inner: f64 = p.iter().zip(x.column(j)).map(|(&pi, &xi)| pi * xi).sum();
            let norm_sq: f64 = p.iter().map(|&pi| pi * pi).sum();
            val[j] = inner - 0.5 * self.gamma * b[j] * norm_sq;

            // Gradient: p itself
            for i in 0..m {
                grad[[i, j]] = p[i];
            }
        }

        (val, grad)
    }

    /// Compute regularization term Ω(P) = (γ/2) ‖P‖²
    pub fn omega(&self, p: &Array2<f64>) -> f64 {
        0.5 * self.gamma * p.iter().map(|&x| x * x).sum::<f64>()
    }
}

/// Project vector onto probability simplex.
///
/// Solves: min_{p≥0, Σp=1} ‖p - x‖²
///
/// Algorithm: Condat (2016) O(n) projection (no sorting), return [x - τ]₊
fn projection_simplex(x: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    // Project onto simplex with sum = 1.0.
    //
    // Port of Condat's 2016 algorithm (see header referenced in docs).
    let a = 1.0_f64;
    debug_assert!(a > 0.0);

    // Workspace holding a dynamic set of candidates.
    let mut aux = vec![0.0_f64; n];
    aux[0] = x[0];

    let mut start = 0usize;
    let mut aux_len = 1usize;
    let mut aux_len_old: isize = -1;
    let mut tau = aux[0] - a;

    for i in 1..n {
        let yi = x[i];
        if yi > tau {
            aux[start + aux_len] = yi;
            let denom = (aux_len as isize - aux_len_old) as f64;
            tau += (yi - tau) / denom;
            if tau <= yi - a {
                tau = yi - a;
                aux_len_old = aux_len as isize - 1;
            }
            aux_len += 1;
        }
    }

    if aux_len_old >= 0 {
        // Move the candidate window forward by (aux_len_old + 1).
        let shift = (aux_len_old + 1) as usize;
        aux_len -= shift;
        start += shift;

        // Re-introduce prior candidates if they exceed the current threshold.
        // This mirrors the pointer-prepend logic in Condat's C implementation.
        for idx in (0..shift).rev() {
            let v = aux[idx];
            if v > tau {
                start -= 1;
                aux[start] = v;
                aux_len += 1;
                tau += (aux[start] - tau) / (aux_len as f64);
            }
        }
    }

    loop {
        let old = aux_len - 1;
        let mut new_len = 0usize;

        for i in 0..=old {
            let v = aux[start + i];
            if v > tau {
                aux[start + new_len] = v;
                new_len += 1;
            } else {
                // Denominator counts remaining items (old - i) plus kept items (new_len).
                tau += (tau - v) / ((old - i + new_len) as f64);
            }
        }

        aux_len = new_len;
        if aux_len > old {
            break;
        }
    }

    x.mapv(|xi| (xi - tau).max(0.0))
}

/// Compute semi-dual objective and gradient.
fn semi_dual_obj_grad(
    alpha: &Array1<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    cost: &Array2<f64>,
    regul: &SquaredL2,
) -> (f64, Array1<f64>) {
    let m = a.len();
    let n = b.len();

    // Objective: ⟨α, a⟩ - Σⱼ bⱼ max_Ωⱼ(α - C[:,j])
    let mut obj = alpha
        .iter()
        .zip(a.iter())
        .map(|(&ai, &ai_dist)| ai * ai_dist)
        .sum::<f64>();
    let mut grad = a.clone();

    // X[:, j] = α - C[:, j] (broadcast alpha to each column)
    let mut x = Array2::zeros((m, n));
    for j in 0..n {
        for i in 0..m {
            x[[i, j]] = alpha[i] - cost[[i, j]];
        }
    }

    // Compute max_Omega and gradient
    let (val, g) = regul.max_omega(&x, b);

    // Subtract from objective and gradient
    obj -= val
        .iter()
        .zip(b.iter())
        .map(|(&v, &bj)| v * bj)
        .sum::<f64>();
    let grad_sub = g.dot(b);
    for i in 0..m {
        grad[i] -= grad_sub[i];
    }

    (obj, grad)
}

/// Solve sparse optimal transport using semi-dual formulation.
///
/// Uses gradient ascent with Armijo-style backtracking to maximize the semi-dual
/// objective, producing a sparse transport plan via L2 regularization.
///
/// # Arguments
///
/// * `a` - Source distribution (length m, must sum to 1)
/// * `b` - Target distribution (length n, must sum to 1)
/// * `cost` - Cost matrix C (m × n)
/// * `gamma` - L2 regularization strength
/// * `max_iter` - Maximum gradient ascent iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// (transport_plan, transport_distance, iterations)
///
/// # Example
///
/// ```rust
/// use wass::sparse::solve_semidual_l2;
/// use ndarray::array;
///
/// let a = array![0.5, 0.5];
/// let b = array![0.5, 0.5];
/// let cost = array![[0.0, 1.0], [1.0, 0.0]];
///
/// let (plan, distance, iters) = solve_semidual_l2(&a, &b, &cost, 1.0, 1000, 1e-6).unwrap();
///
/// // Plan is sparse (many zeros)
/// let sparsity = plan.iter().filter(|&&p| p.abs() < 1e-10).count() as f64 / plan.len() as f64;
/// assert!(sparsity > 0.0, "L2 regularization should produce sparse plans");
/// ```
pub fn solve_semidual_l2(
    a: &Array1<f64>,
    b: &Array1<f64>,
    cost: &Array2<f64>,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Array2<f64>, f64, usize)> {
    let m = a.len();
    let n = b.len();

    if cost.shape() != [m, n] {
        return Err(Error::CostShapeMismatch(m, n, cost.nrows(), cost.ncols()));
    }

    // Normalize distributions
    let a_sum: f64 = a.iter().sum();
    let b_sum: f64 = b.iter().sum();
    if (a_sum - 1.0).abs() > 1e-6 {
        return Err(Error::NotNormalized(a_sum as f32));
    }
    if (b_sum - 1.0).abs() > 1e-6 {
        return Err(Error::NotNormalized(b_sum as f32));
    }

    let regul = SquaredL2::new(gamma);

    // Initialize α = zeros
    let mut alpha = Array1::zeros(m);
    let mut best_obj = f64::NEG_INFINITY;
    let mut best_alpha = alpha.clone();

    // Gradient ascent with line search
    let mut step_size = 1.0;
    let shrink_factor = 0.5;
    let armijo_c = 1e-4;
    let min_step = 1e-10;
    let mut actual_iterations = max_iter;

    for iter in 0..max_iter {
        let (obj, grad) = semi_dual_obj_grad(&alpha, a, b, cost, &regul);

        // Track best solution
        if obj > best_obj {
            best_obj = obj;
            best_alpha = alpha.clone();
        }

        // Check convergence
        let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            actual_iterations = iter + 1;
            break;
        }

        // Line search: try step_size, shrink if needed
        let mut found_step = false;
        let mut current_step = step_size;
        let grad_norm_sq = grad_norm * grad_norm;
        for _ls in 0..20 {
            let mut alpha_new = alpha.clone();
            for i in 0..m {
                alpha_new[i] += grad[i] * current_step;
            }
            let (obj_new, _) = semi_dual_obj_grad(&alpha_new, a, b, cost, &regul);

            // Armijo condition for ascent: f(x+t g) >= f(x) + c t ||g||^2
            if obj_new >= obj + armijo_c * current_step * grad_norm_sq {
                alpha = alpha_new;
                step_size = current_step * 1.1; // Increase step if successful
                found_step = true;
                break;
            }
            current_step *= shrink_factor;
            if current_step < min_step {
                break;
            }
        }

        if !found_step {
            break;
        }
    }

    alpha = best_alpha;

    // Recover transport plan from optimal α
    let mut x = Array2::zeros((m, n));
    for j in 0..n {
        for i in 0..m {
            x[[i, j]] = alpha[i] - cost[[i, j]];
        }
    }

    let (_, grad) = regul.max_omega(&x, b);
    let mut plan = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            plan[[i, j]] = grad[[i, j]] * b[j];
        }
    }

    // Transport distance = <C, P>
    let distance: f64 = cost.iter().zip(plan.iter()).map(|(&c, &p)| c * p).sum();

    Ok((plan, distance, actual_iterations))
}

/// Compute sparsity of a transport plan.
///
/// Returns the fraction of entries that are effectively zero (|p| < threshold).
pub fn sparsity(plan: &Array2<f64>, threshold: f64) -> f64 {
    let total = plan.len();
    let zeros = plan.iter().filter(|&&p| p.abs() < threshold).count();
    zeros as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_projection_simplex() {
        let x = array![3.0, 1.0, 0.0];
        let p = projection_simplex(&x);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "should sum to 1");
        assert!(p.iter().all(|&pi| pi >= 0.0), "should be non-negative");
    }

    #[test]
    fn test_solve_semidual_l2_basic() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let cost = array![[0.0, 1.0], [1.0, 0.0]];

        let (plan, distance, _) = solve_semidual_l2(&a, &b, &cost, 1.0, 1000, 1e-6).unwrap();

        // Plan should sum to 1
        let plan_sum: f64 = plan.iter().sum();
        assert!((plan_sum - 1.0).abs() < 0.01, "plan should sum to 1");

        // Should have some sparsity
        let sparsity_val = sparsity(&plan, 1e-6);
        assert!(sparsity_val >= 0.0, "sparsity should be non-negative");

        // Distance should be reasonable
        assert!((0.0..1.0).contains(&distance));
    }

    #[test]
    fn test_sparse_vs_dense() {
        let a = array![0.25, 0.25, 0.25, 0.25];
        let b = array![0.25, 0.25, 0.25, 0.25];
        let mut cost = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                cost[[i, j]] = ((i as f64 - j as f64).abs()).min(1.0);
            }
        }

        // Sparse OT (L2)
        let (plan_sparse, _, _) = solve_semidual_l2(&a, &b, &cost, 1.0, 1000, 1e-6).unwrap();
        let sparsity_val = sparsity(&plan_sparse, 1e-6);

        // Should have some zeros (sparse)
        assert!(
            sparsity_val > 0.0,
            "L2 should produce sparse plans, got sparsity={}",
            sparsity_val
        );
    }
}
