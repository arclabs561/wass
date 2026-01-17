# wass

Optimal transport primitives for geometry-aware distribution comparison.
Implements the Sinkhorn algorithm for entropy-regularized OT and fast 1D Wasserstein distances.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/wass) | [docs.rs](https://docs.rs/wass)

```rust
use wass::{wasserstein_1d, sinkhorn};
use ndarray::array;

// 1D Wasserstein (fast, closed-form)
let a = [0.0, 0.25, 0.5, 0.25];
let b = [0.25, 0.5, 0.25, 0.0];
let w1 = wasserstein_1d(&a, &b);

// General transport with Sinkhorn
let cost = array![[0.0, 1.0], [1.0, 0.0]];
let a = array![0.5, 0.5];
let b = array![0.5, 0.5];
let (plan, distance) = sinkhorn(&a, &b, &cost, 0.1, 100);
```

## Functions

| Function | Use Case | Complexity |
|----------|----------|------------|
| `wasserstein_1d` | 1D distributions | O(n) |
| `sinkhorn` | General transport (dense) | O(n^2 x iter) |
| `sinkhorn_with_convergence` | With early stopping | O(n^2 x iter) |
| `sparse::solve_semidual_l2` | Sparse transport (L2) | O(n^2 x iter) |
| `earth_mover_distance` | Exact (approx) | O(n^2 x iter) |
| `euclidean_cost_matrix` | Point clouds | O(m x n x d) |
| `sliced_wasserstein` | High-dim approx | O(n_proj x n log n) |

## Why Optimal Transport?

- No support issues (unlike KL divergence)
- Geometry-aware comparison
- Meaningful interpolation between distributions

## Sparse vs Dense OT

| Method | Regularization | Sparsity | Use Case |
|--------|---------------|----------|----------|
| `sinkhorn` | Entropy | Dense (0% zeros) | General purpose, smooth |
| `sparse::solve_semidual_l2` | L2 | Sparse (30-50% zeros) | Interpretable alignments, matching |

Sparse OT produces interpretable transport plans where many entries are exactly zero,
making it easier to see which sources map to which targets. This is valuable for
document alignment, entity matching, and other tasks requiring interpretability.
