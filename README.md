# wass

Optimal transport: Wasserstein distance, Sinkhorn algorithm.

(wass: from Wasserstein distance)

Dual-licensed under MIT or Apache-2.0.

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
| `sinkhorn` | General transport | O(n^2 x iter) |
| `sinkhorn_with_convergence` | With early stopping | O(n^2 x iter) |
| `earth_mover_distance` | Exact (approx) | O(n^2 x iter) |
| `euclidean_cost_matrix` | Point clouds | O(m x n x d) |
| `sliced_wasserstein` | High-dim approx | O(n_proj x n log n) |

## Why Optimal Transport?

- No support issues (unlike KL divergence)
- Geometry-aware comparison
- Meaningful interpolation between distributions
