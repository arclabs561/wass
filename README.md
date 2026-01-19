# wass

Optimal transport primitives for geometry-aware distribution comparison.
Implements the Sinkhorn algorithm for entropy-regularized OT, including **unbalanced** transport for robust partial matching.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/wass) | [docs.rs](https://docs.rs/wass)

```rust
use wass::{wasserstein_1d, sinkhorn, unbalanced_sinkhorn_divergence_general};
use ndarray::array;

// 1D Wasserstein (fast, closed-form)
let a = [0.0, 0.25, 0.5, 0.25];
let b = [0.25, 0.5, 0.25, 0.0];
let w1 = wasserstein_1d(&a, &b);

// Unbalanced OT (Robust Document Alignment)
// Compare two distributions with different supports and outliers.
// "rho" controls how much we penalize mass creation/destruction.
let a_weights = array![0.5, 0.5]; // e.g. "AI", "Pizza"
let b_weights = array![0.5, 0.5]; // e.g. "ML", "Sushi"
// ... build cost matrices ...
let div = unbalanced_sinkhorn_divergence_general(
    &a_weights, &b_weights, 
    &cost_ab, &cost_aa, &cost_bb, 
    0.1,  // epsilon (blur)
    1.0,  // rho (unbalanced penalty)
    1000, 1e-3
).unwrap();
```

## Key Features

- **Balanced OT**: Standard Sinkhorn for probability distributions.
- **Unbalanced OT**: Robust transport for partial matches, outliers, and unnormalized measures (e.g. document alignment).
- **Sparse OT**: L2-regularized transport for interpretable, sparse alignments (via `sparse` module).
- **Log-domain stabilization**: Numerically stable implementations for small epsilon / large costs.
- **Divergences**: Proper debiased Sinkhorn divergences (positive, definite) for metric use.

## Examples

Run these to see OT in action:

- **Robust Document Alignment**: Shows how unbalanced OT aligns core topics while ignoring outliers (headers/footers/typos).
  ```bash
  cargo run -p wass --example noisy_ocr_matching
  ```

- **Mass Mismatch**: Shows how divergence scales with the unbalanced penalty parameter.
  ```bash
  cargo run -p wass --example unbalanced_sinkhorn_mass_mismatch
  ```

- **Balanced Divergence**:
  ```bash
  cargo run -p wass --example sinkhorn_divergence_same_support
  ```

## Functions

| Function | Use Case | Complexity |
|----------|----------|------------|
| `wasserstein_1d` | 1D distributions | O(n) |
| `sinkhorn` | General transport (dense) | O(n^2 x iter) |
| `sinkhorn_with_convergence` | With early stopping | O(n^2 x iter) |
| `unbalanced_sinkhorn_divergence_general` | Robust comparison (different supports) | O(mn x iter) |
| `sparse::solve_semidual_l2` | Sparse transport (L2) | O(n^2 x iter) |
| `sliced_wasserstein` | High-dim approx | O(n_proj x n log n) |

## Why Optimal Transport?

- **No support issues**: Unlike KL divergence, OT compares distributions with disjoint supports.
- **Geometry-aware**: Respects the underlying metric space (e.g. word embedding distance).
- **Robustness**: Unbalanced OT handles outliers and noise ("pizza" vs "sushi") without breaking the alignment of the signal ("AI" vs "ML").
