# wass

Optimal transport in Rust. Sinkhorn algorithm, unbalanced transport, sparse transport, Gromov-Wasserstein, semidiscrete OT.

## What it provides

| Function | What it does |
|---|---|
| `wasserstein_1d` | Closed-form 1D Wasserstein distance, O(n) |
| `sinkhorn` / `sinkhorn_log` | Entropy-regularized OT (log-domain for stability) |
| `sinkhorn_divergence_*` | Debiased Sinkhorn divergences (positive, symmetric) |
| `unbalanced_sinkhorn_*` | Robust OT for partial matching and outliers |
| `euclidean_cost_matrix` | L2 cost matrix from point clouds |
| `sq_euclidean_cost_matrix` | Squared L2 cost (correct for W2 OT-CFM) |
| `sliced_wasserstein` | High-dimensional approximation via random projections |
| `gromov_wasserstein` | Structure-preserving matching across metric spaces |
| `semidiscrete::fit_potentials_sgd_neg_dot` | Semidiscrete OT via SGD on dual potentials |
| `sparse::solve_semidual_l2` | L2-regularized sparse transport plans |

## Usage

```toml
[dependencies]
wass = "0.1.0"
```

```rust
use wass::{wasserstein_1d, sinkhorn_log_with_convergence};
use ndarray::array;

// 1D (closed-form)
let w1 = wasserstein_1d(&[0.0, 0.5, 0.5], &[0.5, 0.5, 0.0]);

// General (Sinkhorn, log-domain stable)
let a = array![0.5, 0.5];
let b = array![0.5, 0.5];
let cost = array![[0.0, 1.0], [1.0, 0.0]];
let (plan, dist, iters) = sinkhorn_log_with_convergence(
    &a, &b, &cost, 0.1, 1000, 1e-6
).unwrap();
```

## Examples

```bash
cargo run -p wass --example noisy_ocr_matching              # unbalanced OT for document alignment
cargo run -p wass --example unbalanced_sinkhorn_mass_mismatch # divergence vs mass penalty
cargo run -p wass --example sinkhorn_divergence_same_support  # balanced divergence
cargo run -p wass --example sparse_vs_sinkhorn               # L2 sparse plans vs entropic dense plans
```

## Tests

```bash
cargo test -p wass
```

58 tests covering Sinkhorn convergence, transport plan marginal validity, divergence properties (symmetry, non-negativity, convexity, cost-shift invariance), unbalanced OT, Gromov-Wasserstein, sparse transport, semidiscrete OT, flow drift, and EMD.

## License

MIT OR Apache-2.0
