# wass

Optimal transport in Rust. Sinkhorn algorithm, unbalanced transport, sparse transport, Gromov-Wasserstein, semidiscrete OT.

## Problem

You have two distributions (point clouds, histograms, token sequences) and need to measure how far apart they are, or find the cheapest way to move mass from one to the other. Optimal transport gives a principled answer: the minimum-cost coupling between the two. This library provides the algorithms.

## Examples

**Noisy OCR alignment**. Given a clean reference and a noisy OCR scan with headers/footers, unbalanced Sinkhorn matches the real tokens while ignoring the junk:

```bash
cargo run --example noisy_ocr_matching
```

```text
Reference (9 tokens): "The quarterly earnings showed steady growth in all sectors"
Noisy OCR (20 tokens): "HEADER: CONFIDENTIAL 2025 The qarterly earnigns ..."

Aligning with Unbalanced Sinkhorn (epsilon=0.1)
Rho    Divergence Interpretation
------------------------------------------------------------
0.5    0.3150     Ignores outliers (robust)
  credible matches (p>=0.02, dist<=0.70):
    quarterly       -> qarterly         p=0.12  dist=0.38
    earnings        -> earnigns         p=0.10  dist=0.49
    showed          -> showd            p=0.10  dist=0.50
    growth          -> grwth            p=0.10  dist=0.48
    sectors         -> sectrs           p=0.11  dist=0.43
```

**Structure-preserving graph matching**. Gromov-Wasserstein aligns two metric spaces by their internal distance structure, without requiring them to share a common embedding:

```bash
cargo run --example gromov_wasserstein_graph_match
```

**Sparse vs. dense plans**. L2-regularized sparse transport plans vs. entropic Sinkhorn plans -- sparse plans have exact zeros, useful when you want hard assignments:

```bash
cargo run --example sparse_vs_sinkhorn
```

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

## Tests

```bash
cargo test -p wass
```

58 tests covering Sinkhorn convergence, transport plan marginal validity, divergence properties (symmetry, non-negativity, convexity, cost-shift invariance), unbalanced OT, Gromov-Wasserstein, sparse transport, semidiscrete OT, flow drift, and EMD.

## License

MIT OR Apache-2.0
