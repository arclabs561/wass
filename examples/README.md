# Examples

## Which example should I run?

| I want to...                              | Example                                                          |
|-------------------------------------------|------------------------------------------------------------------|
| Compare two histograms on the same bins   | `sinkhorn_divergence_same_support`                               |
| Align documents or noisy text             | `document_alignment_demo`, `noisy_ocr_matching`                  |
| Match graphs without shared coordinates   | `gromov_wasserstein_graph_match`                                 |
| Get sparse (hard-assignment) plans        | `sparse_vs_sinkhorn`                                             |
| Handle outliers or mass mismatch          | `unbalanced_outlier_tradeoff`, `unbalanced_sinkhorn_mass_mismatch` |
| Transfer a color palette via OT           | `color_transfer`                                                 |

## Example descriptions

- **`sinkhorn_divergence_same_support`** -- Computes Sinkhorn divergence between two histograms on shared bins. Shows the difference between raw Sinkhorn cost and debiased divergence.

- **`document_alignment_demo`** -- Aligns two noisy documents (different phrasing, boilerplate, OCR errors) using unbalanced OT with char n-gram embeddings and TF-IDF weights. See also: `noisy_ocr_matching`.

- **`noisy_ocr_matching`** -- Aligns a clean reference string against a noisy OCR scan with headers/footers using unbalanced Sinkhorn. Focuses on the rho parameter's effect on outlier rejection. See also: `document_alignment_demo`.

- **`gromov_wasserstein_graph_match`** -- Recovers node correspondences between isomorphic graphs using Gromov-Wasserstein distance (no shared coordinate system needed). Shows both isomorphic (permuted) and non-isomorphic (path vs star) cases.

- **`sparse_vs_sinkhorn`** -- Compares dense Sinkhorn plans (entropic regularization, all entries positive) against sparse OT plans (L2 regularization, exact zeros). Shows the effect of the regularization parameter on sparsity.

- **`unbalanced_outlier_tradeoff`** -- Demonstrates how the rho parameter controls outlier sensitivity: small rho deletes outlier mass cheaply, large rho forces expensive transport.

- **`unbalanced_sinkhorn_mass_mismatch`** -- Shows unbalanced OT behavior when source and target have different total mass. Varying rho controls the penalty for mass creation/destruction.

- **`color_transfer`** -- Computes an OT plan between two small color palettes (warm and cool RGB triplets) using Sinkhorn, then applies the plan to map one palette onto the other. A minimal version of the classic color-transfer application.

## Running

```sh
cargo run -p wass --example <name>
```
