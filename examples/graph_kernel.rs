//! Sliced Wasserstein graph kernel.
//!
//! Computes a kernel between graphs by treating each graph as a distribution
//! over node feature vectors, then measuring sliced Wasserstein distance
//! between these distributions.
//!
//! Pipeline:
//! 1. Extract node features from graph structure (degree, clustering coeff, etc.)
//! 2. Represent each graph as a point cloud in feature space
//! 3. Compute pairwise sliced Wasserstein distance
//! 4. Convert to kernel via K(G1, G2) = exp(-SW(G1, G2) / sigma)
//!
//! Reference: Ma, Kolouri & Muandet (2025), "Sliced Wasserstein Graph Kernels"

use ndarray::Array2;
use wass::sliced_wasserstein;

fn main() {
    // Three small graphs with different structures.

    // Graph 1: ring (cycle) -- 6 nodes, each with degree 2.
    let g1_adj = vec![
        vec![1, 5],
        vec![0, 2],
        vec![1, 3],
        vec![2, 4],
        vec![3, 5],
        vec![4, 0],
    ];

    // Graph 2: star -- node 0 connected to all others.
    let g2_adj = vec![
        vec![1, 2, 3, 4, 5],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
    ];

    // Graph 3: two triangles connected by an edge (barbell).
    let g3_adj = vec![
        vec![1, 2],
        vec![0, 2],
        vec![0, 1, 3],
        vec![2, 4, 5],
        vec![3, 5],
        vec![3, 4],
    ];

    // Extract node features for each graph as Array2 (n x d).
    let f1 = extract_features(&g1_adj);
    let f2 = extract_features(&g2_adj);
    let f3 = extract_features(&g3_adj);

    // Compute pairwise sliced Wasserstein distances.
    let n_projections = 100;
    let seed = 42;

    let d12 = sliced_wasserstein(&f1, &f2, n_projections, seed, 1.0);
    let d13 = sliced_wasserstein(&f1, &f3, n_projections, seed, 1.0);
    let d23 = sliced_wasserstein(&f2, &f3, n_projections, seed, 1.0);
    let d11 = sliced_wasserstein(&f1, &f1, n_projections, seed, 1.0);

    println!("Sliced Wasserstein graph kernel");
    println!();
    println!("Graphs:");
    println!("  G1: ring (6 nodes, all degree 2)");
    println!("  G2: star (6 nodes, hub + 5 leaves)");
    println!("  G3: barbell (two triangles bridged)");
    println!();
    println!("Node features per graph (degree, clustering_coeff, neighbor_degree_avg):");
    println!("  G1: {} nodes x {} features", f1.nrows(), f1.ncols());
    println!("  G2: {} nodes x {} features", f2.nrows(), f2.ncols());
    println!("  G3: {} nodes x {} features", f3.nrows(), f3.ncols());
    println!();

    println!("Pairwise sliced Wasserstein distances:");
    println!("  SW(G1, G1) = {d11:.6} (self-distance)");
    println!("  SW(G1, G2) = {d12:.6}");
    println!("  SW(G1, G3) = {d13:.6}");
    println!("  SW(G2, G3) = {d23:.6}");
    println!();

    // Convert to RBF kernel: K(G1, G2) = exp(-SW(G1,G2)^2 / (2*sigma^2))
    let sigma = 1.0;
    let k12 = (-d12 * d12 / (2.0 * sigma * sigma)).exp();
    let k13 = (-d13 * d13 / (2.0 * sigma * sigma)).exp();
    let k23 = (-d23 * d23 / (2.0 * sigma * sigma)).exp();

    println!("RBF kernel matrix (sigma={sigma}):");
    println!("       G1      G2      G3");
    println!("  G1  1.000   {k12:.4}  {k13:.4}");
    println!("  G2  {k12:.4}  1.000   {k23:.4}");
    println!("  G3  {k13:.4}  {k23:.4}  1.000");
    println!();

    // G1 (ring) should be more similar to G3 (barbell) than to G2 (star)
    // because both have relatively uniform degree distributions.
    if d13 < d12 {
        println!("Ring is closer to barbell than to star (expected: uniform-ish degree)");
    } else {
        println!("Star is closer to ring than barbell (unexpected)");
    }
}

/// Extract structural node features from an adjacency list.
///
/// Features per node (3D): [degree, clustering_coefficient, avg_neighbor_degree]
/// Returns Array2 of shape (n, 3).
fn extract_features(adj: &[Vec<usize>]) -> Array2<f32> {
    let n = adj.len();
    let dim = 3;
    let mut features = Array2::<f32>::zeros((n, dim));

    for i in 0..n {
        let degree = adj[i].len() as f32;
        features[[i, 0]] = degree;

        // Clustering coefficient: fraction of neighbor pairs that are connected.
        let neighbors = &adj[i];
        let k = neighbors.len();
        if k >= 2 {
            let mut triangles = 0;
            for &u in neighbors {
                for &v in neighbors {
                    if u < v && adj[u].contains(&v) {
                        triangles += 1;
                    }
                }
            }
            let max_triangles = k * (k - 1) / 2;
            features[[i, 1]] = triangles as f32 / max_triangles as f32;
        }

        // Average neighbor degree.
        if k > 0 {
            let avg_nd: f32 =
                neighbors.iter().map(|&j| adj[j].len() as f32).sum::<f32>() / k as f32;
            features[[i, 2]] = avg_nd;
        }
    }

    features
}
