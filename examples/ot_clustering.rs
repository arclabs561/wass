// OT distance between clusters vs centroid distance.
//
// Centroid distance collapses each cluster to a single point, discarding shape.
// Wasserstein (Sinkhorn divergence) compares the full empirical distributions,
// so it can distinguish a tight spherical cluster from an elongated one even
// when their centroids are equidistant.
//
// Setup:
//   - Cluster A: tight sphere at (-4, 0)
//   - Cluster B: tight sphere at ( 4, 0)
//   - Cluster C: elongated ellipse at ( 0, 4), stretched along x-axis
//
// Centroid distances: d(A,C) ~ d(B,C) because centroids are roughly
// equidistant. But the Sinkhorn divergence between B and C differs from
// A and C because C's spread along x brings some of its mass closer to B's
// support region.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Generate `n` 2D points from a Gaussian with given center and per-axis std dev.
fn sample_cluster(
    rng: &mut StdRng,
    cx: f32,
    cy: f32,
    std_x: f32,
    std_y: f32,
    n: usize,
) -> Vec<Vec<f32>> {
    let dx = Normal::new(cx as f64, std_x as f64).expect("valid normal");
    let dy = Normal::new(cy as f64, std_y as f64).expect("valid normal");
    (0..n)
        .map(|_| vec![dx.sample(rng) as f32, dy.sample(rng) as f32])
        .collect()
}

/// Squared Euclidean cost matrix between two point sets.
fn cost_matrix(xs: &[Vec<f32>], ys: &[Vec<f32>]) -> Array2<f32> {
    let m = xs.len();
    let n = ys.len();
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let dx = xs[i][0] - ys[j][0];
            let dy = xs[i][1] - ys[j][1];
            c[[i, j]] = dx * dx + dy * dy;
        }
    }
    c
}

/// Centroid of a point set.
fn centroid(pts: &[Vec<f32>]) -> [f32; 2] {
    let n = pts.len() as f32;
    let sx: f32 = pts.iter().map(|p| p[0]).sum();
    let sy: f32 = pts.iter().map(|p| p[1]).sum();
    [sx / n, sy / n]
}

/// Euclidean distance between two 2D points.
fn euclidean(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Compute Sinkhorn divergence between two point sets treated as empirical
/// distributions (uniform weights).
fn cluster_sinkhorn_divergence(xs: &[Vec<f32>], ys: &[Vec<f32>]) -> f32 {
    let m = xs.len();
    let n = ys.len();

    let a = Array1::from_elem(m, 1.0 / m as f32);
    let b = Array1::from_elem(n, 1.0 / n as f32);

    let cost_ab = cost_matrix(xs, ys);
    let cost_aa = cost_matrix(xs, xs);
    let cost_bb = cost_matrix(ys, ys);

    let reg = 5.0; // large enough for f32 stability on squared-Euclidean costs
    let max_iter = 2000;
    let tol = 5e-2;

    wass::sinkhorn_divergence_general(&a, &b, &cost_ab, &cost_aa, &cost_bb, reg, max_iter, tol)
        .expect("Sinkhorn divergence converged")
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 80;

    // -- generate raw points ------------------------------------------------
    let cluster_a = sample_cluster(&mut rng, -4.0, 0.0, 0.5, 0.5, n);
    let cluster_b = sample_cluster(&mut rng, 4.0, 0.0, 0.5, 0.5, n);
    // C: same centroid distance from A and B as each other, but elongated along x.
    let cluster_c = sample_cluster(&mut rng, 0.0, 4.0, 3.0, 0.3, n);

    // -- cluster with k-means -----------------------------------------------
    let all_points: Vec<Vec<f32>> = cluster_a
        .iter()
        .chain(cluster_b.iter())
        .chain(cluster_c.iter())
        .cloned()
        .collect();

    let fit = clump::Kmeans::new(3)
        .with_seed(7)
        .fit(&all_points)
        .expect("k-means converged");

    println!("k-means converged in {} iterations", fit.iters);
    println!("centroids:");
    for (i, c) in fit.centroids.iter().enumerate() {
        println!("  cluster {i}: ({:.2}, {:.2})", c[0], c[1]);
    }

    // -- group points by cluster label --------------------------------------
    let mut groups: Vec<Vec<Vec<f32>>> = vec![vec![]; 3];
    for (i, &label) in fit.labels.iter().enumerate() {
        groups[label].push(all_points[i].clone());
    }

    let names = ["X", "Y", "Z"];

    // -- centroid distances -------------------------------------------------
    println!("\n--- Centroid (Euclidean) distances ---");
    let centroids: Vec<[f32; 2]> = groups.iter().map(|g| centroid(g)).collect();
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d = euclidean(&centroids[i], &centroids[j]);
            println!("  d({}, {}) = {d:.4}", names[i], names[j]);
        }
    }

    // -- Sinkhorn divergences -----------------------------------------------
    println!("\n--- Sinkhorn divergence (distributional) ---");
    for i in 0..3 {
        for j in (i + 1)..3 {
            let sd = cluster_sinkhorn_divergence(&groups[i], &groups[j]);
            println!("  SD({}, {}) = {sd:.4}", names[i], names[j]);
        }
    }

    // -- ratio comparison: where shape matters --------------------------------
    // Collect pairwise values for a ratio comparison.
    let mut cd = [0.0f32; 3]; // centroid distances: (0,1), (0,2), (1,2)
    let mut sd = [0.0f32; 3]; // Sinkhorn divergences
    let mut idx = 0;
    for i in 0..3 {
        for j in (i + 1)..3 {
            cd[idx] = euclidean(&centroids[i], &centroids[j]);
            sd[idx] = cluster_sinkhorn_divergence(&groups[i], &groups[j]);
            idx += 1;
        }
    }

    println!("\n--- Ratio comparison ---");
    println!(
        "  centroid ratio  d({},{})/d({},{}) = {:.4}",
        names[0],
        names[2],
        names[1],
        names[2],
        cd[1] / cd[2]
    );
    println!(
        "  Sinkhorn ratio SD({},{})/SD({},{}) = {:.4}",
        names[0],
        names[2],
        names[1],
        names[2],
        sd[1] / sd[2]
    );
    println!();
    println!("Centroid distance treats each cluster as a point -- ratios reflect");
    println!("only center positions. Sinkhorn divergence sees the full distribution,");
    println!("so the elongated cluster (stretched along x) registers differently");
    println!("relative to each neighbour.");
}
