//! Point Cloud Registration via Sinkhorn OT
//!
//! Generates two 2D point clouds -- a source circle and a target that is the same
//! circle rotated, translated, and perturbed with Gaussian noise -- then uses
//! Sinkhorn to compute a transport plan and applies barycentric mapping to
//! register the source onto the target.
//!
//! Run: cargo run -p wass --example point_cloud_registration

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::f32::consts::PI;
use wass::sinkhorn_log_with_convergence;

/// Generate `n` points equally spaced on a circle of given radius centered at the origin.
fn circle_points(n: usize, radius: f32) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * i as f32 / n as f32;
            [radius * theta.cos(), radius * theta.sin()]
        })
        .collect()
}

/// Rotate a point cloud by `angle` radians, translate by `(dx, dy)`, and add
/// independent Gaussian noise with the given standard deviation.
fn transform(
    pts: &[[f32; 2]],
    angle: f32,
    dx: f32,
    dy: f32,
    noise_std: f32,
    rng: &mut impl rand::Rng,
) -> Vec<[f32; 2]> {
    let normal = Normal::new(0.0f32, noise_std).unwrap();
    let (sin, cos) = angle.sin_cos();
    pts.iter()
        .map(|&[x, y]| {
            let rx = cos * x - sin * y + dx + normal.sample(rng);
            let ry = sin * x + cos * y + dy + normal.sample(rng);
            [rx, ry]
        })
        .collect()
}

/// Squared Euclidean distance between two 2D points.
fn sq_dist(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// For each point in `transported`, find the distance to its nearest neighbor in `target`.
fn nearest_neighbor_distances(transported: &[[f32; 2]], target: &[[f32; 2]]) -> Vec<f32> {
    transported
        .iter()
        .map(|p| {
            target
                .iter()
                .map(|q| sq_dist(p, q).sqrt())
                .fold(f32::INFINITY, f32::min)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 30;
    let radius = 5.0;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // -- Generate point clouds --
    let source = circle_points(n, radius);
    let target = transform(&source, /*angle=*/ 0.4, /*dx=*/ 2.0, /*dy=*/ -1.5, /*noise=*/ 0.3, &mut rng);

    // -- Cost matrix: pairwise squared Euclidean distance --
    let mut cost = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cost[[i, j]] = sq_dist(&source[i], &target[j]);
        }
    }

    // Uniform weights.
    let a = Array1::from_elem(n, 1.0 / n as f32);
    let b = a.clone();

    // -- Sinkhorn --
    let reg = 2.0;
    let max_iter = 1000;
    let tol = 1e-6;
    let (plan, distance, iters) =
        sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol)?;

    println!("Points per cloud   : {n}");
    println!("Transport cost     : {distance:.4}");
    println!("Converged in       : {iters} iterations (reg={reg}, tol={tol})");
    println!();

    // -- Barycentric mapping: transport source -> target --
    let mut transported: Vec<[f32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let row_sum: f32 = plan.row(i).sum();
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        for j in 0..n {
            let w = plan[[i, j]] / row_sum;
            x += w * target[j][0];
            y += w * target[j][1];
        }
        transported.push([x, y]);
    }

    // -- Registration error --
    let nn_dists = nearest_neighbor_distances(&transported, &target);
    let mean_err: f32 = nn_dists.iter().sum::<f32>() / nn_dists.len() as f32;
    let max_err: f32 = nn_dists.iter().cloned().fold(0.0f32, f32::max);

    println!("Registration error (transported -> nearest target):");
    println!("  mean : {mean_err:.4}");
    println!("  max  : {max_err:.4}");

    Ok(())
}
