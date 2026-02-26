//! Color Palette Transfer via Optimal Transport
//!
//! Computes a Sinkhorn transport plan between two small palettes (warm and cool
//! RGB triplets), then applies the plan to map the warm palette onto the cool one.
//!
//! This is a minimal version of the color-transfer application described in
//! Rabin et al. (2012), "Wasserstein Barycenter and Its Application to Texture Mixing."
//! Full image pipelines operate on pixel distributions in Lab color space; here we
//! work with 8 colors per palette to keep the output readable.
//!
//! Run: cargo run -p wass --example color_transfer

use ndarray::{Array1, Array2};
use wass::sinkhorn_log_with_convergence;

/// Euclidean distance in RGB space (each channel in 0..255).
fn rgb_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn print_palette(name: &str, palette: &[[f32; 3]]) {
    println!("{name}:");
    for (i, c) in palette.iter().enumerate() {
        println!(
            "  [{i}] R={:5.1}  G={:5.1}  B={:5.1}",
            c[0], c[1], c[2]
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -- Palettes --
    // Warm palette: reds, oranges, yellows.
    let warm: Vec<[f32; 3]> = vec![
        [220.0, 50.0, 30.0],  // red
        [240.0, 100.0, 20.0], // red-orange
        [250.0, 150.0, 30.0], // orange
        [255.0, 200.0, 50.0], // amber
        [255.0, 230.0, 80.0], // yellow
        [200.0, 40.0, 60.0],  // crimson
        [180.0, 70.0, 20.0],  // rust
        [230.0, 180.0, 60.0], // gold
    ];

    // Cool palette: blues, teals, purples.
    let cool: Vec<[f32; 3]> = vec![
        [30.0, 60.0, 200.0],  // blue
        [20.0, 100.0, 220.0], // cerulean
        [50.0, 150.0, 210.0], // sky
        [60.0, 200.0, 200.0], // teal
        [80.0, 220.0, 180.0], // aqua
        [70.0, 40.0, 180.0],  // indigo
        [100.0, 60.0, 160.0], // purple
        [40.0, 170.0, 190.0], // cyan
    ];

    let n = warm.len();
    assert_eq!(n, cool.len());

    print_palette("Warm palette (source)", &warm);
    println!();
    print_palette("Cool palette (target)", &cool);
    println!();

    // -- Cost matrix: pairwise RGB Euclidean distances --
    let mut cost = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cost[[i, j]] = rgb_distance(&warm[i], &cool[j]);
        }
    }

    // Uniform weights (each color equally important).
    let a = Array1::from_elem(n, 1.0 / n as f32);
    let b = a.clone();

    // -- Sinkhorn --
    let reg = 5.0; // moderate regularization (cost values are in 0..~400 range)
    let max_iter = 500;
    let tol = 1e-6;

    let (plan, distance, iters) =
        sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol)?;

    println!("Sinkhorn distance: {distance:.2}  (converged in {iters} iterations, reg={reg})");
    println!();

    // -- Print transport plan --
    println!("Transport plan (rows=warm, cols=cool), entries * {n}:");
    print!("{:>8}", "");
    for j in 0..n {
        print!("  cool{j}");
    }
    println!();
    for i in 0..n {
        print!("warm{i}  ", );
        for j in 0..n {
            // Scale by n so that a dominant entry reads close to 1.0.
            print!("  {:.3}", plan[[i, j]] * n as f32);
        }
        println!();
    }
    println!();

    // -- Apply transport: map each warm color to its barycentric image in cool space --
    println!("Transferred palette (warm -> cool via transport plan):");
    for i in 0..n {
        let row_sum: f32 = plan.row(i).sum();
        let mut mapped = [0.0f32; 3];
        for j in 0..n {
            let weight = plan[[i, j]] / row_sum;
            mapped[0] += weight * cool[j][0];
            mapped[1] += weight * cool[j][1];
            mapped[2] += weight * cool[j][2];
        }
        println!(
            "  warm[{i}] ({:5.1},{:5.1},{:5.1}) -> ({:5.1},{:5.1},{:5.1})",
            warm[i][0], warm[i][1], warm[i][2], mapped[0], mapped[1], mapped[2],
        );
    }

    Ok(())
}
