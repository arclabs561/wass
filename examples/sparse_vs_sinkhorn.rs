//! Compare sparse OT (L2 regularization) vs dense Sinkhorn (entropic regularization).
//!
//! Both solve the same transport problem, but sparse OT produces plans with
//! exact zeros -- useful when you need hard assignments between items.
//!
//! Run: cargo run --example sparse_vs_sinkhorn

use ndarray::{array, Array2};
use wass::sparse::{solve_semidual_l2, sparsity};
use wass::sinkhorn_log_with_convergence;

fn main() {
    // 4 sources -> 4 targets, grid cost
    let a = array![0.25, 0.25, 0.25, 0.25];
    let b = array![0.25, 0.25, 0.25, 0.25];

    // Cost: distance on a line (|i - j|)
    let mut cost_f64 = Array2::zeros((4, 4));
    let mut cost_f32 = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let c = (i as f64 - j as f64).abs();
            cost_f64[[i, j]] = c;
            cost_f32[[i, j]] = c as f32;
        }
    }

    println!("=== Sparse OT vs Sinkhorn ===\n");
    println!("Cost matrix (|i - j| distance on a line):");
    println!("{:6.2}\n", cost_f64);

    // Sinkhorn (entropic regularization) -- always dense
    let reg = 0.1;
    let (plan_sink, dist_sink, iters_sink) =
        sinkhorn_log_with_convergence(&a, &b, &cost_f32, reg as f32, 500, 1e-6).unwrap();
    let plan_sink_f64 = plan_sink.mapv(|x| x as f64);
    let sp_sink = sparsity(&plan_sink_f64, 1e-6);

    println!("Sinkhorn (epsilon={}):", reg);
    println!("  plan:\n{:8.4}", plan_sink);
    println!("  distance: {:.4}, iters: {}, sparsity: {:.1}%\n", dist_sink, iters_sink, sp_sink * 100.0);

    // Sparse OT (L2 regularization) -- produces zeros
    let gamma = 1.0;
    let (plan_sparse, dist_sparse, iters_sparse) =
        solve_semidual_l2(&a.mapv(|x| x as f64), &b.mapv(|x| x as f64), &cost_f64, gamma, 1000, 1e-6).unwrap();
    let sp_sparse = sparsity(&plan_sparse, 1e-6);

    println!("Sparse OT (gamma={}):", gamma);
    println!("  plan:\n{:8.4}", plan_sparse);
    println!("  distance: {:.4}, iters: {}, sparsity: {:.1}%\n", dist_sparse, iters_sparse, sp_sparse * 100.0);

    // Try increasing gamma for more sparsity
    for &g in &[0.5, 2.0, 5.0] {
        let (plan, dist, _) =
            solve_semidual_l2(&a.mapv(|x| x as f64), &b.mapv(|x| x as f64), &cost_f64, g, 1000, 1e-6).unwrap();
        let sp = sparsity(&plan, 1e-6);
        println!("gamma={:.1}: distance={:.4}, sparsity={:.1}%", g, dist, sp * 100.0);
    }

    println!();
    println!("Key insight: Sinkhorn plans are always dense (all entries > 0).");
    println!("Sparse OT produces exact zeros, giving hard assignments.");
    println!("Higher gamma = more regularization = sparser plans.");
}
