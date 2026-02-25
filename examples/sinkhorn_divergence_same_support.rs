use ndarray::{array, Array2};

fn line_cost(n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

fn main() {
    // Two histograms on the same bins.
    let a = array![0.0, 0.2, 0.5, 0.3];
    let b = array![0.3, 0.5, 0.2, 0.0];
    let cost = line_cost(a.len());

    let reg = 0.1;
    let max_iter = 2000;
    let tol = 1e-2;

    let div = wass::sinkhorn_divergence_same_support(&a, &b, &cost, reg, max_iter, tol).unwrap();
    println!("Sinkhorn divergence (same support) = {div}");
}
