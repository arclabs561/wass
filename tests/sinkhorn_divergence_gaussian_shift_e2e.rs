use ndarray::{Array1, Array2};

fn line_cost(n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

fn gaussian_hist(n: usize, mean: f32, sigma: f32) -> Array1<f32> {
    let mut v = Array1::zeros(n);
    for i in 0..n {
        let x = i as f32;
        let z = (x - mean) / sigma;
        v[i] = (-0.5 * z * z).exp();
    }
    let s = v.sum();
    if s > 0.0 {
        v /= s;
    }
    v
}

#[test]
fn sinkhorn_divergence_increases_with_mean_shift() {
    // Relatable e2e: two histograms are “same shape” but shifted on a line.
    // OT/Sinkhorn divergence should increase with shift magnitude.
    let n = 32;
    let cost = line_cost(n);
    let reg = 0.2;
    let max_iter = 3000;
    let tol = 1e-2;

    let a = gaussian_hist(n, 10.0, 2.0);
    let b1 = gaussian_hist(n, 11.0, 2.0);
    let b2 = gaussian_hist(n, 14.0, 2.0);

    let d1 = wass::sinkhorn_divergence_same_support(&a, &b1, &cost, reg, max_iter, tol).unwrap();
    let d2 = wass::sinkhorn_divergence_same_support(&a, &b2, &cost, reg, max_iter, tol).unwrap();

    assert!(d1 >= -1e-6);
    assert!(d2 >= -1e-6);
    assert!(
        d2 > d1,
        "expected bigger shift => bigger divergence: d1={} d2={}",
        d1,
        d2
    );
}
