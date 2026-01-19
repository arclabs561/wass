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
fn unbalanced_divergence_increases_with_rho_on_outlier() {
    let n = 64;
    let cost = line_cost(n);
    let reg = 0.2;
    let max_iter = 6000;
    let tol = 1e-3;

    let a = gaussian_hist(n, 20.0, 3.0);
    let mut b = a.clone() * 0.9;
    b[55] += 0.1;

    let rho_small = 0.05;
    let rho_large = 20.0;

    let d_small =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho_small, max_iter, tol).unwrap();
    let d_large =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho_large, max_iter, tol).unwrap();

    assert!(d_small <= d_large + 1e-4, "d_small={} d_large={}", d_small, d_large);
}

