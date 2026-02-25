use ndarray::{array, Array1, Array2};

/// Build a simple 1D ground cost matrix on shared bins: C[i,j] = |i-j|.
fn line_cost(n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

#[test]
fn sinkhorn_divergence_monotone_under_interpolation_same_support() {
    // Real use-case: we have histograms on shared support and we want a smooth distance.
    // If we interpolate b_t = (1-t)a + t b, divergence to a should increase with t.
    let a = array![0.0, 0.2, 0.5, 0.3];
    let b = array![0.3, 0.5, 0.2, 0.0];
    let cost = line_cost(a.len());

    let reg = 0.1;
    let max_iter = 2000;
    let tol = 1e-2;

    let d0 = wass::sinkhorn_divergence_same_support(&a, &a, &cost, reg, max_iter, tol).unwrap();
    assert!(d0.abs() < 1e-5);

    let mut prev = d0;
    for step in 1..=5 {
        let t = step as f32 / 5.0;
        let bt: Array1<f32> = (&a * (1.0 - t)) + (&b * t);
        let d = wass::sinkhorn_divergence_same_support(&a, &bt, &cost, reg, max_iter, tol).unwrap();
        assert!(
            d >= prev - 1e-4,
            "expected nondecreasing divergence along interpolation: prev={} d={} at t={}",
            prev,
            d,
            t
        );
        prev = d;
    }
}
