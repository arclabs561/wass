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

#[test]
fn unbalanced_divergence_is_zero_on_diagonal_with_sparse_zeros() {
    // Pressure-test the log-weight handling for exact zeros.
    // If we regress to "ln(x+eps)" style behavior here, this will typically fail.
    let a = array![0.0, 0.2, 0.0, 0.3, 0.5, 0.0];
    let cost = line_cost(a.len());

    let reg = 0.2;
    let rho = 1.0;
    let max_iter = 6000;
    let tol = 1e-3;

    let d =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &a, &cost, reg, rho, max_iter, tol)
            .unwrap();

    assert!(d.abs() < 1e-4, "expected diagonal ~0 even with zeros, got {}", d);
}

