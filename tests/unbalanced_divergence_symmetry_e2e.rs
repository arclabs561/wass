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
fn unbalanced_sinkhorn_divergence_same_support_is_symmetric_even_with_mass_mismatch() {
    let a = array![0.2, 0.3, 0.5]; // mass 1.0
    let b = array![0.1, 0.1, 0.3]; // mass 0.5
    let cost = line_cost(a.len());

    let reg = 0.2;
    let rho = 1.0;
    let max_iter = 4000;
    let tol = 1e-3;

    let ab =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho, max_iter, tol)
            .unwrap();
    let ba =
        wass::unbalanced_sinkhorn_divergence_same_support(&b, &a, &cost, reg, rho, max_iter, tol)
            .unwrap();

    assert!(
        (ab - ba).abs() < 1e-4,
        "expected symmetry: ab={} ba={}",
        ab,
        ba
    );
}
