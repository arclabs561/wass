use ndarray::{array, Array1, Array2};

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
fn unbalanced_sinkhorn_handles_mass_mismatch() {
    // Same support histograms, but different total mass.
    // Balanced OT in this crate normalizes internally; unbalanced should not.
    let a = Array1::from_vec(vec![0.0, 0.2, 0.5, 0.3]); // mass 1.0
    let b = Array1::from_vec(vec![0.0, 0.1, 0.25, 0.15]); // mass 0.5 (scaled down)
    let cost = line_cost(a.len());

    let reg = 0.1;
    let max_iter = 4000;
    let tol = 1e-3;

    // With small rho (cheap mass change), divergence should be small.
    let rho_small = 0.05;
    let d_small =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho_small, max_iter, tol).unwrap();

    // With large rho (expensive mass change), divergence should be larger.
    let rho_large = 10.0;
    let d_large =
        wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho_large, max_iter, tol).unwrap();

    assert!(d_small <= d_large + 1e-3, "d_small={} d_large={}", d_small, d_large);
    assert!(d_small >= -1e-6);
    assert!(d_large >= -1e-6);
}

#[test]
fn unbalanced_sinkhorn_divergence_zero_on_diagonal() {
    let a = array![0.0, 0.2, 0.5, 0.3];
    let cost = line_cost(a.len());
    let reg = 0.1;
    let rho = 1.0;
    let d = wass::unbalanced_sinkhorn_divergence_same_support(&a, &a, &cost, reg, rho, 4000, 1e-3).unwrap();
    assert!(d.abs() < 1e-4, "d={}", d);
}

