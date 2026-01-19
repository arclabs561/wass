use ndarray::{array, Array1, Array2};

#[test]
fn sinkhorn_log_with_convergence_respects_zero_mass_support() {
    // A has hard zeros at i=0 and i=2. B has hard zero at j=2.
    let a: Array1<f32> = array![0.0, 1.0, 0.0];
    let b: Array1<f32> = array![0.5, 0.5, 0.0];

    // Simple 1D line cost on indices.
    let mut cost: Array2<f32> = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            cost[[i, j]] = (i as f32 - j as f32).abs();
        }
    }

    let reg = 0.2;
    let max_iter = 2000;
    let tol = 1e-3;
    let (plan, _obj, _iters) = wass::sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol).unwrap();

    // Row sums should match a (after normalization inside the function, a sums to 1 already).
    let r0 = plan.row(0).sum();
    let r1 = plan.row(1).sum();
    let r2 = plan.row(2).sum();
    assert!(r0.abs() < 1e-6, "row0 should be 0, got {}", r0);
    assert!((r1 - 1.0).abs() < 5e-3, "row1 should be ~1, got {}", r1);
    assert!(r2.abs() < 1e-6, "row2 should be 0, got {}", r2);

    // Col sums should match b (b sums to 1 already).
    let c0 = plan.column(0).sum();
    let c1 = plan.column(1).sum();
    let c2 = plan.column(2).sum();
    assert!((c0 - 0.5).abs() < 5e-3, "col0 should be ~0.5, got {}", c0);
    assert!((c1 - 0.5).abs() < 5e-3, "col1 should be ~0.5, got {}", c1);
    assert!(c2.abs() < 1e-6, "col2 should be 0, got {}", c2);

    // And any entries in forbidden support should be ~0.
    assert!(plan[[0, 0]].abs() < 1e-6);
    assert!(plan[[0, 1]].abs() < 1e-6);
    assert!(plan[[0, 2]].abs() < 1e-6);
    assert!(plan[[2, 0]].abs() < 1e-6);
    assert!(plan[[2, 1]].abs() < 1e-6);
    assert!(plan[[2, 2]].abs() < 1e-6);
    assert!(plan[[0, 2]].abs() < 1e-6);
    assert!(plan[[1, 2]].abs() < 1e-6);
    assert!(plan[[2, 2]].abs() < 1e-6);
}

