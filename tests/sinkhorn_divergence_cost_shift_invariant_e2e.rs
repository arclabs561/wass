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
fn sinkhorn_divergence_is_invariant_to_constant_cost_shift_same_support() {
    // Debiased Sinkhorn divergence should be invariant to adding a constant to all entries of C:
    // OTε shifts by constant * transported_mass (which is 1 after normalization), and the
    // self-terms shift the same way, so the debiasing cancels.
    let a = array![0.2, 0.3, 0.5];
    let b = array![0.5, 0.4, 0.1];
    let cost = line_cost(a.len());

    let reg = 0.2;
    let max_iter = 4000;
    let tol = 1e-3;

    let d1 = wass::sinkhorn_divergence_same_support(&a, &b, &cost, reg, max_iter, tol).unwrap();

    let shift = 7.0f32;
    let cost2 = cost.mapv(|x| x + shift);
    let d2 = wass::sinkhorn_divergence_same_support(&a, &b, &cost2, reg, max_iter, tol).unwrap();

    assert!(
        (d1 - d2).abs() < 1e-4,
        "expected cost-shift invariance: d1={} d2={}",
        d1,
        d2
    );
}

