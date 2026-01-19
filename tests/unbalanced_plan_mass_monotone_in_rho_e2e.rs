use ndarray::{array, Array1, Array2};

fn line_cost(m: usize, n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

#[test]
fn unbalanced_transport_matches_marginals_better_when_rho_increases() {
    // In unbalanced OT, increasing rho penalizes marginal mismatch more,
    // so the plan's marginals should move closer to the requested (a,b).
    //
    // This is a “pipeline-level” invariant: rho should tighten the constraints.

    // Source has mass concentrated near the start.
    let a: Array1<f32> = array![0.5, 0.5, 0.0, 0.0];
    // Target has similar mass plus an outlier at the end.
    let b: Array1<f32> = array![0.45, 0.45, 0.0, 0.10];

    let cost = line_cost(a.len(), b.len());
    let reg = 0.1;
    let max_iter = 4000;
    let tol = 1e-3;

    let rho_small = 0.5;
    let rho_big = 10.0;

    let (p_small, _obj_small, _iters_small) =
        wass::unbalanced_sinkhorn_log_with_convergence(&a, &b, &cost, reg, rho_small, max_iter, tol)
            .unwrap();
    let (p_big, _obj_big, _iters_big) =
        wass::unbalanced_sinkhorn_log_with_convergence(&a, &b, &cost, reg, rho_big, max_iter, tol)
            .unwrap();

    let row_small = p_small.sum_axis(ndarray::Axis(1));
    let col_small = p_small.sum_axis(ndarray::Axis(0));
    let row_big = p_big.sum_axis(ndarray::Axis(1));
    let col_big = p_big.sum_axis(ndarray::Axis(0));

    let err_small = row_small
        .iter()
        .zip(a.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        .max(
            col_small
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f32>(),
        );

    let err_big = row_big
        .iter()
        .zip(a.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        .max(
            col_big
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f32>(),
        );

    assert!(
        err_big <= err_small + 1e-6,
        "expected rho↑ to tighten marginal match: rho_small err={} rho_big err={}",
        err_small,
        err_big
    );
}

