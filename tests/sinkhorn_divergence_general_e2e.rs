use ndarray::{array, Array1, Array2};

fn euclidean_cost_1d(points: &Array2<f32>) -> Array2<f32> {
    // points: (n,1)
    let n = points.nrows();
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            c[[i, j]] = (points[[i, 0]] - points[[j, 0]]).abs();
        }
    }
    c
}

fn euclidean_cost_cross_1d(x: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    let m = x.nrows();
    let n = y.nrows();
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = (x[[i, 0]] - y[[j, 0]]).abs();
        }
    }
    c
}

#[test]
fn sinkhorn_divergence_general_is_symmetric_and_zero_on_diagonal() {
    // Real use-case: distributions on different supports (e.g. point clouds / different bins).
    // We provide cost_ab, cost_aa, cost_bb explicitly.
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![[0.5], [1.5]];

    let a = Array1::from_vec(vec![0.2, 0.3, 0.5]);
    let b = Array1::from_vec(vec![0.6, 0.4]);

    let cost_aa = euclidean_cost_1d(&x);
    let cost_bb = euclidean_cost_1d(&y);
    let cost_ab = euclidean_cost_cross_1d(&x, &y);
    let cost_ba = euclidean_cost_cross_1d(&y, &x);

    let reg = 0.1;
    let max_iter = 2000;
    let tol = 1e-2;

    let daa =
        wass::sinkhorn_divergence_general(&a, &a, &cost_aa, &cost_aa, &cost_aa, reg, max_iter, tol)
            .unwrap();
    let dbb =
        wass::sinkhorn_divergence_general(&b, &b, &cost_bb, &cost_bb, &cost_bb, reg, max_iter, tol)
            .unwrap();
    assert!(daa.abs() < 1e-5, "daa={}", daa);
    assert!(dbb.abs() < 1e-5, "dbb={}", dbb);

    let dab =
        wass::sinkhorn_divergence_general(&a, &b, &cost_ab, &cost_aa, &cost_bb, reg, max_iter, tol)
            .unwrap();
    let dba =
        wass::sinkhorn_divergence_general(&b, &a, &cost_ba, &cost_bb, &cost_aa, reg, max_iter, tol)
            .unwrap();
    assert!((dab - dba).abs() < 1e-4, "dab={} dba={}", dab, dba);
    assert!(dab >= -1e-6);
}
