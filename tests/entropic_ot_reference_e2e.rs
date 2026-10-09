//! Entropic OT values checked against references that do not run Sinkhorn:
//! a direct minimization of the primal objective (2 x 2 balanced, 1 x 1
//! unbalanced), and the small-epsilon limit S_eps -> W_1 on a line.

use ndarray::{array, Array2};

/// Minimize a convex function on [lo, hi] by golden-section search.
fn golden_min(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    for _ in 0..200 {
        let c = b - phi * (b - a);
        let d = a + phi * (b - a);
        if f(c) < f(d) {
            b = d;
        } else {
            a = c;
        }
    }
    f(0.5 * (a + b))
}

fn xlogy_ratio(p: f64, q: f64) -> f64 {
    if p <= 0.0 {
        0.0
    } else {
        p * (p / q).ln()
    }
}

/// OT_eps(a, b) = min_{P in U(a, b)} <C, P> + eps * KL(P || a (x) b) for 2-point
/// marginals, minimized over the one free entry P_00 (Feydy et al. 2019, Eq. 1).
fn entropic_ot_2x2(a: [f64; 2], b: [f64; 2], c: [[f64; 2]; 2], eps: f64) -> f64 {
    let objective = |p00: f64| {
        let p = [[p00, a[0] - p00], [b[0] - p00, a[1] - b[0] + p00]];
        let mut v = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                v += c[i][j] * p[i][j] + eps * xlogy_ratio(p[i][j], a[i] * b[j]);
            }
        }
        v
    };
    let lo = (a[0] + b[0] - 1.0).max(0.0);
    let hi = a[0].min(b[0]);
    golden_min(lo, hi, objective)
}

#[test]
fn sinkhorn_divergence_matches_primal_entropic_ot() {
    // S_eps(a, b) = OT_eps(a, b) - (OT_eps(a, a) + OT_eps(b, b)) / 2, where
    // OT_eps includes the entropic term, not just <C, P>.
    let (a, b) = ([0.5, 0.5], [0.9, 0.1]);
    let c = [[0.0, 1.0], [1.0, 0.0]];
    let eps = 1.0;
    let expected = entropic_ot_2x2(a, b, c, eps)
        - 0.5 * (entropic_ot_2x2(a, a, c, eps) + entropic_ot_2x2(b, b, c, eps));

    let a32 = array![0.5f32, 0.5];
    let b32 = array![0.9f32, 0.1];
    let cost: Array2<f32> = array![[0.0, 1.0], [1.0, 0.0]];

    let same =
        wass::sinkhorn_divergence_same_support(&a32, &b32, &cost, eps as f32, 5000, 1e-6).unwrap();
    assert!(
        (same as f64 - expected).abs() < 1e-4,
        "same-support divergence {same}, primal reference {expected}"
    );

    let general =
        wass::sinkhorn_divergence_general(&a32, &b32, &cost, &cost, &cost, eps as f32, 5000, 1e-6)
            .unwrap();
    assert!(
        (general as f64 - expected).abs() < 1e-4,
        "general divergence {general}, primal reference {expected}"
    );
}

#[test]
fn sinkhorn_divergence_small_eps_approaches_w1_on_a_line() {
    // Three points on a line, |x_i - x_j| cost. At eps = 0.005, cost / eps
    // reaches 400, which overflows exp in f32 unless the marginal check stays
    // in log space. S_eps -> W_1 as eps -> 0, and W_1 is the area between CDFs:
    // |0.6 - 0.2| + |0.8 - 0.4| = 0.8.
    let a = array![0.6f32, 0.2, 0.2];
    let b = array![0.2f32, 0.2, 0.6];
    let mut cost = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            cost[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    let s = wass::sinkhorn_divergence_same_support(&a, &b, &cost, 0.005, 20000, 1e-4)
        .expect("small eps must converge or error, not return a wrong value");
    assert!(
        (s - 0.8).abs() < 0.02,
        "S_eps at eps = 0.005: {s}, W_1 = 0.8"
    );
}

#[test]
fn unbalanced_objective_matches_primal_minimum() {
    // Chizat et al. (2018): min_P eps KL(P || K) + rho KL(P 1 || a) + rho KL(P^T 1 || b),
    // K = exp(-C / eps), generalized KL(p || q) = p ln(p / q) - p + q. Since
    // eps KL(P || K) already contains <C, P>, there is no separate transport term.
    let (a, b, c, eps, rho) = (1.0f64, 0.5f64, 0.5f64, 0.1f64, 1.0f64);
    let kl = |p: f64, q: f64| xlogy_ratio(p, q) - p + q;
    let k = (-c / eps).exp();
    let objective = |p: f64| eps * kl(p, k) + rho * kl(p, a) + rho * kl(p, b);
    let expected = golden_min(1e-12, 2.0, objective);

    let (_plan, obj, _iters) = wass::unbalanced_sinkhorn_log_with_convergence(
        &array![a as f32],
        &array![b as f32],
        &array![[c as f32]],
        eps as f32,
        rho as f32,
        5000,
        1e-7,
    )
    .unwrap();
    assert!(
        (obj as f64 - expected).abs() < 1e-4,
        "unbalanced objective {obj}, primal minimum {expected}"
    );
}
