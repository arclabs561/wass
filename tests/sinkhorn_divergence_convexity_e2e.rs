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

fn normalize(mut x: Array1<f32>) -> Array1<f32> {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let s = x.sum();
    if s > 0.0 {
        x /= s;
    } else {
        x[0] = 1.0;
    }
    x
}

#[test]
fn sinkhorn_divergence_is_convex_in_first_argument_same_support() {
    // Sinkhorn divergence is convex in each argument (Feydy et al. 2018 / Genevay et al. 2018).
    // This is a good “definition-level” invariant that catches many subtle mistakes.
    let n = 16;
    let cost = line_cost(n);
    let reg = 0.2;
    let max_iter = 3000;
    let tol = 1e-2;

    // Deterministic pseudo-random-ish weights (no rng dependency in tests).
    let a1 = normalize(Array1::from_iter(
        (0..n).map(|i| ((i * 37 + 11) % 101) as f32),
    ));
    let a2 = normalize(Array1::from_iter(
        (0..n).map(|i| ((i * 19 + 7) % 97) as f32),
    ));
    let b = normalize(Array1::from_iter(
        (0..n).map(|i| ((i * 13 + 5) % 89) as f32),
    ));

    let lambda = 0.3f32;
    let a_mix = &a1 * lambda + &a2 * (1.0 - lambda);

    let s1 = wass::sinkhorn_divergence_same_support(&a1, &b, &cost, reg, max_iter, tol).unwrap();
    let s2 = wass::sinkhorn_divergence_same_support(&a2, &b, &cost, reg, max_iter, tol).unwrap();
    let sm = wass::sinkhorn_divergence_same_support(&a_mix, &b, &cost, reg, max_iter, tol).unwrap();

    let rhs = lambda * s1 + (1.0 - lambda) * s2;

    // Numerical reality: we allow a small slack.
    assert!(
        sm <= rhs + 5e-3,
        "convexity violated: S(mix,b)={} vs rhs={} (s1={}, s2={})",
        sm,
        rhs,
        s1,
        s2
    );
}
