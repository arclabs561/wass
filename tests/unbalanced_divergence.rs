//! Unbalanced Sinkhorn divergence against a primal reference (Séjourné et al. 2019).
use ndarray::array;

/// Séjourné et al. 2019 (arXiv:1910.12958) S_eps = OT_ab - OT_aa/2 - OT_bb/2 + eps/2 (m_a - m_b)^2,
/// with OT_eps = <C,P> + eps KL(P|a⊗b) + rho KL(P1|a) + rho KL(P^T1|b).
/// Reference value computed by an f64 primal solver (numpy, 20k log-domain iterations): 0.27975.
/// GeomLoss's dual formula (rho + eps/2) <a, e^{-f_aa/rho} - e^{-f_ba/rho}> + ... already
/// equals S_eps; wass adds `0.5 * eps * (m_a - m_b)^2` a second time.
#[test]
fn unbalanced_divergence_matches_primal_reference() {
    let a = array![0.5f32, 0.3, 0.2];
    let b = array![0.9f32, 0.6, 0.5];
    let c = array![[0.0f32, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]];
    let s = wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &c, 0.5, 1.0, 5000, 1e-6)
        .unwrap();
    let g = wass::unbalanced_sinkhorn_divergence_general(&a, &b, &c, &c, &c, 0.5, 1.0, 5000, 1e-6)
        .unwrap();
    assert!(
        (s - 0.27975).abs() < 2e-3,
        "same_support S = {s}, expected 0.27975"
    );
    assert!(
        (g - 0.27975).abs() < 2e-3,
        "general S = {g}, expected 0.27975"
    );
}

/// `wfr_distance` documents an error when the solver does not converge; one
/// iteration with `tol = 1e-9` cannot reach the tolerance.
#[test]
fn wfr_distance_errors_when_not_converged() {
    let a = array![0.5f32, 0.3, 0.2];
    let b = array![0.1f32, 0.3, 0.9];
    let c = array![[0.0f32, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]];
    let r = wass::wfr::wfr_distance(&a, &b, &c, 1.0, 0.01, 1, 1e-9);
    assert!(
        matches!(r, Err(wass::Error::SinkhornNotConverged(1))),
        "got {r:?}"
    );
    // Enough iterations still converge and return a distance.
    assert!(wass::wfr::wfr_distance(&a, &b, &c, 1.0, 0.1, 5000, 1e-4).is_ok());
}

/// Like the balanced divergences, the unbalanced ones report non-convergence
/// instead of returning a value from unconverged potentials.
#[test]
fn unbalanced_divergences_error_when_not_converged() {
    let a = array![0.5f32, 0.3, 0.2];
    let b = array![0.9f32, 0.6, 0.5];
    let c = array![[0.0f32, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]];
    let s = wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &c, 0.5, 1.0, 1, 1e-9);
    assert!(
        matches!(s, Err(wass::Error::SinkhornNotConverged(1))),
        "got {s:?}"
    );
    let g = wass::unbalanced_sinkhorn_divergence_general(&a, &b, &c, &c, &c, 0.5, 1.0, 1, 1e-9);
    assert!(
        matches!(g, Err(wass::Error::SinkhornNotConverged(1))),
        "got {g:?}"
    );
}
