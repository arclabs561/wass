//! Closed-form checks for sliced Wasserstein, Wasserstein-Fisher-Rao and
//! Gromov-Wasserstein, each derived from the definition rather than from
//! current output.

use ndarray::{array, Array1, Array2};
use wass::gromov::gromov_wasserstein;
use wass::sliced_wasserstein;
use wass::wfr::wfr_distance;

/// In one dimension every projection direction is +1 or -1, so sliced W_p
/// equals the exact 1-D W_p. x = {0, 1}, y = {0, 3}: sorted gaps 0 and 2, so
/// W_p^p = 2^p / 2 and W_1.5 = (2^1.5 / 2)^(1/1.5) = 2^(1/3).
#[test]
fn sliced_wasserstein_fractional_p_matches_1d_definition() {
    let x = array![[0.0f32], [1.0]];
    let y = array![[0.0f32], [3.0]];
    let sw = sliced_wasserstein(&x, &y, 8, 7, 1.5);
    let expected = 2f32.powf(1.0 / 3.0);
    assert!((sw - expected).abs() < 1e-5, "sw={sw} expected={expected}");
}

/// Uniform measures on {0, 1} and on {0, 0, 1, 1} are the same distribution,
/// so every W_p between them is 0. Pairing only the first min(m, n) sorted
/// values would report a positive distance.
#[test]
fn sliced_wasserstein_unequal_sizes_uses_quantile_coupling() {
    let x = array![[0.0f32], [1.0]];
    let y = array![[0.0f32], [0.0], [1.0], [1.0]];
    let sw = sliced_wasserstein(&x, &y, 8, 7, 1.0);
    assert!(sw.abs() < 1e-6, "same distribution, sw={sw}");
}

/// x = {0, 1}, y = {0, 1, 100}: the 1-D W_1 between the uniform measures is
/// the integral of |F^-1 - G^-1| over [0, 1]:
/// [0, 1/3]: |0 - 0|, [1/3, 1/2]: |0 - 1|, [1/2, 2/3]: |1 - 1|, [2/3, 1]: |1 - 100|,
/// so W_1 = 1/6 + 99/3 = 33.1667.
#[test]
fn sliced_wasserstein_unequal_sizes_matches_quantile_integral() {
    let x = array![[0.0f32], [1.0]];
    let y = array![[0.0f32], [1.0], [100.0]];
    let sw = sliced_wasserstein(&x, &y, 8, 7, 1.0);
    let expected = 1.0 / 6.0 + 99.0 / 3.0;
    assert!((sw - expected).abs() < 1e-3, "sw={sw} expected={expected}");
}

fn sq_cost_two_points(d: f32) -> Array2<f32> {
    array![[0.0, d * d], [d * d, 0.0]]
}

/// Chizat, Peyre, Schmitzer & Vialard (2018) give the WFR distance between
/// two Diracs with length scale delta in closed form:
/// WFR^2 = 4 delta^2 (m0 + m1 - 2 sqrt(m0 m1) cos(min(d / (2 delta), pi/2))).
fn wfr_dirac_closed_form(m0: f32, m1: f32, d: f32, delta: f32) -> f32 {
    let angle = (d / (2.0 * delta)).min(std::f32::consts::FRAC_PI_2);
    (4.0 * delta * delta * (m0 + m1 - 2.0 * (m0 * m1).sqrt() * angle.cos())).sqrt()
}

#[test]
fn wfr_between_diracs_matches_closed_form() {
    let cost = sq_cost_two_points(1.0);
    let a: Array1<f32> = array![1.0, 0.0];
    let b: Array1<f32> = array![0.0, 1.0];
    for &delta in &[0.5f32, 1.0] {
        let got = wfr_distance(&a, &b, &cost, delta, 0.005, 5000, 1e-6).unwrap();
        let want = wfr_dirac_closed_form(1.0, 1.0, 1.0, delta);
        assert!(
            (got - want).abs() < 0.03 * want,
            "delta={delta}: got {got}, closed form {want}"
        );
    }
}

/// For equal masses and a large length scale, WFR approaches W_2. Unit
/// masses at distance 1 have W_2 = 1.
#[test]
fn wfr_large_length_scale_approaches_w2() {
    let cost = sq_cost_two_points(1.0);
    let a: Array1<f32> = array![1.0, 0.0];
    let b: Array1<f32> = array![0.0, 1.0];
    let got = wfr_distance(&a, &b, &cost, 16.0, 0.005, 5000, 1e-6).unwrap();
    assert!((got - 1.0).abs() < 0.05, "got {got}, W_2 = 1");
}

/// The returned GW value must be the distortion of the returned plan,
/// E(P) = sum_{ijkl} (C1_ik - C2_jl)^2 P_ij P_kl.
#[test]
fn gromov_wasserstein_cost_is_distortion_of_returned_plan() {
    let c1 = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
    let c2 = array![[0.0, 1.5, 0.5], [1.5, 0.0, 1.0], [0.5, 1.0, 0.0]];
    let p = array![0.2, 0.5, 0.3];
    let q = array![0.4, 0.4, 0.2];
    let (plan, dist) = gromov_wasserstein(&c1, &c2, &p, &q, 0.05, 1, 500).unwrap();

    let mut distortion = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let diff: f64 = c1[[i, k]] - c2[[j, l]];
                    distortion += diff * diff * plan[[i, j]] * plan[[k, l]];
                }
            }
        }
    }
    assert!(
        (dist - distortion).abs() < 1e-3,
        "returned {dist}, distortion of returned plan {distortion}"
    );
}
