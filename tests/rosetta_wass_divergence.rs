//! Rosetta correctness fixtures: wass de-biased Sinkhorn divergence on the same
//! support (`sinkhorn_divergence_same_support`) asserted against POT.
//!
//! Reference values in `fixtures/rosetta/wass_sinkhorn_divergence.json` come from
//! `gen_wass_sinkhorn_divergence.py` (their provenance). wass computes
//!
//!   S = max(0, OT(a, b) - 0.5 * (OT(a, a) + OT(b, b)))
//!
//! where each OT(x, y) is the *transport cost* <C, P> of the log-domain entropic
//! OT plan (NOT the full regularized objective). The reference composes the same
//! quantity from POT's log-domain plans reduced to <C, P>, so it is
//! definitionally identical to wass and differs only in f32-vs-f64 arithmetic.
//! This is the deferred divergence half of `rosetta_wass.rs`.
//!
//! TIGHT tolerance class with an f32 floor. S is a cancellation of three
//! same-magnitude OT terms, so its absolute error is bounded by the per-term f32
//! floor (~1e-4), not by |S|; the comparison uses 1e-4, not the f64 crates' 1e-9.
//! The `identical_zero_divergence` case (a == b) exercises the de-biasing
//! property S(a, a) = 0, which both sides produce exactly.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_wass_sinkhorn_divergence.py`.

use ndarray::{Array1, Array2};
use serde::Deserialize;
use wass::sinkhorn_divergence_same_support;

const FIXTURE: &str = include_str!("fixtures/rosetta/wass_sinkhorn_divergence.json");

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    reg: f64,
    max_iter: usize,
    tol: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    cost: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    divergence: f64,
}

fn close_f32(got: f32, want: f64, label: &str) {
    // f32 floor: the de-biased S cancels three same-magnitude OT terms, so the
    // per-term f32 rounding (~1e-4) is the realistic floor.
    let tol = 1e-4 * (1.0 + want.abs());
    let diff = (got as f64 - want).abs();
    assert!(
        diff <= tol,
        "{label}: wass={got} pot={want} diff={diff} tol={tol}"
    );
}

fn to_f32_vec(xs: &[f64]) -> Array1<f32> {
    Array1::from(xs.iter().map(|&x| x as f32).collect::<Vec<f32>>())
}

fn to_f32_mat(rows: &[Vec<f64>]) -> Array2<f32> {
    let d = rows[0].len();
    let mut m = Array2::zeros((rows.len(), d));
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            m[[i, j]] = v as f32;
        }
    }
    m
}

#[test]
fn rosetta_sinkhorn_divergence_matches_pot() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    assert!(!fx.cases.is_empty(), "fixture has no cases");

    for case in &fx.cases {
        let a = to_f32_vec(&case.a);
        let b = to_f32_vec(&case.b);
        let cost = to_f32_mat(&case.cost);

        let s = sinkhorn_divergence_same_support(
            &a,
            &b,
            &cost,
            case.reg as f32,
            case.max_iter,
            case.tol as f32,
        )
        .unwrap_or_else(|e| {
            panic!(
                "{}: sinkhorn_divergence_same_support failed: {e:?}",
                case.name
            )
        });

        close_f32(
            s,
            case.expected.divergence,
            &format!("{}: divergence", case.name),
        );
    }
}
