//! Rosetta correctness fixtures: wass entropic OT (Sinkhorn) asserted against
//! POT (Python Optimal Transport).
//!
//! Reference values in `fixtures/rosetta/wass_sinkhorn.json` come from
//! `gen_wass.py` (their provenance). wass and POT share the entropic model
//! (kernel exp(-C/reg), u/v scaling iterations, plan diag(u) K diag(v), cost
//! <C, P>), so both converge to the same unique plan for a fixed reg.
//!
//! TIGHT tolerance class with an f32 floor: wass's OT layer computes in f32, POT
//! in f64, so the gap is f32 rounding. Comparison is 1e-4, the realistic f32
//! Sinkhorn floor, not the 1e-9 the f64 crates use.
//!
//! The cost matrix is normalized by its max (standard OT practice). That keeps
//! C/reg inside f32's representable range; without it, an early run showed
//! plain f32 Sinkhorn diverging ~1% from POT because exp(-C/reg) underflowed to
//! zero in f32 while staying nonzero in f64. Small reg with un-normalized cost
//! is the regime wass::sinkhorn_log exists for, which is a separate (deferred)
//! check. Also deferred: sinkhorn_divergence (its de-biasing convention needs a
//! closer read before a POT mapping is safe).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_wass.py`.

use ndarray::{Array1, Array2};
use serde::Deserialize;
use wass::sinkhorn;

const FIXTURE: &str = include_str!("fixtures/rosetta/wass_sinkhorn.json");

#[derive(Deserialize)]
struct Fixture {
    reg: f64,
    max_iter: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    cost: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    plan: Vec<Vec<f64>>,
    distance: f64,
}

fn close_f32(got: f32, want: f64, label: &str) {
    // f32 Sinkhorn floor: 1e-4 relative, much looser than the f64 crates' 1e-9.
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
fn rosetta_sinkhorn_matches_pot() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let a = to_f32_vec(&fx.a);
    let b = to_f32_vec(&fx.b);
    let cost = to_f32_mat(&fx.cost);

    let (plan, distance) = sinkhorn(&a, &b, &cost, fx.reg as f32, fx.max_iter);

    assert_eq!(plan.nrows(), fx.expected.plan.len(), "plan rows");
    for (i, row) in fx.expected.plan.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            close_f32(plan[[i, j]], want, &format!("plan[{i}][{j}]"));
        }
    }
    close_f32(distance, fx.expected.distance, "distance");
}
