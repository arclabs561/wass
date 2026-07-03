//! Rosetta correctness fixtures: wass log-domain entropic OT
//! (`sinkhorn_log`) asserted against POT (Python Optimal Transport).
//!
//! Reference values in `fixtures/rosetta/wass_sinkhorn_log.json` come from
//! `gen_wass_sinkhorn_log.py` (their provenance). wass::sinkhorn_log and POT's
//! `sinkhorn_log` solve the same entropic OT problem (dual potentials f, g via
//! log-sum-exp, plan P_ij = exp((f_i + g_j - C_ij)/reg), transport cost <C, P>).
//! The entropic OT plan is unique for a fixed (a, b, C, reg), so both converge
//! to the same plan and the same <C, P>.
//!
//! This is the deferred half of `rosetta_wass.rs`: `sinkhorn` (matrix scaling)
//! needs the cost normalized so C/reg stays in f32 range; `sinkhorn_log` is the
//! version built for reg < 0.1 * max(C). Cases B and C sit in that small-reg
//! regime (reg/max(C) ~ 0.02) where plain f32 Sinkhorn underflows.
//!
//! TIGHT tolerance class with an f32 floor: wass computes in f32, POT in f64, so
//! the gap is f32 rounding. Comparison is 1e-4, the realistic f32 Sinkhorn floor,
//! not the 1e-9 the f64 crates use.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_wass_sinkhorn_log.py`.

use ndarray::{Array1, Array2};
use serde::Deserialize;
use wass::sinkhorn_log;

const FIXTURE: &str = include_str!("fixtures/rosetta/wass_sinkhorn_log.json");

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
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
fn rosetta_sinkhorn_log_matches_pot() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    assert!(!fx.cases.is_empty(), "fixture has no cases");

    for case in &fx.cases {
        let a = to_f32_vec(&case.a);
        let b = to_f32_vec(&case.b);
        let cost = to_f32_mat(&case.cost);

        let (plan, distance) = sinkhorn_log(&a, &b, &cost, case.reg as f32, case.max_iter);

        assert_eq!(
            plan.nrows(),
            case.expected.plan.len(),
            "{}: plan rows",
            case.name
        );
        for (i, row) in case.expected.plan.iter().enumerate() {
            for (j, &want) in row.iter().enumerate() {
                close_f32(
                    plan[[i, j]],
                    want,
                    &format!("{}: plan[{i}][{j}]", case.name),
                );
            }
        }
        close_f32(
            distance,
            case.expected.distance,
            &format!("{}: distance", case.name),
        );
    }
}
