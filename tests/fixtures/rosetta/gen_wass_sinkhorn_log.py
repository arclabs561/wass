# /// script
# requires-python = ">=3.10"
# dependencies = ["pot", "numpy"]
# ///
"""Rosetta fixture generator for wass log-domain Sinkhorn (`sinkhorn_log`).

Provenance for wass_sinkhorn_log.json. Reference plans come from POT's
log-domain solver, ot.sinkhorn(..., method="sinkhorn_log"), which solves the
same entropic OT problem as wass::sinkhorn_log: kernel exp(-C/reg), dual
potentials f, g updated by log-sum-exp, plan P_ij = exp((f_i + g_j - C_ij)/reg),
transport cost <C, P>. The entropic OT plan is unique for a fixed (a, b, C, reg),
so both solvers converge to the same plan and the same <C, P>.

Why a separate fixture from wass_sinkhorn.json: the plain matrix-scaling
`sinkhorn` needs the cost normalized so C/reg stays in f32's representable range.
`sinkhorn_log` is the version that stays stable when reg < 0.1 * max(C) (its own
docs say to prefer it there), so this fixture deliberately includes an
un-normalized-cost / small-reg case that would underflow plain f32 Sinkhorn.

TIGHT tolerance class with an f32 floor: wass computes in f32, POT in f64, so the
gap is f32 rounding. The Rust test compares within 1e-4 (the realistic f32
Sinkhorn floor), not the 1e-9 the f64 crates use. wass::sinkhorn_log normalizes
its inputs internally, so a, b are stored already normalized to sum 1.

Regenerate: uv run tests/fixtures/rosetta/gen_wass_sinkhorn_log.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import ot

SEED = 7
rng = np.random.default_rng(SEED)

# Reachable marginal threshold in f64. Small-reg entropic OT converges slowly, so
# 1e-12 is not reached in a sane iteration budget; 1e-9 is, and it puts the
# reference plan at the true fixed point (both solvers then agree independent of
# the exact iteration count).
STOP_THR = 1e-9


def dist_matrix(pa, pb):
    m, n = len(pa), len(pb)
    c = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = pa[i] - pb[j]
            c[i, j] = float(np.sqrt(d @ d))
    return c


def sq_dist_matrix(pa, pb):
    m, n = len(pa), len(pb)
    c = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = pa[i] - pb[j]
            c[i, j] = float(d @ d)
    return c


def make_case(name, a, b, cost, reg, max_iter):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / a.sum()
    b = b / b.sum()
    plan = ot.sinkhorn(
        a, b, cost, reg, method="sinkhorn_log", numItermax=max_iter, stopThr=STOP_THR
    )
    distance = float(np.sum(plan * cost))
    # Marginal error of the reference plan; must be tiny so the plan is at the
    # true entropic-OT fixed point, not a partial-convergence artifact.
    marg_err = float(
        max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max())
    )
    return {
        "name": name,
        "reg": reg,
        "max_iter": max_iter,
        "a": a.tolist(),
        "b": b.tolist(),
        "cost": cost.tolist(),
        "expected": {"plan": plan.tolist(), "distance": distance},
        "_marg_err": marg_err,
    }


cases = []

# Case A: square (4x4), uniform marginals, symmetric distance cost, moderate reg.
# The moderate-reg / symmetric-cost baseline; the small-reg log-domain regime is
# cases B and C.
pts_a = rng.normal(0.0, 1.0, size=(4, 2))
cost_a = dist_matrix(pts_a, pts_a)  # symmetric, zero diagonal
cases.append(
    make_case(
        "uniform_symmetric",
        np.full(4, 1.0),
        np.full(4, 1.0),
        cost_a,
        reg=0.2,
        max_iter=20000,
    )
)

# Case B: rectangular (5x4), skewed marginals, UN-normalized squared-euclidean
# cost, small reg -> reg < 0.1 * max(C). This is the log-domain regime: plain f32
# Sinkhorn underflows here, sinkhorn_log does not.
pa = rng.normal(0.0, 1.0, size=(5, 2))
pb = rng.normal(0.7, 1.0, size=(4, 2))
cost_b = sq_dist_matrix(pa, pb)  # NOT normalized; max entry can be several units
cases.append(
    make_case(
        "skewed_small_reg",
        rng.dirichlet(np.full(5, 2.0)),
        rng.dirichlet(np.full(4, 2.0)),
        cost_b,
        reg=0.1,
        max_iter=20000,
    )
)

# Case C: square (6x6), skewed marginals, symmetric distance cost, small reg.
# reg/max(C) ~ 0.02: solidly in the log-domain regime (plain f32 Sinkhorn
# underflows here) while still converging in a sane iteration budget.
pts_c = rng.normal(0.0, 1.5, size=(6, 2))
cost_c = dist_matrix(pts_c, pts_c)
cases.append(
    make_case(
        "square_skewed_small_reg",
        rng.dirichlet(np.full(6, 1.5)),
        rng.dirichlet(np.full(6, 1.5)),
        cost_c,
        reg=0.1,
        max_iter=20000,
    )
)

fixture = {
    "provenance": {
        "generator": "gen_wass_sinkhorn_log.py",
        "library": "POT (Python Optimal Transport)",
        "pot_version": ot.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "wass::sinkhorn_log is f32; comparison floor is 1e-4. Case B/C sit in the small-reg regime the log-domain solver exists for.",
    },
    "cases": cases,
}

for c in cases:
    cost = np.asarray(c["cost"])
    reg_over_maxc = c["reg"] / cost.max()
    marg_err = c.pop("_marg_err")
    assert marg_err < 1e-6, f"{c['name']}: reference not converged (marg_err={marg_err:.2e})"
    print(
        f"{c['name']:24s} shape={cost.shape} reg={c['reg']} "
        f"reg/maxC={reg_over_maxc:.4f} marg_err={marg_err:.2e} "
        f"distance={c['expected']['distance']:.10f}"
    )

out = Path(__file__).parent / "wass_sinkhorn_log.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print(f"wrote {out}")
