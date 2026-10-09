# /// script
# requires-python = ">=3.10"
# dependencies = ["pot", "numpy"]
# ///
"""Rosetta fixture generator for wass de-biased Sinkhorn divergence on the same
support (`sinkhorn_divergence_same_support`).

Provenance for wass_sinkhorn_divergence.json. wass::sinkhorn_divergence_same_support
computes, for two histograms on the same n-bin support with an n x n cost C:

    S = max(0, OT(a, b) - 0.5 * (OT(a, a) + OT(b, b)))

where each OT(x, y) is the entropic OT value <C, P> + reg * KL(P || x (x) y) at
the optimal plan P (Feydy et al. 2019, Eq. 1). The entropy term is what makes
S a divergence; <C, P> alone does not.

The reference takes each plan from POT's log-domain solver
(ot.sinkhorn(..., method="sinkhorn_log")) and evaluates the same objective, so
it differs from wass only in f32-vs-f64 arithmetic.

Note on tolerance: S is a *cancellation* of three same-magnitude OT terms, so its
absolute error is bounded by the per-term f32 floor (~1e-4), not by |S|. The Rust
test therefore compares S with an absolute-ish 1e-4 floor, the crate's TIGHT-f32
class. wass normalizes a, b internally; they are stored already normalized.

Regenerate: uv run tests/fixtures/rosetta/gen_wass_sinkhorn_divergence.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import ot

SEED = 11
rng = np.random.default_rng(SEED)

# Reachable marginal threshold in f64 (1e-12 is not reached at small reg in a
# sane budget). Puts every OT term at its true fixed point so the de-biased S is
# grounded, not an artifact of partial convergence.
STOP_THR = 1e-9
MAX_ITER = 20000


def symmetric_cost(n):
    """Symmetric zero-diagonal cost from a seeded point cloud (a metric on the
    shared support)."""
    pts = rng.normal(0.0, 1.0, size=(n, 2))
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = pts[i] - pts[j]
            c[i, j] = float(np.sqrt(d @ d))
    return c


def ot_cost(a, b, cost, reg):
    plan = ot.sinkhorn(
        a, b, cost, reg, method="sinkhorn_log", numItermax=MAX_ITER, stopThr=STOP_THR
    )
    marg_err = max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max())
    assert marg_err < 1e-6, f"OT term not converged (marg_err={marg_err:.2e})"
    product = np.outer(a, b)
    positive = plan > 0
    kl = float(np.sum(plan[positive] * np.log(plan[positive] / product[positive])))
    return float(np.sum(plan * cost)) + reg * kl


def make_case(name, a, b, cost, reg):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / a.sum()
    b = b / b.sum()
    ot_ab = ot_cost(a, b, cost, reg)
    ot_aa = ot_cost(a, a, cost, reg)
    ot_bb = ot_cost(b, b, cost, reg)
    divergence = max(0.0, ot_ab - 0.5 * (ot_aa + ot_bb))
    return {
        "name": name,
        "reg": reg,
        # wass-side convergence controls; tight enough that the f32 plan is near
        # the true fixed point, so OT_eps matches POT's f64 value to the f32 floor.
        "max_iter": MAX_ITER,
        "tol": 1e-6,
        "a": a.tolist(),
        "b": b.tolist(),
        "cost": cost.tolist(),
        "expected": {
            "ot_ab": ot_ab,
            "ot_aa": ot_aa,
            "ot_bb": ot_bb,
            "divergence": divergence,
        },
    }


cases = []

# Case A: n=4, uniform a vs skewed b, symmetric cost, moderate reg.
cost4 = symmetric_cost(4)
cases.append(
    make_case(
        "uniform_vs_skewed",
        np.full(4, 1.0),
        rng.dirichlet(np.full(4, 2.0)),
        cost4,
        reg=0.5,
    )
)

# Case B: n=5, two distinct skewed distributions, symmetric cost, smaller reg.
cost5 = symmetric_cost(5)
cases.append(
    make_case(
        "skewed_vs_skewed",
        rng.dirichlet(np.full(5, 2.0)),
        rng.dirichlet(np.full(5, 2.0)),
        cost5,
        reg=0.2,
    )
)

# Case C: n=3, a == b. De-biasing property: S(a, a) = 0 exactly on both sides.
cost3 = symmetric_cost(3)
a3 = rng.dirichlet(np.full(3, 2.0))
cases.append(make_case("identical_zero_divergence", a3, a3, cost3, reg=0.5))

fixture = {
    "provenance": {
        "generator": "gen_wass_sinkhorn_divergence.py",
        "library": "POT (Python Optimal Transport)",
        "pot_version": ot.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "Divergence de-biased from entropic OT values <C,P> + reg*KL(P||a(x)b) (Feydy 2019), matching wass; f32 cancellation floor is ~1e-4.",
    },
    "cases": cases,
}

out = Path(__file__).parent / "wass_sinkhorn_divergence.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for c in cases:
    e = c["expected"]
    print(
        f"{c['name']:26s} reg={c['reg']} ot_ab={e['ot_ab']:.8f} "
        f"ot_aa={e['ot_aa']:.8f} ot_bb={e['ot_bb']:.8f} S={e['divergence']:.10f}"
    )
print(f"wrote {out}")
