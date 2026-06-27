# /// script
# requires-python = ">=3.10"
# dependencies = ["pot", "numpy"]
# ///
"""Rosetta fixture generator for wass entropic OT (Sinkhorn).

Provenance for wass_sinkhorn.json. The reference transport plan comes from POT
(Python Optimal Transport), ot.sinkhorn, which uses the same entropic model as
wass: kernel K = exp(-C/reg), scaling iterations u = a/(Kv), v = b/(K^T u), plan
P = diag(u) K diag(v), transport cost <C, P>.

TIGHT tolerance class with an f32 floor: wass computes in f32 (its OT layer is
f32 by design), POT in f64. Both converge to the same unique entropic OT plan
for a fixed reg, so the only gap is f32 rounding. The Rust test compares within
1e-4, the realistic f32 Sinkhorn floor, not the 1e-9 used for the f64 crates.

Deferred: sinkhorn_divergence (its de-biasing convention -- whether OT_eps is the
transport cost or the full regularized objective -- needs a closer read before a
POT mapping is safe).

Regenerate: uv run tests/fixtures/rosetta/gen_wass.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import ot

SEED = 0
rng = np.random.default_rng(SEED)

m, n = 5, 4
a = rng.dirichlet(np.full(m, 2.0))
b = rng.dirichlet(np.full(n, 2.0))

# Squared-euclidean ground cost between two small point clouds.
pa = rng.normal(0.0, 1.0, size=(m, 2))
pb = rng.normal(0.6, 1.0, size=(n, 2))
cost = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        d = pa[i] - pb[j]
        cost[i, j] = float(d @ d)
# Normalize cost by its max (standard OT practice). This keeps C/reg in f32's
# representable range so wass's plain (non-log-domain) f32 Sinkhorn does not
# underflow against POT's f64 kernel. Small reg with un-normalized cost is the
# regime that needs wass::sinkhorn_log, which is a separate (deferred) check.
cost = cost / cost.max()

reg = 0.1
max_iter = 2000

plan = ot.sinkhorn(a, b, cost, reg, numItermax=max_iter, stopThr=1e-12)
distance = float(np.sum(plan * cost))

fixture = {
    "provenance": {
        "generator": "gen_wass.py",
        "library": "POT (Python Optimal Transport)",
        "pot_version": ot.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "wass is f32; comparison floor is 1e-4, not the f64 crates' 1e-9.",
    },
    "reg": reg,
    "max_iter": max_iter,
    "a": a.tolist(),
    "b": b.tolist(),
    "cost": cost.tolist(),
    "expected": {
        "plan": plan.tolist(),
        "distance": distance,
    },
}

out = Path(__file__).parent / "wass_sinkhorn.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print(f"distance {distance:.10f}")
print(f"plan row sums {plan.sum(axis=1)}")
print(f"wrote {out}")
