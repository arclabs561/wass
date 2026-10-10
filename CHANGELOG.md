# Changelog

## [Unreleased]

## [0.2.3] - 2026-10-09

### Changed

- Declare `rust-version = "1.75"`.

### Fixed

- `sinkhorn_divergence`, `sinkhorn_divergence_same_support` and
  `sinkhorn_divergence_general` debias with the entropic OT value
  `<C, P> + eps * KL(P || a (x) b)` (Feydy et al. 2019) instead of `<C, P>`.
  Divergence values differ from 0.2.2 for the same inputs; on a 2 x 2 case
  the result is 0.1913 where 0.2.2 returned 0.2291.
- `sinkhorn_with_convergence` and `sinkhorn_log_with_convergence` return
  `Error::Domain` when the marginal error is NaN, and the log-domain
  convergence check combines exponents so it no longer computes `inf * 0`
  once `C / eps` exceeds about 88. Previously the NaN was dropped and the solver reported
  convergence (three points at `eps = 0.005` gave 2.9e-40 instead of about
  0.8).
- `unbalanced_sinkhorn_log_with_convergence` no longer adds `<C, P>` to an
  objective whose `eps * KL(P || K)` term already contains it; the returned
  objective is lower than in 0.2.2.
- `sliced_wasserstein` handles unequal sample sizes by integrating over the
  1-D quantile coupling instead of pairing the first `min(m, n)` sorted
  projections, and handles fractional `p` (it previously used `powi(p as i32)`).
- `wfr_distance` follows Chizat et al. (2018): unit weight on the KL terms
  and a `4 * delta^2` scaling, returning `2 * rho * sqrt(divergence)`, so
  large `rho` approaches W2 as documented. `rho` is the length scale `delta`;
  transport beyond `pi * delta` is never used. Values differ from 0.2.2.
- `gromov_wasserstein` returns the distortion of the returned plan rather than
  the linearized cost paired with it. Its docs describe the method as mirror
  descent (Peyre's entropic iteration), not Frank-Wolfe.
- Comparators use `total_cmp`.

## [0.2.2] - 2026-06-13

### Added

- `barycenter` module: `barycenter` / `barycenter_with_convergence` compute the fixed-support entropic Wasserstein barycenter via log-domain iterative Bregman projections (correct at small `reg`, where the linear-domain form silently degrades to a histogram average), and `free_support_barycenter` computes the free-support barycenter (support points move) by alternating Sinkhorn with a barycentric-projection position update. Tests include the 1D Gaussian/Bures closed-form oracle and a rotation/translation equivariance property test. New `barycenter_morph` example.

### Changed

- Documentation polish; no API changes.

