# Changelog

## [Unreleased]

### Added

- `barycenter` module: `barycenter` / `barycenter_with_convergence` compute the fixed-support entropic Wasserstein barycenter via log-domain iterative Bregman projections (correct at small `reg`, where the linear-domain form silently degrades to a histogram average), and `free_support_barycenter` computes the free-support barycenter (support points move) by alternating Sinkhorn with a barycentric-projection position update. Tests include the 1D Gaussian/Bures closed-form oracle and a rotation/translation equivariance property test. New `barycenter_morph` example.

### Changed

- Documentation polish; no API changes.

