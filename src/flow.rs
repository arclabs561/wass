//! Flow / drift primitives.
//!
//! In this stack, we treat "drift" as **infinitesimal transport**:
//! a vector field describes the local velocity needed to move mass/points.
//!
//! This module intentionally stays small and composable; higher-level training
//! objectives belong in L2 (`fynch`) and domain wiring belongs in L5+.

use ndarray::{Array1, ArrayView1};

/// A vector field representing a continuous drift in a (latent) space.
pub trait VectorField {
    /// Evaluate the velocity at point `x` and time `t`.
    fn velocity(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64>;
}

/// Computes the drift between two points.
///
/// \[
/// v = \frac{\text{target} - \text{source}}{\Delta t}
/// \]
///
/// Panics if `source.len() != target.len()` or if `dt == 0`.
pub fn flow_drift(source: &[f64], target: &[f64], dt: f64) -> Vec<f64> {
    assert_eq!(source.len(), target.len(), "dimension mismatch");
    assert!(dt != 0.0, "dt must be non-zero");
    source
        .iter()
        .zip(target)
        .map(|(&s, &t)| (t - s) / dt)
        .collect()
}

