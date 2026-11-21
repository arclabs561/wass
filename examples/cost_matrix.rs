//! Read a balanced transport problem as JSON on stdin.
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::{error::Error, io};
use wass::sinkhorn_log_with_convergence;

#[derive(Deserialize)]
struct Problem {
    a: Vec<f32>,
    b: Vec<f32>,
    cost: Vec<Vec<f32>>,
    reg: f32,
}

impl Problem {
    fn solve(self) -> Result<(Array2<f32>, f32, usize), Box<dyn Error>> {
        let shape = (self.a.len(), self.b.len());
        if shape.0 == 0 || shape.1 == 0 {
            return Err("a and b must each contain at least one mass".into());
        }
        if self.cost.len() != shape.0 || self.cost.iter().any(|row| row.len() != shape.1) {
            return Err(format!("cost must have {} rows of {} numbers", shape.0, shape.1).into());
        }
        let cost = Array2::from_shape_vec(shape, self.cost.into_iter().flatten().collect())?;
        if !cost
            .iter()
            .chain(&self.a)
            .chain(&self.b)
            .all(|x| x.is_finite())
        {
            return Err("masses and costs must be finite".into());
        }
        Ok(sinkhorn_log_with_convergence(
            &Array1::from(self.a),
            &Array1::from(self.b),
            &cost,
            self.reg,
            2000,
            1e-6,
        )?)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let problem: Problem = serde_json::from_reader(io::stdin().lock())?;
    let (plan, cost, iterations) = problem.solve()?;
    println!("transport cost = {cost:.6} ({iterations} iterations)");
    println!("plan:\n{plan:.6}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_source_splits_across_two_targets() {
        let problem: Problem =
            serde_json::from_str(r#"{"a":[1],"b":[3,1],"cost":[[0,4]],"reg":1}"#).unwrap();
        let (plan, cost, _) = problem.solve().unwrap();
        assert!((plan[[0, 0]] - 0.75).abs() < 1e-6);
        assert!((plan[[0, 1]] - 0.25).abs() < 1e-6);
        assert!((cost - 1.).abs() < 1e-6);
    }

    #[test]
    fn ragged_rows_with_the_right_total_length_are_rejected() {
        let problem: Problem =
            serde_json::from_str(r#"{"a":[1,1],"b":[1,1],"cost":[[0],[1,2,3]],"reg":1}"#).unwrap();
        assert!(problem
            .solve()
            .unwrap_err()
            .to_string()
            .contains("2 rows of 2"));
    }
}
