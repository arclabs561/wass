use ndarray::{Array1, Array2};

fn line_cost(n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

fn main() {
    // End-to-end demo: unbalanced OT is about *mass mismatch*.
    //
    // Here `b` has half the total mass of `a`. We compute the unbalanced Sinkhorn divergence
    // with two different rho values. Smaller rho => cheaper mass creation/destruction.
    // Avoid exact zeros here to keep KL penalties finite and the demo interpretable.
    let a = Array1::from_vec(vec![0.05, 0.2, 0.5, 0.25]); // mass 1.0
    let b = Array1::from_vec(vec![0.025, 0.1, 0.25, 0.125]); // mass 0.5 (scaled down)

    let cost = line_cost(a.len());
    let reg = 0.1;
    let max_iter = 4000;
    let tol = 1e-3;

    println!("mass(a) = {:.6}", a.sum());
    println!("mass(b) = {:.6}", b.sum());
    println!("reg(eps) = {reg}  tol = {tol}  max_iter = {max_iter}");
    println!();

    for &rho in &[0.05, 0.5, 2.0, 10.0] {
        let (plan, obj_ab, iters) =
            wass::unbalanced_sinkhorn_log_with_convergence(&a, &b, &cost, reg, rho, max_iter, tol).unwrap();

        let row_mass: f32 = plan.sum_axis(ndarray::Axis(1)).sum();
        let col_mass: f32 = plan.sum_axis(ndarray::Axis(0)).sum();
        let plan_mass: f32 = plan.sum();

        let div =
            wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho, max_iter, tol).unwrap();

        println!(
            "rho={rho:>6.2}  iters={iters:>4}  obj={obj_ab:.6}  div={div:.12}"
        );
        println!(
            "             plan_mass={plan_mass:.6}  row_mass={row_mass:.6}  col_mass={col_mass:.6}"
        );
    }
}

