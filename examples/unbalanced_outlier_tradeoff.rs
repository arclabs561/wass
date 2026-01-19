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

fn gaussian_hist(n: usize, mean: f32, sigma: f32) -> Array1<f32> {
    let mut v = Array1::zeros(n);
    for i in 0..n {
        let x = i as f32;
        let z = (x - mean) / sigma;
        v[i] = (-0.5 * z * z).exp();
    }
    let s = v.sum();
    if s > 0.0 {
        v /= s;
    }
    v
}

fn main() {
    // Relatable story:
    // - `a` is a clean distribution (a “signal”)
    // - `b` is mostly the same signal but with a far-away outlier mass.
    //
    // With small rho (cheap mass change), UOT can “delete” the outlier cheaply.
    // With large rho (expensive mass change), it is forced to pay transport / mismatch.
    let n = 64;
    let cost = line_cost(n);
    let reg = 0.2;
    let max_iter = 6000;
    let tol = 1e-3;

    let a = gaussian_hist(n, 20.0, 3.0);

    // b = mostly a, plus an outlier spike far away.
    let mut b = a.clone() * 0.9;
    b[55] += 0.1; // outlier

    println!("mass(a)={:.6} mass(b)={:.6}", a.sum(), b.sum());
    println!("reg(eps)={reg} tol={tol} max_iter={max_iter}");
    println!();
    println!("rho     div(a,b)   plan_mass");

    for &rho in &[0.05, 0.2, 1.0, 5.0, 20.0] {
        let (plan, _obj, _iters) =
            wass::unbalanced_sinkhorn_log_with_convergence(&a, &b, &cost, reg, rho, max_iter, tol).unwrap();
        let div = wass::unbalanced_sinkhorn_divergence_same_support(&a, &b, &cost, reg, rho, max_iter, tol).unwrap();
        println!("{rho:>5.2}  {div:>9.6}  {mass:>9.6}", rho=rho, div=div, mass=plan.sum());
    }
}

