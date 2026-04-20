use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use wass::{sinkhorn, sinkhorn_log};

fn uniform_dist(n: usize) -> Array1<f32> {
    Array1::from_elem(n, 1.0 / n as f32)
}

/// Squared Euclidean cost matrix on [0,1] grid points.
fn grid_cost(n: usize) -> Array2<f32> {
    let mut c = Array2::zeros((n, n));
    for i in 0..n {
        let xi = i as f32 / (n - 1) as f32;
        for j in 0..n {
            let xj = j as f32 / (n - 1) as f32;
            let d = xi - xj;
            c[[i, j]] = d * d;
        }
    }
    c
}

fn bench_sinkhorn(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn");

    for &n in &[100usize, 500, 1000] {
        let a = uniform_dist(n);
        let b = uniform_dist(n);
        let cost = grid_cost(n);

        group.bench_with_input(BenchmarkId::new("sinkhorn/reg=0.1", n), &n, |bench, _| {
            bench.iter(|| sinkhorn(black_box(&a), black_box(&b), black_box(&cost), 0.1, 100))
        });

        group.bench_with_input(
            BenchmarkId::new("sinkhorn_log/reg=0.1", n),
            &n,
            |bench, _| {
                bench.iter(|| sinkhorn_log(black_box(&a), black_box(&b), black_box(&cost), 0.1, 50))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sinkhorn);
criterion_main!(benches);
