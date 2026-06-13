// Wasserstein barycenters, rendered in the terminal.
//
// A barycenter is the "average" of distributions under optimal transport. It is
// NOT the pointwise average: OT moves mass along the ground geometry instead of
// blending it in place. Two demos make that visible:
//
//   1. Free-support barycenter as a SHAPE MORPH. The free-support barycenter of
//      a square and a circle, swept across mixing weights, interpolates the
//      shape itself (support points move) rather than fading one into the other.
//
//   2. Fixed-support barycenter vs the naive average of two 1D humps. The OT
//      barycenter is a single hump BETWEEN the inputs; the arithmetic average
//      keeps both humps. This gap is the whole reason the crate exists.
//
// Run: `cargo run --example barycenter_morph`

use ndarray::{Array1, Array2};
use wass::barycenter::{barycenter, free_support_barycenter};

fn main() {
    shape_morph();
    println!();
    hump_vs_average();
}

// ── 1. Free-support shape morph ──────────────────────────────────────────────

fn shape_morph() {
    println!("=== Free-support barycenter: morphing a square into a circle ===\n");

    let k = 28;
    let square = polygon_outline(k, 4); // square = 4-gon outline
    let circle = polygon_outline(k, k); // many-sided polygon ~ circle
    let unit = Array1::from_elem(k, 1.0 / k as f32);

    for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let (y, _) = free_support_barycenter(
            &[
                (square.clone(), unit.clone()),
                (circle.clone(), unit.clone()),
            ],
            &[1.0 - t, t],
            k,
            0.02, // small reg -> sharp support, relies on log-domain Sinkhorn
            400,
            120,
            None,
        )
        .expect("barycenter");
        println!("  weight on circle = {t:.2}");
        render_points(&y);
        println!();
    }
    println!("  (square at 0.00, circle at 1.00; the outline bends between them)");
}

// `sides`-gon outline sampled at `n` points, inscribed in the unit circle.
fn polygon_outline(n: usize, sides: usize) -> Array2<f32> {
    Array2::from_shape_fn((n, 2), |(i, col)| {
        let frac = i as f32 / n as f32; // around the perimeter, [0, 1)
        let seg = frac * sides as f32; // which side + position along it
        let s = seg.floor() as usize % sides;
        let u = seg - seg.floor();
        // corners of the regular `sides`-gon
        let ang0 = std::f32::consts::TAU * s as f32 / sides as f32;
        let ang1 = std::f32::consts::TAU * (s + 1) as f32 / sides as f32;
        let (x0, y0) = (ang0.cos(), ang0.sin());
        let (x1, y1) = (ang1.cos(), ang1.sin());
        if col == 0 {
            x0 + u * (x1 - x0)
        } else {
            y0 + u * (y1 - y0)
        }
    })
}

fn render_points(pts: &Array2<f32>) {
    const W: usize = 33;
    const H: usize = 17;
    const LO: f32 = -1.35;
    const HI: f32 = 1.35;
    let mut grid = vec![vec![' '; W]; H];
    for i in 0..pts.nrows() {
        let (x, y) = (pts[[i, 0]], pts[[i, 1]]);
        let col = (((x - LO) / (HI - LO)) * (W - 1) as f32).round();
        let row = (((HI - y) / (HI - LO)) * (H - 1) as f32).round();
        if (0.0..W as f32).contains(&col) && (0.0..H as f32).contains(&row) {
            grid[row as usize][col as usize] = '#';
        }
    }
    for row in &grid {
        let line: String = row.iter().collect();
        println!("    {}", line.trim_end());
    }
}

// ── 2. Fixed-support barycenter vs naive average ─────────────────────────────

fn hump_vs_average() {
    println!("=== Fixed-support barycenter vs naive average of two 1D humps ===\n");

    let n = 41;
    let cost = Array2::from_shape_fn((n, n), |(i, j)| {
        let d = i as f32 - j as f32;
        d * d
    });
    let left = hump(n, 9.0, 2.5);
    let right = hump(n, 31.0, 2.5);

    let bary = barycenter(&[left.clone(), right.clone()], &cost, &[0.5, 0.5], 1.0, 500).unwrap();
    let avg = Array1::from_shape_fn(n, |i| 0.5 * (left[i] + right[i]));

    println!("  inputs: two separated humps (left + right, overlaid):");
    let both = Array1::from_shape_fn(n, |i| left[i].max(right[i]));
    render_hist(&both);
    println!("\n  naive average -- keeps BOTH humps (not a transport average):");
    render_hist(&avg);
    println!("\n  OT barycenter -- a single hump BETWEEN them:");
    render_hist(&bary);
}

fn hump(n: usize, mean: f32, std: f32) -> Array1<f32> {
    let mut v = Array1::from_shape_fn(n, |i| {
        let z = (i as f32 - mean) / std;
        (-0.5 * z * z).exp()
    });
    let s = v.sum();
    v.mapv_inplace(|x| x / s);
    v
}

// Vertical bar chart: rows are height levels (tallest at top), columns are bins.
fn render_hist(h: &Array1<f32>) {
    const ROWS: usize = 8;
    let max = h.iter().copied().fold(0.0f32, f32::max);
    let heights: Vec<usize> = h
        .iter()
        .map(|&p| {
            if max > 0.0 {
                (p / max * ROWS as f32).round() as usize
            } else {
                0
            }
        })
        .collect();
    for level in (1..=ROWS).rev() {
        let line: String = heights
            .iter()
            .map(|&hgt| if hgt >= level { '#' } else { ' ' })
            .collect();
        println!("    {}", line.trim_end());
    }
}
