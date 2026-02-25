//! Gromov-Wasserstein graph matching: aligning two metric spaces.
//!
//! GW optimal transport finds correspondences between two sets of objects
//! using only their internal pairwise distances -- no shared coordinate system needed.
//!
//! This example shows:
//! 1. Two isomorphic graphs with permuted node labels
//! 2. GW recovers the correct alignment (transport plan concentrates on the permutation)
//! 3. Non-isomorphic graphs produce diffuse plans (no perfect matching)
//!
//! Reference: Memoli (2011), "Gromov-Wasserstein Distances and Metric Measure Spaces"
//!
//! Run: cargo run -p wass --example gromov_wasserstein_graph_match

use ndarray::{array, Array1, Array2};
use wass::gromov::gromov_wasserstein;

fn main() {
    println!("=== Gromov-Wasserstein Graph Matching ===\n");

    // --- Part 1: Isomorphic graphs (permuted labels) ---
    // Graph A: path graph 0-1-2-3 (shortest-path distances)
    let c1 = array![
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0]
    ];

    // Graph B: same path graph but with nodes permuted as [2,0,3,1]
    // i.e., B[0]=A[2], B[1]=A[0], B[2]=A[3], B[3]=A[1]
    let perm = [2, 0, 3, 1]; // B[i] corresponds to A[perm[i]]
    let mut c2 = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            c2[[i, j]] = c1[[perm[i], perm[j]]];
        }
    }

    let n = 4;
    let p = Array1::from_elem(n, 1.0 / n as f64);
    let q = p.clone();

    let (plan, dist) = gromov_wasserstein(&c1, &c2, &p, &q, 0.05, 50, 100).unwrap();

    println!("--- Isomorphic path graphs (permuted labels) ---\n");
    println!("True permutation: B[i] = A[perm[i]]: {:?}", perm);
    println!("GW distance: {:.6}\n", dist);
    println!("Transport plan (rows=A, cols=B):");
    print!("{:>8}", "");
    for j in 0..n {
        print!("    B{j}  ");
    }
    println!();
    for i in 0..n {
        print!("  A{i}  ");
        for j in 0..n {
            print!("  {:.3} ", plan[[i, j]]);
        }
        // Mark the argmax
        let argmax = (0..n)
            .max_by(|&a, &b| plan[[i, a]].partial_cmp(&plan[[i, b]]).unwrap())
            .unwrap();
        print!("  <- A{i} matches B{argmax} (should be B{})", perm.iter().position(|&x| x == i).unwrap());
        println!();
    }

    // Verify matching: for each A[i], the highest-weight B[j] should satisfy perm[j] = i
    let mut correct = 0;
    for i in 0..n {
        let argmax = (0..n)
            .max_by(|&a, &b| plan[[i, a]].partial_cmp(&plan[[i, b]]).unwrap())
            .unwrap();
        if perm[argmax] == i {
            correct += 1;
        }
    }
    println!("\nMatching accuracy: {correct}/{n}");

    // --- Part 2: Non-isomorphic graphs ---
    println!("\n--- Non-isomorphic graphs ---\n");

    // Graph A: path 0-1-2-3 (same as above)
    // Graph C: star graph (node 0 connected to all others)
    let c3 = array![
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 2.0],
        [1.0, 2.0, 0.0, 2.0],
        [1.0, 2.0, 2.0, 0.0]
    ];

    let (plan2, dist2) = gromov_wasserstein(&c1, &c3, &p, &q, 0.05, 50, 100).unwrap();

    println!("Path vs Star: GW distance = {:.6}", dist2);
    println!("Transport plan:");
    print!("{:>8}", "");
    for j in 0..n {
        print!("   S{j}  ");
    }
    println!();
    for i in 0..n {
        print!("  P{i}  ");
        for j in 0..n {
            print!("  {:.3} ", plan2[[i, j]]);
        }
        println!();
    }

    println!("\nKey observations:");
    println!("  - Isomorphic graphs: plan concentrates on valid matchings");
    println!("    (path graphs have a reversal symmetry, so GW finds two valid");
    println!("     permutations and splits mass between them -- this is correct)");
    println!("  - Non-isomorphic graphs: GW distance is higher, plan is more diffuse");
    println!("  - GW only uses internal distances, not coordinates (structure-preserving)");
    println!("  - Entropic regularization (eps={eps:.2}) smooths the plan; smaller eps gives");
    println!("     sharper assignments but slower convergence", eps = 0.05);
}
