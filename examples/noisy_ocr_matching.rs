//! Noisy OCR / Scraping Alignment Demo
//!
//! Demonstrates using Unbalanced Optimal Transport to align a "clean" reference text
//! with a "noisy" OCR scan that contains typos and irrelevant headers/footers.
//!
//! # The Problem
//!
//! You have a clean database entry: "Quarterly earnings showed steady growth."
//! You scraped a PDF: "HEADER: CONFIDENTIAL Qarterly earnngs showd stdy grwth FOOTER: PG 1"
//!
//! Standard distance (Levenshtein) fails because of the massive header/footer insertions.
//! Balanced OT fails because the masses don't match (header/footer must map to something).
//!
//! # The Solution
//!
//! Unbalanced Sinkhorn Divergence allows:
//! 1. **Robustness to Typos**: handled by the ground cost (embedding distance).
//! 2. **Robustness to Outliers**: handled by the marginal relaxation (rho).
//!    The "HEADER" tokens are destroyed rather than transported.

mod common;

use ndarray::{Array1, Array2};
use wass::{unbalanced_sinkhorn_divergence_general, unbalanced_sinkhorn_log_with_convergence};

use common::textish::{base_weight, cosine_dist, embed_char_ngrams_signed, normalize_token};

fn top_k_by<T: Copy>(xs: &[(T, f32)], k: usize) -> Vec<(T, f32)> {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Data
    let ref_text = "The quarterly earnings showed steady growth in all sectors";
    let ocr_text = "HEADER: CONFIDENTIAL 2025 \
                    The qarterly earnigns showd stdy grwth in al sectrs \
                    FOOTER: PAGE 1 OF 10 DO NOT DISTRIBUTE";

    // Split into tokens
    let ref_tokens_raw: Vec<&str> = ref_text.split_whitespace().collect();
    let ocr_tokens_raw: Vec<&str> = ocr_text.split_whitespace().collect();

    let ref_tokens: Vec<String> = ref_tokens_raw.iter().map(|t| normalize_token(t)).collect();
    let ocr_tokens: Vec<String> = ocr_tokens_raw.iter().map(|t| normalize_token(t)).collect();

    println!("Reference ({} tokens): \"{}\"", ref_tokens.len(), ref_text);
    println!("Noisy OCR ({} tokens): \"{}\"", ocr_tokens.len(), ocr_text);
    println!();

    // 2. Embed Tokens (signed n-gram hashing)
    let dim = 2048;
    let ref_vecs: Vec<Array1<f32>> = ref_tokens
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();
    let ocr_vecs: Vec<Array1<f32>> = ocr_tokens
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();

    // 3. Build Weights (downweight boilerplate/stopwords/short tokens/numbers)
    fn weights_for(tokens: &[String]) -> Array1<f32> {
        let mut w = Array1::zeros(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            w[i] = base_weight(t);
        }
        let s = w.sum();
        if s > 0.0 {
            w /= s;
        }
        w
    }

    let ref_weights = weights_for(&ref_tokens);
    let ocr_weights = weights_for(&ocr_tokens);

    // 4. Build Cost Matrices
    let make_cost = |xs: &[Array1<f32>], ys: &[Array1<f32>]| {
        let m = xs.len();
        let n = ys.len();
        let mut c = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                c[[i, j]] = cosine_dist(&xs[i], &ys[j]);
            }
        }
        c
    };

    let c_ab = make_cost(&ref_vecs, &ocr_vecs);
    let c_aa = make_cost(&ref_vecs, &ref_vecs);
    let c_bb = make_cost(&ocr_vecs, &ocr_vecs);

    // 5. Run UOT with different rhos
    // Low rho = ignores outliers (HEADER/FOOTER).
    // High rho = forces matching everything.

    // Slightly larger epsilon makes typos cheaper to match than to delete.
    let reg = 0.1;
    let max_iter = 1000;
    let tol = 1e-3;

    println!("Aligning with Unbalanced Sinkhorn (epsilon={})", reg);
    println!("{:<6} {:<10} {:<30}", "Rho", "Divergence", "Interpretation");
    println!("{}", "-".repeat(60));

    for &rho in &[0.5, 10.0] {
        let div = unbalanced_sinkhorn_divergence_general(
            &ref_weights,
            &ocr_weights,
            &c_ab,
            &c_aa,
            &c_bb,
            reg,
            rho,
            max_iter,
            tol,
        )?;

        let interpretation = if rho < 1.0 {
            "Ignores outliers (robust)"
        } else {
            "Forces full match (sensitive)"
        };

        println!("{:<6.1} {:<10.4} {}", rho, div, interpretation);

        // Show the plan to prove it works
        let (plan, _, _) = unbalanced_sinkhorn_log_with_convergence(
            &ref_weights,
            &ocr_weights,
            &c_ab,
            reg,
            rho,
            max_iter,
            tol,
        )?;

        // Diagnostics we care about in practice:
        // - which tokens matched credibly (low dist, nontrivial mass)
        // - which source mass was deleted
        // - which target mass was unused (i.e. "header/footer" got ignored)
        let p_min = 0.02f32;
        let dist_max = 0.70f32;

        let mut good: Vec<(usize, usize, f32)> = Vec::new();
        let mut deleted_src: Vec<(usize, f32)> = Vec::new();
        let mut unused_tgt: Vec<(usize, f32)> = Vec::new();

        for i in 0..ref_tokens.len() {
            let row_sum = plan.row(i).sum();
            let deleted = (ref_weights[i] - row_sum).max(0.0);
            if deleted > 0.0 {
                deleted_src.push((i, deleted));
            }

            let mut best_j = 0usize;
            let mut best_p = 0.0f32;
            for j in 0..ocr_tokens.len() {
                let p = plan[[i, j]];
                if p > best_p {
                    best_p = p;
                    best_j = j;
                }
            }
            let dist = c_ab[[i, best_j]];
            if best_p >= p_min && dist <= dist_max {
                good.push((i, best_j, best_p));
            }
        }

        for j in 0..ocr_tokens.len() {
            let col_sum = plan.column(j).sum();
            let unused = (ocr_weights[j] - col_sum).max(0.0);
            if unused > 0.0 {
                unused_tgt.push((j, unused));
            }
        }

        let plan_mass = plan.sum();
        let deleted_total: f32 = deleted_src.iter().map(|(_, x)| *x).sum();
        let unused_total: f32 = unused_tgt.iter().map(|(_, x)| *x).sum();

        println!("  plan_mass={plan_mass:.3}  deleted_src={deleted_total:.3}  unused_tgt={unused_total:.3}");
        println!();

        println!("  credible matches (p>={p_min:.2}, dist<={dist_max:.2}):");
        if good.is_empty() {
            println!("    (none)");
        } else {
            for (i, j, p) in good.iter() {
                println!(
                    "    {:15} -> {:15}  p={:.2}  dist={:.2}  w_src={:.3} w_tgt={:.3}",
                    ref_tokens[*i],
                    ocr_tokens[*j],
                    *p,
                    c_ab[[*i, *j]],
                    ref_weights[*i],
                    ocr_weights[*j]
                );
            }
        }

        println!();
        println!("  most deleted source tokens:");
        for (i, m) in top_k_by(
            &deleted_src
                .iter()
                .map(|(i, m)| (*i, *m))
                .collect::<Vec<_>>(),
            6,
        ) {
            println!(
                "    {:15} deleted={:.3}  w_src={:.3}",
                ref_tokens[i], m, ref_weights[i]
            );
        }

        println!();
        println!("  most unused target tokens:");
        for (j, m) in top_k_by(
            &unused_tgt.iter().map(|(j, m)| (*j, *m)).collect::<Vec<_>>(),
            8,
        ) {
            println!(
                "    {:15} unused={:.3}  w_tgt={:.3}",
                ocr_tokens[j], m, ocr_weights[j]
            );
        }

        println!();
    }

    Ok(())
}
