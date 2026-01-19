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
    let ref_vecs: Vec<Array1<f32>> = ref_tokens.iter().map(|t| embed_char_ngrams_signed(t, dim)).collect();
    let ocr_vecs: Vec<Array1<f32>> = ocr_tokens.iter().map(|t| embed_char_ngrams_signed(t, dim)).collect();

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
            &ref_weights, &ocr_weights, &c_ab, &c_aa, &c_bb,
            reg, rho, max_iter, tol
        )?;

        let interpretation = if rho < 1.0 {
            "Ignores outliers (robust)"
        } else {
            "Forces full match (sensitive)"
        };

        println!("{:<6.1} {:<10.4} {}", rho, div, interpretation);

        // Show the plan to prove it works
        let (plan, _, _) = unbalanced_sinkhorn_log_with_convergence(
            &ref_weights, &ocr_weights, &c_ab, reg, rho, max_iter, tol
        )?;

        if rho < 1.0 {
            println!("\n  Robust Alignment (rho={}):", rho);
            for (i, ref_tok) in ref_tokens.iter().enumerate() {
                // Find best match in OCR
                let mut best_j = 0;
                let mut max_p = 0.0;
                for j in 0..ocr_tokens.len() {
                    if plan[[i, j]] > max_p {
                        max_p = plan[[i, j]];
                        best_j = j;
                    }
                }
                
                if max_p > 0.01 {
                    println!(
                        "    {:15} -> {:15} (p={:.2}, dist={:.2}, w={:.3})",
                        ref_tok,
                        ocr_tokens[best_j],
                        max_p,
                        c_ab[[i, best_j]],
                        ref_weights[i]
                    );
                } else {
                    println!(
                        "    {:15} -> [Deleted/Lost] (w={:.3})",
                        ref_tok,
                        ref_weights[i]
                    );
                }
            }
            
            // Check what happened to the first token (usually "header"/boilerplate)
            let j0 = 0;
            let used = plan.column(j0).sum();
            println!("  token[0]={:?} used mass: {:.3} (original: {:.3})", ocr_tokens[j0], used, ocr_weights[j0]);
            println!();
        }
    }

    Ok(())
}
