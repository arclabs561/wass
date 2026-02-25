//! Document Alignment Demo (Realistic-ish)
//!
//! This demo is meant to feel like a real pipeline step:
//! “Do these two scraped texts describe the same thing?” despite:
//! - boilerplate (subscribe / cookie banners / nav),
//! - small OCR/typo noise,
//! - different phrasing.
//!
//! We avoid heavyweight models here and use:
//! - **token normalization**
//! - **cheap char n-gram embeddings** (hashing trick)
//! - **unbalanced OT** to delete junk mass instead of forcing a match.

mod common;

use ndarray::{Array1, Array2};
use wass::{unbalanced_sinkhorn_divergence_general, unbalanced_sinkhorn_log_with_convergence};

use common::textish::{
    bag_of_tokens, cosine_dist, embed_char_ngrams_signed, normalize_token, weights_tfidf_2docs,
};

fn top_k_indices_by(xs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = xs.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let doc_a = r#"
        Breaking: Quarterly earnings show steady growth
        The quarterly earnings showed steady growth in all sectors, with revenue up 12%.
        Subscribe to our newsletter. Cookies policy. Share on Twitter.
    "#;

    let doc_b = r#"
        CONFIDENTIAL - INTERNAL MEMO
        Qarterly earnigns showd stdy grwth across sectrs; revnue +12 percent.
        MENU HOME CONTACT PRIVACY POLICY TERMS
    "#;

    let toks_a_raw: Vec<String> = doc_a
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    let toks_b_raw: Vec<String> = doc_b
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    let bow_a = bag_of_tokens(&toks_a_raw);
    let bow_b = bag_of_tokens(&toks_b_raw);

    let toks_a: Vec<String> = bow_a.iter().map(|(t, _)| t.clone()).collect();
    let toks_b: Vec<String> = bow_b.iter().map(|(t, _)| t.clone()).collect();

    let (w_a, w_b) = weights_tfidf_2docs(&bow_a, &bow_b);

    let dim = 2048;
    let vec_a: Vec<Array1<f32>> = toks_a
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();
    let vec_b: Vec<Array1<f32>> = toks_b
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();

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

    let c_ab = make_cost(&vec_a, &vec_b);
    let c_aa = make_cost(&vec_a, &vec_a);
    let c_bb = make_cost(&vec_b, &vec_b);

    println!("Doc A unique tokens: {}", toks_a.len());
    println!("Doc B unique tokens: {}", toks_b.len());
    println!();

    let reg = 0.1;
    let max_iter = 1200;
    let tol = 1e-3;

    for &rho in &[0.5, 10.0] {
        println!("--- rho={rho} ---");
        let div = unbalanced_sinkhorn_divergence_general(
            &w_a, &w_b, &c_ab, &c_aa, &c_bb, reg, rho, max_iter, tol,
        )?;
        println!("div={:.4}", div);

        // For interpretability we print a few strongest token alignments from the *plan*.
        let (plan, _obj, _iters) =
            unbalanced_sinkhorn_log_with_convergence(&w_a, &w_b, &c_ab, reg, rho, max_iter, tol)?;

        let plan_mass = plan.sum();
        println!("plan_mass={plan_mass:.3}");
        println!();

        // Split alignments into "credible" vs "suspect forced matches".
        // This is opinionated: if the cost is high, we don't pretend it's a good alignment.
        let p_min = 0.01f32;
        let dist_good = 0.70f32;
        let dist_bad = 0.85f32;

        let mut good_edges: Vec<(usize, usize, f32)> = Vec::new();
        let mut suspect_edges: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..toks_a.len() {
            for j in 0..toks_b.len() {
                let p = plan[[i, j]];
                if p < p_min {
                    continue;
                }
                let d = c_ab[[i, j]];
                if d <= dist_good {
                    good_edges.push((i, j, p));
                } else if d >= dist_bad {
                    suspect_edges.push((i, j, p));
                }
            }
        }
        good_edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        suspect_edges.sort_by(|a, b| b.2.total_cmp(&a.2));

        println!("credible alignments (p>={p_min:.2}, dist<={dist_good:.2}):");
        for (i, j, p) in good_edges.iter().take(12) {
            println!(
                "  {:12} -> {:12}  p={:.3}  dist={:.2}  w_a={:.3} w_b={:.3}",
                toks_a[*i],
                toks_b[*j],
                *p,
                c_ab[[*i, *j]],
                w_a[*i],
                w_b[*j]
            );
        }

        println!();
        println!("suspect forced matches (p>={p_min:.2}, dist>={dist_bad:.2}):");
        if suspect_edges.is_empty() {
            println!("  (none)");
        } else {
            for (i, j, p) in suspect_edges.iter().take(8) {
                println!(
                    "  {:12} -> {:12}  p={:.3}  dist={:.2}  w_a={:.3} w_b={:.3}",
                    toks_a[*i],
                    toks_b[*j],
                    *p,
                    c_ab[[*i, *j]],
                    w_a[*i],
                    w_b[*j]
                );
            }
        }

        // Deleted / unused mass summaries.
        let mut deleted_src = vec![0.0f32; toks_a.len()];
        for i in 0..toks_a.len() {
            let row_sum = plan.row(i).sum();
            deleted_src[i] = (w_a[i] - row_sum).max(0.0);
        }
        let mut unused_tgt = vec![0.0f32; toks_b.len()];
        for j in 0..toks_b.len() {
            let col_sum = plan.column(j).sum();
            unused_tgt[j] = (w_b[j] - col_sum).max(0.0);
        }
        let deleted_total: f32 = deleted_src.iter().sum();
        let unused_total: f32 = unused_tgt.iter().sum();

        println!();
        println!("deleted_src_total={deleted_total:.3}  unused_tgt_total={unused_total:.3}");
        println!("top deleted source tokens:");
        for (i, m) in top_k_indices_by(&deleted_src, 6) {
            if m <= 0.0 {
                break;
            }
            println!("  {:12} deleted={:.3}  w_a={:.3}", toks_a[i], m, w_a[i]);
        }
        println!("top unused target tokens:");
        for (j, m) in top_k_indices_by(&unused_tgt, 8) {
            if m <= 0.0 {
                break;
            }
            println!("  {:12} unused={:.3}  w_b={:.3}", toks_b[j], m, w_b[j]);
        }

        println!();
    }

    Ok(())
}
