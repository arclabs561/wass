use ndarray::{Array1, Array2};

fn make_cost(xs: &[ndarray::Array1<f32>], ys: &[ndarray::Array1<f32>]) -> Array2<f32> {
    let m = xs.len();
    let n = ys.len();
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = {
                let dot = xs[i].dot(&ys[j]);
                (1.0 - dot).max(0.0).sqrt()
            };
        }
    }
    c
}

fn normalize_token(tok: &str) -> String {
    let mut out = String::new();
    for c in tok.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        }
    }
    out
}

fn base_weight(tok: &str) -> f32 {
    if tok.is_empty() {
        return 0.0;
    }
    if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
        return 0.1;
    }
    if tok.len() <= 3 {
        return 0.05;
    }
    match tok {
        "header" | "footer" | "confidential" | "distribute" | "copyright" | "internal" | "memo" => 0.05,
        "the" | "in" | "of" | "to" | "and" | "a" | "an" | "all" | "on" | "for" | "do" | "not" | "page" | "our" => 0.1,
        _ => 1.0,
    }
}

fn embed_char_ngrams_signed(text: &str, dim: usize) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(dim);
    if dim == 0 {
        return v;
    }

    const BOS: u32 = 0x110000;
    const EOS: u32 = 0x110001;
    let mut xs: Vec<u32> = Vec::with_capacity(text.chars().count() + 2);
    xs.push(BOS);
    xs.extend(text.chars().map(|c| c as u32));
    xs.push(EOS);

    fn fnv1a_u32(seq: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &x in seq {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    for n in [3usize, 2, 1] {
        if xs.len() < n {
            continue;
        }
        for i in 0..=xs.len() - n {
            let h = fnv1a_u32(&xs[i..i + n]);
            let idx = (h as usize) % dim;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
    }

    let norm = v.dot(&v).sqrt();
    if norm > 0.0 {
        v /= norm;
    }
    v
}

#[test]
fn noisy_ocr_expected_pairs_get_mass() {
    let ref_text = "The quarterly earnings showed steady growth in all sectors";
    let ocr_text = "HEADER: CONFIDENTIAL 2025 \
                    The qarterly earnigns showd stdy grwth in al sectrs \
                    FOOTER: PAGE 1 OF 10 DO NOT DISTRIBUTE";

    let ref_tokens: Vec<String> = ref_text
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    let ocr_tokens: Vec<String> = ocr_text
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    let dim = 2048;
    let ref_vecs: Vec<Array1<f32>> = ref_tokens
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();
    let ocr_vecs: Vec<Array1<f32>> = ocr_tokens
        .iter()
        .map(|t| embed_char_ngrams_signed(t, dim))
        .collect();

    let w_ref = {
        let mut w = Array1::<f32>::zeros(ref_tokens.len());
        for (i, t) in ref_tokens.iter().enumerate() {
            w[i] = base_weight(t);
        }
        let s = w.sum();
        if s > 0.0 {
            w /= s;
        }
        w
    };
    let w_ocr = {
        let mut w = Array1::<f32>::zeros(ocr_tokens.len());
        for (i, t) in ocr_tokens.iter().enumerate() {
            w[i] = base_weight(t);
        }
        let s = w.sum();
        if s > 0.0 {
            w /= s;
        }
        w
    };

    let c_ab = make_cost(&ref_vecs, &ocr_vecs);

    let reg = 0.1;
    let rho = 0.5;
    let max_iter = 1500;
    let tol = 1e-3;

    let (plan, _obj, _iters) =
        wass::unbalanced_sinkhorn_log_with_convergence(&w_ref, &w_ocr, &c_ab, reg, rho, max_iter, tol)
            .unwrap();

    fn must_match(
        ref_tokens: &[String],
        ocr_tokens: &[String],
        plan: &Array2<f32>,
        ref_tok: &str,
        expected: &str,
        min_p: f32,
    ) {
        let i = ref_tokens.iter().position(|t| t == ref_tok).unwrap();
        let mut best_j = 0usize;
        let mut best_p = 0.0f32;
        for j in 0..ocr_tokens.len() {
            let p = plan[[i, j]];
            if p > best_p {
                best_p = p;
                best_j = j;
            }
        }
        assert!(
            best_p >= min_p,
            "{ref_tok}: best_p too small: {best_p} (best={})",
            ocr_tokens[best_j]
        );
        assert_eq!(
            ocr_tokens[best_j], expected,
            "{ref_tok}: expected {expected}, got {} (p={best_p})",
            ocr_tokens[best_j]
        );
    }

    must_match(&ref_tokens, &ocr_tokens, &plan, "quarterly", "qarterly", 0.02);
    must_match(&ref_tokens, &ocr_tokens, &plan, "earnings", "earnigns", 0.02);
    must_match(&ref_tokens, &ocr_tokens, &plan, "showed", "showd", 0.02);
    must_match(&ref_tokens, &ocr_tokens, &plan, "growth", "grwth", 0.02);
    must_match(&ref_tokens, &ocr_tokens, &plan, "sectors", "sectrs", 0.02);
}

