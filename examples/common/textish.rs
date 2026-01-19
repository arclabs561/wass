#![allow(dead_code)]

use ndarray::Array1;
use std::collections::BTreeMap;

/// Normalize a token for alignment:
/// - lowercase
/// - keep only Unicode alphanumerics
/// - drop everything else (punctuation, hyphens, etc.)
pub fn normalize_token(tok: &str) -> String {
    let mut out = String::new();
    for c in tok.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        }
    }
    out
}

pub fn is_number(tok: &str) -> bool {
    !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit())
}

pub fn is_stopword(tok: &str) -> bool {
    matches!(
        tok,
        "the" | "in" | "of" | "to" | "and" | "a" | "an" | "on" | "for" | "with" | "by" | "at"
            | "from" | "this" | "that" | "is" | "are" | "do" | "not" | "page" | "our" | "all"
    )
}

pub fn is_boilerplate(tok: &str) -> bool {
    matches!(
        tok,
        "subscribe" | "newsletter" | "cookies" | "cookie" | "privacy" | "policy" | "terms"
            | "menu" | "home" | "contact" | "share" | "twitter" | "linkedin" | "copyright"
            | "header" | "footer" | "confidential" | "distribute" | "internal" | "memo"
    )
}

/// Base weight (before TF/IDF) for a token.
///
/// Opinionated: short tokens (<=3) are almost always junk in this kind of pipeline,
/// so we downweight them aggressively.
pub fn base_weight(tok: &str) -> f32 {
    if tok.is_empty() {
        return 0.0;
    }
    if is_number(tok) {
        // Numbers tend to be ambiguous without context; keep but downweight.
        return 0.1;
    }
    if tok.len() <= 3 {
        return 0.05;
    }
    if is_boilerplate(tok) {
        return 0.05;
    }
    if is_stopword(tok) {
        return 0.1;
    }
    1.0
}

/// A small, dependency-free signed hashing embedder for char n-grams.
///
/// Why this instead of the previous `c[i]*31^2 + ...`?
/// - That scheme collides a lot (especially for short tokens).
/// - Collisions create *nonsense alignments* that look plausible numerically.
///
/// This embedder:
/// - adds boundary markers
/// - uses FNV-1a on Unicode scalars
/// - uses a sign bit (±1 updates) to reduce collision bias
pub fn embed_char_ngrams_signed(text: &str, dim: usize) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(dim);
    if dim == 0 {
        return v;
    }

    // Add boundary markers so prefixes/suffixes matter.
    // Use distinct scalars unlikely to appear in normal text.
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

    // Mix multiple n-gram orders: unigram/bigram/trigram.
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

pub fn cosine_dist(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    (1.0 - dot).max(0.0).sqrt()
}

/// Collapse a token stream into a stable bag-of-words.
///
/// Returns sorted `(token, count)` pairs.
pub fn bag_of_tokens(tokens: &[String]) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for t in tokens {
        if t.is_empty() {
            continue;
        }
        if t.len() < 2 {
            continue;
        }
        *counts.entry(t.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

/// Compute TF×(smoothed)IDF style weights for two bags.
///
/// \( \text{idf}(t) = \log\frac{1+N}{1+\mathrm{df}(t)} + 1 \), with \(N=2\).
pub fn weights_tfidf_2docs(
    a: &[(String, usize)],
    b: &[(String, usize)],
) -> (Array1<f32>, Array1<f32>) {
    let mut df: BTreeMap<&str, usize> = BTreeMap::new();
    for (t, _) in a {
        *df.entry(t.as_str()).or_insert(0) += 1;
    }
    for (t, _) in b {
        *df.entry(t.as_str()).or_insert(0) += 1;
    }

    fn idf(df: usize) -> f32 {
        let n = 2.0f32;
        ((1.0 + n) / (1.0 + df as f32)).ln() + 1.0
    }

    let mut w_a = Array1::<f32>::zeros(a.len());
    for (i, (t, c)) in a.iter().enumerate() {
        let df_t = *df.get(t.as_str()).unwrap_or(&1);
        w_a[i] = base_weight(t) * (*c as f32) * idf(df_t);
    }

    let mut w_b = Array1::<f32>::zeros(b.len());
    for (i, (t, c)) in b.iter().enumerate() {
        let df_t = *df.get(t.as_str()).unwrap_or(&1);
        w_b[i] = base_weight(t) * (*c as f32) * idf(df_t);
    }

    let s_a = w_a.sum();
    if s_a > 0.0 {
        w_a /= s_a;
    }
    let s_b = w_b.sum();
    if s_b > 0.0 {
        w_b /= s_b;
    }
    (w_a, w_b)
}

