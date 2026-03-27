//! Word Mover's Distance for Document Comparison
//!
//! Word Mover's Distance (Kusner et al., 2015) measures document similarity as
//! the minimum-cost transport of word embeddings from one document to another.
//!
//! Pipeline:
//! 1. Build a synthetic vocabulary codebook (symproj)
//! 2. Tokenize documents, look up per-word embeddings
//! 3. Compute pairwise L2 cost matrix between word embeddings
//! 4. Uniform marginals (1/n_words per document)
//! 5. Sinkhorn regularized OT -> Word Mover's Distance
//!
//! Also demonstrates chunk-level comparison:
//! - Split documents into sentence chunks (slabs)
//! - Mean-pool each chunk's embeddings into a single vector
//! - Compute OT between chunk-level representations

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use slabs::{Chunker, SentenceChunker};
use symproj::Codebook;
use wass::sinkhorn_log;

// --- Documents ---

const DOCUMENTS: &[&str] = &[
    // 0: machine learning
    "neural networks learn representations from data through gradient descent optimization",
    // 1: machine learning (related to 0)
    "deep learning models optimize parameters using backpropagation on training data",
    // 2: cooking
    "chop the onions and garlic then sauté in olive oil until golden brown",
    // 3: cooking (related to 2)
    "dice the vegetables and fry them in butter until they are soft and caramelized",
    // 4: astronomy
    "the telescope captured images of distant galaxies billions of light years away",
    // 5: astronomy (related to 4)
    "astronomers observed star formation in a nebula using infrared spectroscopy",
    // 6: finance
    "quarterly earnings exceeded expectations driven by strong consumer spending",
    // 7: finance (related to 6)
    "revenue growth outpaced forecasts due to increased retail demand this quarter",
];

// --- Helpers ---

/// Build a synthetic codebook: assign each unique word a random embedding.
/// Words that share a domain get correlated embeddings via a shared offset.
fn build_codebook(
    documents: &[&str],
    dim: usize,
    rng: &mut StdRng,
) -> (Codebook, HashMap<String, u32>) {
    let mut word_to_id: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0u32;

    for doc in documents {
        for word in doc.split_whitespace() {
            let w = word.to_lowercase();
            if !word_to_id.contains_key(&w) {
                word_to_id.insert(w, next_id);
                next_id += 1;
            }
        }
    }

    let vocab_size = next_id as usize;
    let mut matrix = vec![0.0f32; vocab_size * dim];

    for (&_, &id) in word_to_id.iter() {
        let offset = id as usize * dim;
        for d in 0..dim {
            matrix[offset + d] = rng.random_range(-1.0f32..1.0f32);
        }
    }

    // Nudge semantically related words closer by adding a shared domain vector.
    // This makes the WMD results more interpretable.
    let domain_groups: &[&[&str]] = &[
        &[
            "neural",
            "networks",
            "learn",
            "representations",
            "gradient",
            "descent",
            "deep",
            "learning",
            "models",
            "optimize",
            "parameters",
            "backpropagation",
            "optimization",
            "training",
            "data",
        ],
        &[
            "chop",
            "onions",
            "garlic",
            "sauté",
            "olive",
            "oil",
            "golden",
            "brown",
            "dice",
            "vegetables",
            "fry",
            "butter",
            "soft",
            "caramelized",
        ],
        &[
            "telescope",
            "galaxies",
            "light",
            "years",
            "astronomers",
            "star",
            "formation",
            "nebula",
            "infrared",
            "spectroscopy",
            "captured",
            "images",
            "observed",
            "distant",
            "billions",
        ],
        &[
            "quarterly",
            "earnings",
            "expectations",
            "consumer",
            "spending",
            "revenue",
            "growth",
            "forecasts",
            "retail",
            "demand",
            "quarter",
            "exceeded",
            "outpaced",
            "increased",
            "driven",
            "strong",
        ],
    ];

    for group in domain_groups {
        // Shared direction for this domain.
        let mut domain_vec = vec![0.0f32; dim];
        for d in 0..dim {
            domain_vec[d] = rng.random_range(-0.5f32..0.5f32);
        }
        for &word in *group {
            if let Some(&id) = word_to_id.get(word) {
                let offset = id as usize * dim;
                for d in 0..dim {
                    matrix[offset + d] += domain_vec[d];
                }
            }
        }
    }

    let codebook = Codebook::new(matrix, dim).expect("valid codebook");
    (codebook, word_to_id)
}

/// Tokenize a document and return the token IDs present in the vocabulary.
fn tokenize(doc: &str, word_to_id: &HashMap<String, u32>) -> Vec<u32> {
    doc.split_whitespace()
        .filter_map(|w| word_to_id.get(&w.to_lowercase()).copied())
        .collect()
}

/// Compute L2 cost matrix between two sets of embeddings.
fn l2_cost_matrix(embeddings_a: &[Vec<f32>], embeddings_b: &[Vec<f32>]) -> Array2<f32> {
    let m = embeddings_a.len();
    let n = embeddings_b.len();
    let mut cost = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let dist_sq: f32 = embeddings_a[i]
                .iter()
                .zip(embeddings_b[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            cost[[i, j]] = dist_sq.sqrt();
        }
    }
    cost
}

/// Compute Word Mover's Distance between two documents.
fn word_movers_distance(
    doc_a: &str,
    doc_b: &str,
    codebook: &Codebook,
    word_to_id: &HashMap<String, u32>,
    reg: f32,
    max_iter: usize,
) -> f32 {
    let ids_a = tokenize(doc_a, word_to_id);
    let ids_b = tokenize(doc_b, word_to_id);

    if ids_a.is_empty() || ids_b.is_empty() {
        return f32::INFINITY;
    }

    let emb_a = codebook.encode_sequence_ids(&ids_a);
    let emb_b = codebook.encode_sequence_ids(&ids_b);

    let cost = l2_cost_matrix(&emb_a, &emb_b);

    // Uniform marginals.
    let a = Array1::from_elem(emb_a.len(), 1.0 / emb_a.len() as f32);
    let b = Array1::from_elem(emb_b.len(), 1.0 / emb_b.len() as f32);

    let (_plan, distance) = sinkhorn_log(&a, &b, &cost, reg, max_iter);
    distance
}

// --- Chunk-level comparison ---

/// Mean-pool a set of embeddings into a single vector.
fn mean_pool(embeddings: &[Vec<f32>], dim: usize) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![0.0; dim];
    }
    let mut out = vec![0.0f32; dim];
    for emb in embeddings {
        for (o, &e) in out.iter_mut().zip(emb.iter()) {
            *o += e;
        }
    }
    let n = embeddings.len() as f32;
    for o in out.iter_mut() {
        *o /= n;
    }
    out
}

/// Compute chunk-level OT distance between two documents.
///
/// Each document is split into sentence chunks (via slabs), each chunk is
/// mean-pooled into a single vector, then Sinkhorn computes the transport
/// cost between the two sets of chunk embeddings.
fn chunk_level_distance(
    doc_a: &str,
    doc_b: &str,
    codebook: &Codebook,
    word_to_id: &HashMap<String, u32>,
    reg: f32,
    max_iter: usize,
) -> f32 {
    let chunker = SentenceChunker::new(1);

    let chunks_a = chunker.chunk(doc_a);
    let chunks_b = chunker.chunk(doc_b);

    if chunks_a.is_empty() || chunks_b.is_empty() {
        return f32::INFINITY;
    }

    let dim = codebook.dim();

    let chunk_embs_a: Vec<Vec<f32>> = chunks_a
        .iter()
        .map(|slab| {
            let ids = tokenize(&slab.text, word_to_id);
            let embs = codebook.encode_sequence_ids(&ids);
            mean_pool(&embs, dim)
        })
        .collect();

    let chunk_embs_b: Vec<Vec<f32>> = chunks_b
        .iter()
        .map(|slab| {
            let ids = tokenize(&slab.text, word_to_id);
            let embs = codebook.encode_sequence_ids(&ids);
            mean_pool(&embs, dim)
        })
        .collect();

    let cost = l2_cost_matrix(&chunk_embs_a, &chunk_embs_b);

    let a = Array1::from_elem(chunk_embs_a.len(), 1.0 / chunk_embs_a.len() as f32);
    let b = Array1::from_elem(chunk_embs_b.len(), 1.0 / chunk_embs_b.len() as f32);

    let (_plan, distance) = sinkhorn_log(&a, &b, &cost, reg, max_iter);
    distance
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 32;
    let reg = 0.05;
    let max_iter = 200;

    let (codebook, word_to_id) = build_codebook(DOCUMENTS, dim, &mut rng);

    println!("=== Word Mover's Distance (word-level) ===");
    println!();
    println!("Documents:");
    for (i, doc) in DOCUMENTS.iter().enumerate() {
        println!("  [{i}] {doc}");
    }
    println!();

    // Compare within-domain and cross-domain pairs.
    let pairs: &[(usize, usize, &str)] = &[
        (0, 1, "ML vs ML (same domain)"),
        (2, 3, "cooking vs cooking (same domain)"),
        (4, 5, "astronomy vs astronomy (same domain)"),
        (6, 7, "finance vs finance (same domain)"),
        (0, 2, "ML vs cooking (cross domain)"),
        (0, 4, "ML vs astronomy (cross domain)"),
        (2, 6, "cooking vs finance (cross domain)"),
        (4, 6, "astronomy vs finance (cross domain)"),
    ];

    println!("{:<45}  WMD", "Pair");
    println!("{:-<55}", "");
    for &(i, j, label) in pairs {
        let wmd = word_movers_distance(
            DOCUMENTS[i],
            DOCUMENTS[j],
            &codebook,
            &word_to_id,
            reg,
            max_iter,
        );
        println!("{label:<45}  {wmd:.4}");
    }

    println!();
    println!("=== Chunk-Level OT Distance ===");
    println!();

    // For chunk-level, concatenate related docs into longer texts so chunking
    // produces multiple chunks per "document".
    let long_a = format!("{}. {}. {}.", DOCUMENTS[0], DOCUMENTS[6], DOCUMENTS[4]);
    let long_b = format!("{}. {}. {}.", DOCUMENTS[1], DOCUMENTS[7], DOCUMENTS[5]);
    let long_c = format!("{}. {}. {}.", DOCUMENTS[2], DOCUMENTS[3], DOCUMENTS[4]);

    println!("Long document A (ML + finance + astronomy):");
    println!("  {long_a}");
    println!("Long document B (ML + finance + astronomy, paraphrased):");
    println!("  {long_b}");
    println!("Long document C (cooking + cooking + astronomy):");
    println!("  {long_c}");
    println!();

    let dist_ab = chunk_level_distance(&long_a, &long_b, &codebook, &word_to_id, reg, max_iter);
    let dist_ac = chunk_level_distance(&long_a, &long_c, &codebook, &word_to_id, reg, max_iter);
    let dist_bc = chunk_level_distance(&long_b, &long_c, &codebook, &word_to_id, reg, max_iter);

    println!("{:<45}  Chunk OT", "Pair");
    println!("{:-<60}", "");
    println!("{:<45}  {dist_ab:.4}", "A vs B (same topics, paraphrased)");
    println!(
        "{:<45}  {dist_ac:.4}",
        "A vs C (partially overlapping topics)"
    );
    println!(
        "{:<45}  {dist_bc:.4}",
        "B vs C (partially overlapping topics)"
    );

    println!();
    println!("Expected: within-domain WMD < cross-domain WMD.");
    println!("Expected: A vs B (chunk) < A vs C or B vs C (chunk).");
}
