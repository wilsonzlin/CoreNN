use crate::metric::StdMetric;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CompressionMode {
  // TODO Other options:
  // - PCA
  // - UMAP
  // Product Quantization: high compression, slower training.
  PQ,
  // Scalar Quantization (int8): 4x compression, fast, simple.
  SQ,
  // For Matryoshka embeddings.
  Trunc,
}

// This is the sparse object.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Cfg {
  pub beam_width: usize,
  pub compaction_threshold_deletes: usize,
  pub compression_mode: CompressionMode,
  pub compression_threshold: usize,
  pub dim: usize,
  pub distance_threshold: f64,
  pub max_add_edges: usize,
  pub max_edges: usize,
  pub metric: StdMetric,
  pub pq_sample_size: usize,
  pub pq_subspaces: usize,
  pub query_search_list_cap: usize,
  /// Rerank factor for two-phase search. When > 1.0, retrieves k * rerank_factor
  /// candidates using compressed distances, then reranks with exact distances.
  /// 1.0 = no reranking (default), 2.0 = retrieve 2x candidates for reranking.
  pub rerank_factor: f32,
  pub trunc_dims: usize,
  pub update_search_list_cap: usize,
  /// Use faster HNSW-style neighbor selection (O(M×C)) instead of Vamana RNG (O(C²)).
  /// Faster inserts but potentially 10-20% slower queries.
  /// Default: false (use original Vamana RNG for best query performance).
  pub use_hnsw_heuristic: bool,
}

impl Default for Cfg {
  fn default() -> Self {
    let max_edges = 64;
    let query_search_list_cap = max_edges * 2;
    Self {
      beam_width: 4,
      compaction_threshold_deletes: 1_000_000,
      compression_mode: CompressionMode::PQ,
      compression_threshold: 10_000_000,
      distance_threshold: 1.1,
      // Lazy pruning: allow 2x edges before triggering pruning.
      // This amortizes the cost of expensive pruning operations.
      max_add_edges: max_edges * 2,
      max_edges,
      metric: StdMetric::L2, // L2 is the safe bet.
      pq_sample_size: 10_000, // Default: plenty, while fast to train.
      query_search_list_cap,
      rerank_factor: 1.0, // No reranking by default. Set to 2.0-4.0 for better recall with compression.
      update_search_list_cap: query_search_list_cap,
      // These defaults are completely arbitrary, they should be set manually.
      dim: 0,
      pq_subspaces: 64,
      trunc_dims: 64,
      use_hnsw_heuristic: false, // Default to Vamana RNG for best query performance
    }
  }
}
