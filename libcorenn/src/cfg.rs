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
  /// Alpha parameter for Vamana's RobustPrune (α in the DiskANN paper).
  /// Controls the tradeoff between graph sparsity and search path length:
  /// - α = 1.0: Standard RNG pruning, sparser graph, potentially longer paths
  /// - α > 1.0 (e.g., 1.2): More edges kept, guarantees O(log n) diameter
  /// The paper recommends α = 1.2 for disk-based systems.
  pub distance_threshold: f64,
  pub max_add_edges: usize,
  pub max_edges: usize,
  pub metric: StdMetric,
  pub pq_sample_size: usize,
  pub pq_subspaces: usize,
  pub query_search_list_cap: usize,
  pub trunc_dims: usize,
  pub update_search_list_cap: usize,
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
      // α = 1.2 as recommended in DiskANN paper for disk-based systems
      distance_threshold: 1.2,
      // Lazy pruning: allow 2x edges before triggering pruning.
      // This amortizes the cost of expensive pruning operations.
      max_add_edges: max_edges * 2,
      max_edges,
      metric: StdMetric::L2,  // L2 is the safe bet.
      pq_sample_size: 10_000, // Default: plenty, while fast to train.
      query_search_list_cap,
      update_search_list_cap: query_search_list_cap,
      // These defaults are completely arbitrary, they should be set manually.
      dim: 0,
      pq_subspaces: 64,
      trunc_dims: 64,
    }
  }
}
