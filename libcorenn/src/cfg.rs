use crate::metric::StdMetric;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CompressionMode {
  // TODO Other options:
  // - PCA
  // - UMAP
  // - Scalar quantization (int8/int4/int2/int1)
  PQ,
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
  pub distance_threshold: f32,
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
      distance_threshold: 1.1,
      max_add_edges: max_edges,
      max_edges,
      metric: StdMetric::L2Sq, // L2 is the safe bet.
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
