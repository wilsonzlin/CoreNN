use crate::common::StdMetric;
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompressionMode {
  // TODO Other options:
  // - PCA
  // - UMAP
  // - Scalar quantization (int8/int4/int2/int1)
  PQ,
  // For Matryoshka embeddings.
  Trunc,
}

// This is the sparse object stored in the DB.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SparseCfg {
  pub beam_width: Option<usize>,
  pub compaction_threshold_deletes: Option<usize>,
  pub compression_mode: Option<CompressionMode>,
  pub compression_threshold: Option<usize>,
  pub distance_threshold: Option<f32>,
  pub max_add_edges: Option<usize>,
  pub max_edges: Option<usize>,
  pub metric: Option<StdMetric>,
  pub pq_sample_size: Option<usize>,
  pub pq_subspaces: Option<usize>,
  pub query_search_list_cap: Option<usize>,
  pub trunc_dims: Option<usize>,
  pub update_batch_size: Option<usize>,
  pub update_search_list_cap: Option<usize>,
}

// Wrapper that computes final Cfg values.
#[derive(Clone, Debug)]
pub struct EffectiveCfg {
  pub beam_width: usize,
  pub compaction_threshold_deletes: usize,
  pub compression_mode: CompressionMode,
  pub compression_threshold: usize,
  pub distance_threshold: f32,
  pub max_add_edges: usize,
  pub max_edges: usize,
  pub metric: StdMetric,
  pub pq_sample_size: usize,
  pub pq_subspaces: usize,
  pub query_search_list_cap: usize,
  pub trunc_dims: usize,
  pub update_batch_size: usize,
  pub update_search_list_cap: usize,
}

impl EffectiveCfg {
  pub fn new(raw: SparseCfg, mut dim: usize) -> Self {
    if dim == 0 {
      // Assume reasonable default dimension.
      dim = 512;
    }

    // Calculate max_edges based on dimension.
    let max_edges = raw.max_edges.unwrap_or_else(|| {
      // Complete guesses, but reasonable for modern Transformer-based embeddings representing unstructured data.
      // `dim` values near boundaries likely have issues.
      if dim <= 128 {
        dim / 4
      } else if dim <= 1024 {
        dim / 8
      } else {
        dim / 12
      }
    });

    // Calculate query_search_list_cap based on max_edges.
    let query_search_list_cap = raw.query_search_list_cap.unwrap_or_else(|| max_edges * 2);

    Self {
      beam_width: raw.beam_width.unwrap_or(4),
      compaction_threshold_deletes: raw.compaction_threshold_deletes.unwrap_or(1_000_000),
      compression_mode: raw.compression_mode.unwrap_or(CompressionMode::PQ),
      compression_threshold: raw.compression_threshold.unwrap_or(10_000_000),
      distance_threshold: raw.distance_threshold.unwrap_or(1.1),
      max_add_edges: raw.max_add_edges.unwrap_or(max_edges),
      max_edges,
      // L2 is the safe bet.
      metric: raw.metric.unwrap_or(StdMetric::L2),
      // Default: plenty, while fast to train.
      pq_sample_size: raw.pq_sample_size.unwrap_or(10_000),
      pq_subspaces: raw.pq_subspaces.unwrap_or(dim / 8),
      query_search_list_cap,
      // This default is completely arbitrary — anyone using Trunc (i.e. Matryoshka embeddings) should manually set this.
      trunc_dims: raw.trunc_dims.unwrap_or(64),
      update_batch_size: raw.update_batch_size.unwrap_or(num_cpus::get()),
      update_search_list_cap: raw.update_search_list_cap.unwrap_or(query_search_list_cap),
    }
  }
}
