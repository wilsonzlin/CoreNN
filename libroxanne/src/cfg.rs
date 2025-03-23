use crate::common::StdMetric;
use crate::util::AtomUsz;
use parking_lot::RwLock;
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
pub struct CfgRaw {
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

// Wrapper that computes final Cfg values, many of which are dynamic: system RAM, vector count, dim., etc.
pub struct Cfg {
  pub raw: RwLock<CfgRaw>,
  // Atomic and public as this should change after first insert.
  pub dim: AtomUsz,
}

impl Cfg {
  pub fn new(raw: CfgRaw) -> Self {
    Self {
      raw: RwLock::new(raw),
      dim: AtomUsz::new(0),
    }
  }

  fn dim(&self) -> usize {
    let mut dim = self.dim.get();
    if dim == 0 {
      // Assume reasonable default.
      dim = 512;
    };
    dim
  }

  pub fn beam_width(&self) -> usize {
    self.raw.read().beam_width.unwrap_or(4)
  }

  pub fn compaction_threshold_deletes(&self) -> usize {
    self
      .raw
      .read()
      .compaction_threshold_deletes
      .unwrap_or(1_000_000)
  }

  pub fn compression_mode(&self) -> CompressionMode {
    // PQ is the safe bet.
    self
      .raw
      .read()
      .compression_mode
      .unwrap_or(CompressionMode::PQ)
  }

  pub fn compression_threshold(&self) -> usize {
    self.raw.read().compression_threshold.unwrap_or(10_000_000)
  }

  pub fn distance_threshold(&self) -> f32 {
    self.raw.read().distance_threshold.unwrap_or(1.1)
  }

  pub fn max_add_edges(&self) -> usize {
    // Default to allowing a node to expand up to 2x max_edges.
    self
      .raw
      .read()
      .max_add_edges
      .unwrap_or_else(|| self.max_edges())
  }

  pub fn max_edges(&self) -> usize {
    self.raw.read().max_edges.unwrap_or_else(|| {
      let dim = self.dim();
      // Complete guesses, but reasonable for modern Transformer-based embeddings representing unstructured data.
      // `dim` values near boundaries likely have issues.
      if dim <= 128 {
        dim / 4
      } else if dim <= 1024 {
        dim / 8
      } else {
        dim / 12
      }
    })
  }

  pub fn metric(&self) -> StdMetric {
    // Cosine is the one that works better for high-dimensional embeddings, esp. ones representing unstructured data.
    // However, L2 is important for lots of other classic applications: manual features, GIST, etc.
    // So this is a toss-up. Go with Cosine because more people likely have that use case.
    self.raw.read().metric.unwrap_or(StdMetric::Cosine)
  }

  pub fn pq_sample_size(&self) -> usize {
    // Default: plenty, while fast to train.
    self.raw.read().pq_sample_size.unwrap_or(10_000)
  }

  pub fn pq_subspaces(&self) -> usize {
    self
      .raw
      .read()
      .pq_subspaces
      .unwrap_or_else(|| self.dim() / 8)
  }

  pub fn query_search_list_cap(&self) -> usize {
    self
      .raw
      .read()
      .query_search_list_cap
      .unwrap_or_else(|| self.max_edges() * 2)
  }

  pub fn trunc_dims(&self) -> usize {
    // This default is completely arbitrary — anyone using Trunc (i.e. Matryoshka embeddings) should manually set this.
    self.raw.read().trunc_dims.unwrap_or_else(|| 64)
  }

  pub fn update_batch_size(&self) -> usize {
    self
      .raw
      .read()
      .update_batch_size
      .unwrap_or_else(|| num_cpus::get())
  }

  pub fn update_search_list_cap(&self) -> usize {
    self
      .raw
      .read()
      .update_search_list_cap
      .unwrap_or_else(|| self.query_search_list_cap())
  }
}
