use crate::common::Dtype;
use crate::common::Id;
use crate::common::Metric;
use crate::in_memory::InMemoryIndex;
use crate::pq::ProductQuantizer;
use crate::vamana::VamanaParams;
use dashmap::DashMap;
use data_encoding::BASE64URL_NOPAD;
use linfa::Float;
use ndarray::Array1;
use rand::thread_rng;
use rand::RngCore;
use rmp_serde::from_slice;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;

/// Subset of `InMemoryIndex` fields that we need to store (the rest can be derived).
// https://stackoverflow.com/a/51465659
#[derive(Serialize, Deserialize)]
pub struct InMemoryIndexBlob<T> {
  // This is Arc so it can be easily shared with InMemoryIndex for serialization.
  pub graph: Arc<DashMap<Id, Vec<Id>>>,
  // This is Arc so it can be easily shared with InMemoryIndex for serialization.
  pub vectors: Arc<DashMap<Id, Array1<T>>>,
  pub medoid: Id,
}

impl<T: Dtype> From<&InMemoryIndex<T>> for InMemoryIndexBlob<T> {
  fn from(value: &InMemoryIndex<T>) -> Self {
    Self {
      graph: value.graph.clone(),
      medoid: value.medoid,
      vectors: value.vectors.clone(),
    }
  }
}

impl<T: Dtype> InMemoryIndexBlob<T> {
  pub fn to_index(&self, metric: Metric<T>, params: VamanaParams) -> InMemoryIndex<T> {
    InMemoryIndex {
      graph: self.graph.clone(),
      medoid: self.medoid,
      metric,
      params,
      precomputed_dists: None,
      vectors: self.vectors.clone(),
    }
  }
}

// Some large data should be stored outside the DB:
// - Too big, causing write amplification when RocksDB has to move and compact data around on disk.
// - Doesn't change often, and only loaded once.
pub struct BlobStore {
  dir: PathBuf,
}

impl BlobStore {
  pub fn open(dir: PathBuf) -> Self {
    fs::create_dir_all(&dir).unwrap();
    Self { dir }
  }

  fn write(&self, name: &str, val: &impl Serialize) {
    let p = self.dir.join(name);
    let mut tmp_sfx = vec![0u8; 24];
    thread_rng().fill_bytes(&mut tmp_sfx);
    let p_tmp = p.with_added_extension(format!("tmp_{}", BASE64URL_NOPAD.encode(&tmp_sfx)));
    let f = File::create(&p_tmp).unwrap();
    let mut bw = BufWriter::new(f);
    let mut s = rmp_serde::Serializer::new(&mut bw).with_struct_map();
    val.serialize(&mut s).unwrap();
    fs::rename(&p_tmp, p).unwrap();
  }

  pub fn read_pq_model<T: Dtype + Float>(&self) -> ProductQuantizer<T> {
    let raw = fs::read(self.dir.join("pq_model")).unwrap();
    from_slice(&raw).unwrap()
  }

  pub fn write_pq_model<T: Dtype + Float>(&self, pq_model: &ProductQuantizer<T>) {
    self.write("pq_model", pq_model);
  }
}
