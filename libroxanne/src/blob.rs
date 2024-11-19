use crate::common::Dtype;
use crate::common::Id;
use crate::pq::ProductQuantizer;
use dashmap::DashMap;
use data_encoding::BASE64URL_NOPAD;
use itertools::Itertools;
use linfa::Float;
use ndarray::Array1;
use rand::thread_rng;
use rand::RngCore;
use rmp_serde::from_slice;
use rmp_serde::to_vec_named;
use serde::Deserialize;
use serde::Serialize;
use std::fs;

/// Subset of `InMemoryIndex` fields that we need to store (the rest can be derived).
// https://stackoverflow.com/a/51465659
#[derive(Serialize, Deserialize)]
pub struct InMemoryIndexBlob<T> {
  pub graph: DashMap<Id, Vec<Id>>,
  pub vectors: DashMap<Id, Array1<T>>,
  pub medoid: Id,
}

// Some large data should be stored outside the DB:
// - Too big, causing write amplification when RocksDB has to move and compact data around on disk.
// - Doesn't change often, and only loaded once.
pub struct BlobStore {
  dir: String,
}

impl BlobStore {
  pub fn new(dir: String) -> Self {
    fs::create_dir_all(format!("{}/in_memory_indices", dir)).unwrap();
    Self { dir }
  }

  fn write(&self, name: &str, val_raw: &[u8]) {
    let p = format!("{}/{name}", self.dir);
    let mut tmp_sfx = vec![0u8; 24];
    thread_rng().fill_bytes(&mut tmp_sfx);
    let p_tmp = format!("{p}.tmp_{}", BASE64URL_NOPAD.encode(&tmp_sfx));
    fs::write(&p_tmp, val_raw).unwrap();
    fs::rename(&p_tmp, p).unwrap();
  }

  pub fn read_all_in_memory_indices<T: Dtype>(&self) -> Vec<(usize, InMemoryIndexBlob<T>)> {
    fs::read_dir(format!("{}/in_memory_indices", self.dir))
      .unwrap()
      .map(|e| {
        let id_raw = e.unwrap().file_name().into_string().unwrap();
        let id = usize::from_str_radix(&id_raw, 10).unwrap();
        let raw = fs::read(format!("{}/in_memory_indices/{id}", self.dir)).unwrap();
        let blob: InMemoryIndexBlob<T> = from_slice(&raw).unwrap();
        (id, blob)
      })
      .collect_vec()
  }

  pub fn write_in_memory_index<T: Dtype>(&self, id: usize, blob: &InMemoryIndexBlob<T>) {
    self.write(
      &format!("in_memory_indices/{id}"),
      &to_vec_named(blob).unwrap(),
    );
  }

  pub fn delete_all_in_memory_indices(&self) {
    fs::remove_dir_all(format!("{}/in_memory_indices", self.dir)).unwrap();
    fs::create_dir_all(format!("{}/in_memory_indices", self.dir)).unwrap();
  }

  pub fn read_pq_model<T: Dtype + Float>(&self) -> ProductQuantizer<T> {
    let raw = fs::read(format!("{}/pq_model", self.dir)).unwrap();
    from_slice(&raw).unwrap()
  }

  pub fn write_pq_model<T: Dtype + Float>(&self, pq_model: &ProductQuantizer<T>) {
    let raw = to_vec_named(pq_model).unwrap();
    self.write("pq_model", &raw);
  }
}
