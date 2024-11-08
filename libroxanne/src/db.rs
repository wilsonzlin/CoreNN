use crate::pq::ProductQuantizer;
use crate::vamana::VamanaParams;
use bitcode::Decode;
use bitcode::Encode;
use libroxanne_search::Id;
use libroxanne_search::StdMetric;
use rocksdb::BlockBasedOptions;
use serde::Serialize;
use std::path::Path;

// We store both in one DB entry to leverage one disk page read to get both, as specified in the DiskANN paper. (If we store them as separate DB entries, they are unlikely to be stored in the same disk page.) We don't have to store the embedding.
// Why we chose Bitcode:
// - It's really fast: https://github.com/djkoloski/rust_serialization_benchmark.
// - It compactly stores integers using reduced bits.
// - It doesn't use space to store field names or types.
// These are crucial as we want to pack this in under one disk page read, which DiskANN relies on.
// We don't store as MessagePack like the other DB entries as that's less efficient for this specific use case.
#[derive(Encode, Decode)]
pub struct NodeData {
  pub neighbors: Vec<Id>,
  pub vector: Vec<f32>,
}

impl NodeData {
  pub fn serialize(&self) -> Vec<u8> {
    bitcode::encode(self)
  }

  pub fn deserialize(raw: &[u8]) -> Self {
    bitcode::decode(raw).unwrap()
  }
}

fn rocksdb_options(create: bool) -> rocksdb::Options {
  // https://github.com/facebook/rocksdb/wiki/Setup-Options-and-Basic-Tuning#other-general-options.
  let mut opt = rocksdb::Options::default();
  opt.create_if_missing(create);
  opt.set_error_if_exists(create);
  opt.set_max_background_jobs(num_cpus::get() as i32 * 2);
  opt.set_bytes_per_sync(1024 * 1024 * 4);
  opt.set_write_buffer_size(1024 * 1024 * 128);
  opt.set_compression_type(rocksdb::DBCompressionType::None);

  // https://github.com/facebook/rocksdb/wiki/Block-Cache.
  let mut bbt_opt = BlockBasedOptions::default();
  // Use close to a single page size to avoid excessive reads as that counters design of NodeData (seek DiskANN paper).
  bbt_opt.set_block_size(1024 * 4);
  bbt_opt.disable_cache();
  bbt_opt.set_format_version(6);
  opt.set_block_based_table_factory(&bbt_opt);
  opt
}

pub struct Db {
  db: rocksdb::DB,
}

impl Db {
  pub fn open(dir: impl AsRef<Path>) -> Self {
    let db = rocksdb::DB::open(&rocksdb_options(false), dir).unwrap();
    Db { db }
  }

  pub fn create(dir: impl AsRef<Path>) -> Self {
    let db = rocksdb::DB::open(&rocksdb_options(true), dir).unwrap();
    Db { db }
  }

  pub fn inner(&self) -> &rocksdb::DB {
    &self.db
  }

  pub fn flush(&self) {
    self.db.flush().unwrap();
  }

  fn write_raw(&self, k: impl AsRef<str>, v: Vec<u8>) {
    let k = k.as_ref().to_string().into_bytes();
    self.db.put(k, v).unwrap();
  }

  fn write(&self, k: impl AsRef<str>, v: &impl Serialize) {
    self.write_raw(k, rmp_serde::to_vec_named(v).unwrap());
  }

  pub fn read_cfg(&self) -> VamanaParams {
    let raw = self.db.get_pinned("cfg").unwrap().unwrap();
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn write_cfg(&self, cfg: &VamanaParams) {
    self.write("cfg", cfg);
  }

  pub fn read_dim(&self) -> usize {
    let raw = self.db.get_pinned("dim").unwrap().unwrap();
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn write_dim(&self, dim: usize) {
    self.write("dim", &dim);
  }

  pub fn read_medoid(&self) -> Id {
    let raw = self.db.get_pinned("medoid").unwrap().unwrap();
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn write_medoid(&self, medoid: Id) {
    self.write("medoid", &medoid);
  }

  pub fn read_metric(&self) -> StdMetric {
    let raw = self.db.get_pinned("metric").unwrap().unwrap();
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn write_metric(&self, metric: StdMetric) {
    self.write("metric", &metric);
  }

  pub fn read_node(&self, id: Id) -> NodeData {
    let raw = self.db.get_pinned(format!("node/{id}")).unwrap().unwrap();
    NodeData::deserialize(&raw)
  }

  pub fn write_node(&self, id: Id, node: &NodeData) {
    self.write_raw(format!("node/{id}"), node.serialize());
  }

  pub fn read_pq_model(&self) -> ProductQuantizer<f32> {
    let raw = self.db.get_pinned("pq_model").unwrap().unwrap();
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn write_pq_model(&self, pq: &ProductQuantizer<f32>) {
    self.write("pq_model", pq);
  }

  pub fn read_pq_vec(&self, id: Id) -> Vec<u8> {
    let raw = self.db.get_pinned(format!("pq_vec/{id}")).unwrap().unwrap();
    raw.to_vec()
  }

  pub fn write_pq_vec(&self, id: Id, vec: Vec<u8>) {
    self.write_raw(format!("pq_vec/{id}"), vec);
  }
}
