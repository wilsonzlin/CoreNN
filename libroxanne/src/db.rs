use crate::common::Id;
use crate::common::StdMetric;
use crate::pq::ProductQuantizer;
use bitcode::Decode;
use bitcode::Encode;
use rmp_serde::to_vec_named;
use rocksdb::BlockBasedOptions;
use rocksdb::DBPinnableSlice;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use rocksdb::WriteBatchWithTransaction;
use serde::Serialize;
use std::path::Path;

#[derive(PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum DbKeyT {
  CfgDegreeBound,
  CfgDistanceThreshold,
  CfgUpdateBatchSize,
  CfgUpdateSearchListCap,
  Deleted, // (Id)
  Dim,
  Id,  // (Vec<u8>)
  Key, // (Id)
  Medoid,
  Metric,
  NextId,
  Node,      // (Id)
  NodeCount, // Does not include deleted.
  PqModel,
  PqVec, // (Id)
  TempIndexOffsets,
}

pub trait DbKey {
  fn bytes(&self) -> Vec<u8>;
}

impl DbKey for DbKeyT {
  fn bytes(&self) -> Vec<u8> {
    vec![*self as u8]
  }
}

impl DbKey for (DbKeyT, Id) {
  fn bytes(&self) -> Vec<u8> {
    let mut res = vec![self.0 as u8];
    res.extend_from_slice(&self.1.to_le_bytes());
    res
  }
}

impl<'a> DbKey for (DbKeyT, &'a [u8]) {
  fn bytes(&self) -> Vec<u8> {
    let mut res = vec![self.0 as u8];
    res.extend_from_slice(self.1);
    res
  }
}

impl<'a> DbKey for (DbKeyT, &'a str) {
  fn bytes(&self) -> Vec<u8> {
    let mut res = vec![self.0 as u8];
    res.extend_from_slice(self.1.as_bytes());
    res
  }
}

pub struct DbTransaction {
  batch: WriteBatchWithTransaction<false>,
}

impl DbTransaction {
  pub fn new() -> Self {
    DbTransaction {
      batch: WriteBatchWithTransaction::default(),
    }
  }

  fn write_raw(&mut self, k: impl DbKey, v: Vec<u8>) {
    self.batch.put(k.bytes(), v);
  }

  fn write(&mut self, k: impl DbKey, v: &impl Serialize) {
    self.write_raw(k, to_vec_named(v).unwrap());
  }

  pub fn write_cfg_degree_bound(&mut self, dim: usize) {
    self.write(DbKeyT::CfgDegreeBound, &dim);
  }

  pub fn write_cfg_distance_threshold(&mut self, dist: f64) {
    self.write(DbKeyT::CfgDistanceThreshold, &dist);
  }

  pub fn write_cfg_update_batch_size(&mut self, size: usize) {
    self.write(DbKeyT::CfgUpdateBatchSize, &size);
  }

  pub fn write_cfg_update_search_list_cap(&mut self, cap: usize) {
    self.write(DbKeyT::CfgUpdateSearchListCap, &cap);
  }

  pub fn write_deleted(&mut self, id: Id) {
    self.write_raw((DbKeyT::Deleted, id), Vec::new());
  }

  pub fn write_dim(&mut self, dim: usize) {
    self.write(DbKeyT::Dim, &dim);
  }

  pub fn write_id(&mut self, key: &str, id: Id) {
    self.write((DbKeyT::Id, key), &id);
  }

  pub fn write_key(&mut self, id: Id, key: &str) {
    self.write_raw((DbKeyT::Key, id), key.as_bytes().to_vec());
  }

  pub fn write_medoid(&mut self, medoid: Id) {
    self.write(DbKeyT::Medoid, &medoid);
  }

  pub fn write_metric(&mut self, metric: StdMetric) {
    self.write(DbKeyT::Metric, &metric);
  }

  pub fn write_next_id(&mut self, id: Id) {
    self.write(DbKeyT::NextId, &id);
  }

  pub fn write_node(&mut self, id: Id, node: &NodeData) {
    self.write_raw((DbKeyT::Node, id), node.serialize());
  }

  pub fn write_node_count(&mut self, count: usize) {
    self.write(DbKeyT::NodeCount, &count);
  }

  pub fn write_pq_model(&mut self, pq: &ProductQuantizer<f32>) {
    self.write(DbKeyT::PqModel, pq);
  }

  pub fn write_pq_vec(&mut self, id: Id, vec: Vec<u8>) {
    self.write_raw((DbKeyT::PqVec, id), vec);
  }

  pub fn commit(self, db: &Db) {
    db.db.write(self.batch).unwrap();
  }
}

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
  // DiskANN explicitly avoids dependence on memory caches, so in the same spirit disable this.
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

  fn iter<'a, T>(
    &'a self,
    kt: DbKeyT,
    parser: impl Fn(&[u8]) -> T + 'a,
  ) -> impl Iterator<Item = (Id, T)> + 'a {
    self
      .db
      .full_iterator(IteratorMode::From(&[kt as u8], Direction::Forward))
      .map(|e| e.unwrap())
      .take_while(move |(k, _)| k[0] == (kt as u8))
      .map(move |(k, v)| {
        let id_raw = &k[1..];
        let id = Id::from_le_bytes(id_raw.try_into().unwrap());
        (id, parser(&v))
      })
  }

  pub fn iter_nodes(&self) -> impl Iterator<Item = (Id, NodeData)> + '_ {
    self.iter(DbKeyT::Node, |v| NodeData::deserialize(v))
  }

  fn maybe_read_raw(&self, k: impl DbKey) -> Option<DBPinnableSlice<'_>> {
    self.db.get_pinned(k.bytes()).unwrap()
  }

  fn read_raw(&self, k: impl DbKey) -> DBPinnableSlice<'_> {
    self.maybe_read_raw(k).unwrap()
  }

  pub fn read_cfg_degree_bound(&self) -> usize {
    let raw = self.read_raw(DbKeyT::CfgDegreeBound);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_cfg_distance_threshold(&self) -> f64 {
    let raw = self.read_raw(DbKeyT::CfgDistanceThreshold);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_cfg_update_batch_size(&self) -> usize {
    let raw = self.read_raw(DbKeyT::CfgUpdateBatchSize);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_cfg_update_search_list_cap(&self) -> usize {
    let raw = self.read_raw(DbKeyT::CfgUpdateSearchListCap);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_deleted(&self, id: Id) -> bool {
    self.maybe_read_raw((DbKeyT::Deleted, id)).is_some()
  }

  pub fn read_dim(&self) -> usize {
    let raw = self.read_raw(DbKeyT::Dim);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_id(&self, key: &str) -> Id {
    let raw = self.read_raw((DbKeyT::Id, key));
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_key(&self, id: Id) -> String {
    let raw = self.read_raw((DbKeyT::Key, id));
    String::from_utf8(raw.to_vec()).unwrap()
  }

  pub fn read_medoid(&self) -> Id {
    let raw = self.read_raw(DbKeyT::Medoid);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_metric(&self) -> StdMetric {
    let raw = self.read_raw(DbKeyT::Metric);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_next_id(&self) -> Id {
    let raw = self.read_raw(DbKeyT::NextId);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_node(&self, id: Id) -> NodeData {
    let raw = self.read_raw((DbKeyT::Node, id));
    NodeData::deserialize(&raw)
  }

  pub fn read_node_count(&self) -> usize {
    let raw = self.read_raw(DbKeyT::NodeCount);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_pq_model(&self) -> ProductQuantizer<f32> {
    let raw = self.read_raw(DbKeyT::PqModel);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_pq_vec(&self, id: Id) -> Vec<u8> {
    let raw = self.read_raw((DbKeyT::PqVec, id));
    raw.to_vec()
  }
}
