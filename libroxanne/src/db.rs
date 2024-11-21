use crate::common::Dtype;
use crate::common::Id;
use bitcode::Decode;
use bitcode::Encode;
use bytemuck::cast_slice;
use ndarray::Array1;
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
  Deleted, // (Id)
  HasLongTermIndex,
  Id,  // (Vec<u8>)
  Key, // (Id)
  Medoid,
  Node,  // (Id)
  PqVec, // (Id)
  TempIndexCount,
  WriteAheadLogVector, // (Id)
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

  fn delete_raw(&mut self, k: impl DbKey) {
    self.batch.delete(k.bytes());
  }

  pub fn delete_deleted(&mut self, id: Id) {
    self.delete_raw((DbKeyT::Deleted, id));
  }

  pub fn delete_has_long_term_index(&mut self) {
    self.delete_raw(DbKeyT::HasLongTermIndex);
  }

  pub fn delete_id(&mut self, key: &str) {
    self.delete_raw((DbKeyT::Id, key));
  }

  pub fn delete_key(&mut self, id: Id) {
    self.delete_raw((DbKeyT::Key, id));
  }

  pub fn delete_medoid(&mut self) {
    self.delete_raw(DbKeyT::Medoid);
  }

  pub fn delete_node(&mut self, id: Id) {
    self.delete_raw((DbKeyT::Node, id));
  }

  pub fn delete_pq_vec(&mut self, id: Id) {
    self.delete_raw((DbKeyT::PqVec, id));
  }

  pub fn delete_temp_index_count(&mut self) {
    self.delete_raw(DbKeyT::TempIndexCount);
  }

  pub fn delete_write_ahead_log_vector(&mut self, id: Id) {
    self.delete_raw((DbKeyT::WriteAheadLogVector, id));
  }

  fn write_raw(&mut self, k: impl DbKey, v: impl AsRef<[u8]>) {
    self.batch.put(k.bytes(), v);
  }

  fn write(&mut self, k: impl DbKey, v: &impl Serialize) {
    self.write_raw(k, to_vec_named(v).unwrap());
  }

  pub fn write_deleted(&mut self, id: Id) {
    self.write_raw((DbKeyT::Deleted, id), Vec::new());
  }

  pub fn write_has_long_term_index(&mut self) {
    self.write_raw(DbKeyT::HasLongTermIndex, Vec::new());
  }

  pub fn write_id(&mut self, key: &str, id: Id) {
    self.write((DbKeyT::Id, key), &id);
  }

  pub fn write_key(&mut self, id: Id, key: &str) {
    self.write_raw((DbKeyT::Key, id), key.as_bytes());
  }

  pub fn write_medoid(&mut self, medoid: Id) {
    self.write(DbKeyT::Medoid, &medoid);
  }

  pub fn write_node<T: Dtype>(&mut self, id: Id, node: &NodeData<T>) {
    self.write_raw((DbKeyT::Node, id), node.serialize());
  }

  pub fn write_pq_vec(&mut self, id: Id, vec: Vec<u8>) {
    self.write_raw((DbKeyT::PqVec, id), vec);
  }

  pub fn write_temp_index_count(&mut self, count: usize) {
    self.write(DbKeyT::TempIndexCount, &count);
  }

  pub fn write_write_ahead_log_vector<T: Dtype>(&mut self, id: Id, vec: &[T]) {
    self.write_raw((DbKeyT::WriteAheadLogVector, id), cast_slice(vec));
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
pub struct NodeData<T> {
  pub neighbors: Vec<Id>,
  pub vector: Vec<T>,
}

impl<T: Dtype> NodeData<T> {
  pub fn serialize(&self) -> Vec<u8> {
    bitcode::encode(self)
  }

  pub fn deserialize(raw: &[u8]) -> Self {
    bitcode::decode(raw).unwrap()
  }
}

fn rocksdb_options() -> rocksdb::Options {
  // https://github.com/facebook/rocksdb/wiki/Setup-Options-and-Basic-Tuning#other-general-options.
  let mut opt = rocksdb::Options::default();
  opt.create_if_missing(true);
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
    let db = rocksdb::DB::open(&rocksdb_options(), dir).unwrap();
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
    parser: impl (Fn(Box<[u8]>) -> T) + 'a,
  ) -> impl Iterator<Item = (Id, T)> + 'a {
    self
      .db
      .full_iterator(IteratorMode::From(&[kt as u8], Direction::Forward))
      .map(|e| e.unwrap())
      .take_while(move |(k, _)| k[0] == (kt as u8))
      .map(move |(k, v)| {
        let id_raw = &k[1..];
        let id = Id::from_le_bytes(id_raw.try_into().unwrap());
        (id, parser(v))
      })
  }

  pub fn iter_deleted(&self) -> impl Iterator<Item = Id> + '_ {
    self.iter(DbKeyT::Deleted, |_| ()).map(|(id, _)| id)
  }

  pub fn iter_ids(&self) -> impl Iterator<Item = (Id, String)> + '_ {
    self.iter(DbKeyT::Id, |v| String::from_utf8(v.into_vec()).unwrap())
  }

  pub fn iter_nodes<T: Dtype>(&self) -> impl Iterator<Item = (Id, NodeData<T>)> + '_ {
    self.iter(DbKeyT::Node, |v| NodeData::deserialize(&v))
  }

  pub fn iter_write_ahead_log_vectors<T: Dtype>(
    &self,
  ) -> impl Iterator<Item = (Id, Array1<T>)> + '_ {
    self.iter(DbKeyT::WriteAheadLogVector, |v| {
      Array1::from_vec(cast_slice(&v).to_vec())
    })
  }

  fn maybe_read_raw(&self, k: impl DbKey) -> Option<DBPinnableSlice<'_>> {
    self.db.get_pinned(k.bytes()).unwrap()
  }

  fn read_raw(&self, k: impl DbKey) -> DBPinnableSlice<'_> {
    self.maybe_read_raw(k).unwrap()
  }

  pub fn read_deleted(&self, id: Id) -> bool {
    self.maybe_read_raw((DbKeyT::Deleted, id)).is_some()
  }

  pub fn read_has_long_term_index(&self) -> bool {
    self.maybe_read_raw(DbKeyT::HasLongTermIndex).is_some()
  }

  pub fn maybe_read_id(&self, key: &str) -> Option<Id> {
    let raw = self.maybe_read_raw((DbKeyT::Id, key))?;
    Some(rmp_serde::from_slice(&raw).unwrap())
  }

  pub fn read_key(&self, id: Id) -> String {
    let raw = self.read_raw((DbKeyT::Key, id));
    String::from_utf8(raw.to_vec()).unwrap()
  }

  pub fn read_medoid(&self) -> Id {
    let raw = self.read_raw(DbKeyT::Medoid);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_node<T: Dtype>(&self, id: Id) -> NodeData<T> {
    let raw = self.read_raw((DbKeyT::Node, id));
    NodeData::deserialize(&raw)
  }

  pub fn read_pq_vec(&self, id: Id) -> Vec<u8> {
    let raw = self.read_raw((DbKeyT::PqVec, id));
    raw.to_vec()
  }

  pub fn read_temp_index_count(&self) -> usize {
    let raw = self.read_raw(DbKeyT::TempIndexCount);
    rmp_serde::from_slice(&raw).unwrap()
  }

  pub fn read_write_ahead_log_vector<T: Dtype>(&self, id: Id) -> Vec<T> {
    let raw = self.read_raw((DbKeyT::WriteAheadLogVector, id));
    cast_slice(&raw).to_vec()
  }
}
