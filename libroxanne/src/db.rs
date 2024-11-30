use crate::common::Dtype;
use crate::common::Id;
use bitcode::Decode;
use bitcode::Encode;
use flume::r#async::RecvStream;
use futures::Stream;
use futures::StreamExt;
use rmp_serde::to_vec_named;
use rocksdb::BlockBasedOptions;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use rocksdb::WriteBatchWithTransaction;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::spawn_blocking;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DbIndexMode {
  BruteForce,
  InMemory,
  LongTerm,
}

#[derive(PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum DbKeyT {
  AdditionalOutNeighbors, // (Id)
  Deleted,                // (Id)
  IndexMode,
  Id,  // (Vec<u8>)
  Key, // (Id)
  Medoid,
  Node, // (Id)
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

  pub fn delete_additional_out_neighbors(&mut self, id: Id) {
    self.delete_raw((DbKeyT::AdditionalOutNeighbors, id));
  }

  pub fn delete_deleted(&mut self, id: Id) {
    self.delete_raw((DbKeyT::Deleted, id));
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

  fn write_raw(&mut self, k: impl DbKey, v: impl AsRef<[u8]>) {
    self.batch.put(k.bytes(), v);
  }

  fn write(&mut self, k: impl DbKey, v: &impl Serialize) {
    self.write_raw(k, to_vec_named(v).unwrap());
  }

  pub fn write_additional_out_neighbors(&mut self, id: Id, neighbors: &[Id]) {
    // Use bitcode to squeeze even more byte savings, important for the purpose of AdditionalOutNeighbors.
    self.write_raw(
      (DbKeyT::AdditionalOutNeighbors, id),
      bitcode::encode(neighbors),
    );
  }

  pub fn write_deleted(&mut self, id: Id) {
    self.write_raw((DbKeyT::Deleted, id), Vec::new());
  }

  pub fn write_id(&mut self, key: &str, id: Id) {
    self.write((DbKeyT::Id, key), &id);
  }

  pub fn write_index_mode(&mut self, mode: DbIndexMode) {
    self.write(DbKeyT::IndexMode, &mode);
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

  pub async fn commit(self, db: &Db) {
    let db = db.db.clone();
    spawn_blocking(move || db.write(self.batch).unwrap())
      .await
      .unwrap();
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
  // Use Arc so we can use spawn_blocking.
  db: Arc<rocksdb::DB>,
}

impl Db {
  pub async fn open(dir: PathBuf) -> Self {
    let db = spawn_blocking(move || rocksdb::DB::open(&rocksdb_options(), dir).unwrap())
      .await
      .unwrap()
      .into();
    Db { db }
  }

  pub fn inner(&self) -> &rocksdb::DB {
    &self.db
  }

  pub async fn flush(&self) {
    let db = self.db.clone();
    spawn_blocking(move || db.flush().unwrap()).await.unwrap();
  }

  fn iter<T: Send + 'static>(
    &self,
    kt: DbKeyT,
    parser: impl (Fn(Box<[u8]>) -> T) + Send + 'static,
  ) -> RecvStream<(Id, T)> {
    let db = self.db.clone();
    let (send, recv) = flume::bounded(16);
    spawn_blocking(move || {
      for e in db.full_iterator(IteratorMode::From(&[kt as u8], Direction::Forward)) {
        let (k, v) = e.unwrap();
        if k[0] != (kt as u8) {
          break;
        }
        let id_raw = &k[1..];
        let id = Id::from_le_bytes(id_raw.try_into().unwrap());
        let Ok(_) = send.send((id, parser(v))) else {
          break;
        };
      }
    });
    recv.into_stream()
  }

  pub fn iter_additional_out_neighbors(&self) -> RecvStream<(Id, Vec<Id>)> {
    self.iter(DbKeyT::AdditionalOutNeighbors, |v| {
      bitcode::decode(&v).unwrap()
    })
  }

  pub fn iter_deleted(&self) -> impl Stream<Item = Id> + '_ {
    self.iter(DbKeyT::Deleted, |_| ()).map(|(id, _)| id)
  }

  pub fn iter_ids(&self) -> RecvStream<(Id, String)> {
    self.iter(DbKeyT::Id, |v| String::from_utf8(v.into_vec()).unwrap())
  }

  pub fn iter_nodes<T: Dtype>(&self) -> RecvStream<(Id, NodeData<T>)> {
    self.iter(DbKeyT::Node, |v| NodeData::deserialize(&v))
  }

  async fn maybe_read_raw(&self, k: impl DbKey) -> Option<Vec<u8>> {
    let db = self.db.clone();
    let key = k.bytes();
    spawn_blocking(move || db.get(key).unwrap()).await.unwrap()
  }

  async fn read_raw(&self, k: impl DbKey) -> Vec<u8> {
    self.maybe_read_raw(k).await.unwrap()
  }

  pub async fn read_additional_out_neighbors(&self, id: Id) -> Vec<Id> {
    self
      .maybe_read_raw((DbKeyT::AdditionalOutNeighbors, id))
      .await
      .map(|raw| bitcode::decode(&raw).unwrap())
      .unwrap_or_default()
  }

  pub async fn read_index_mode(&self) -> DbIndexMode {
    self
      .maybe_read_raw(DbKeyT::IndexMode)
      .await
      .map(|raw| rmp_serde::from_slice(&raw).unwrap())
      .unwrap_or(DbIndexMode::BruteForce)
  }

  pub async fn read_deleted(&self, id: Id) -> bool {
    self.maybe_read_raw((DbKeyT::Deleted, id)).await.is_some()
  }

  pub async fn maybe_read_id(&self, key: &str) -> Option<Id> {
    let raw = self.maybe_read_raw((DbKeyT::Id, key)).await?;
    Some(rmp_serde::from_slice(&raw).unwrap())
  }

  pub async fn maybe_read_key(&self, id: Id) -> Option<String> {
    self
      .maybe_read_raw((DbKeyT::Key, id))
      .await
      .map(|raw| String::from_utf8(raw.to_vec()).unwrap())
  }

  pub async fn read_key(&self, id: Id) -> String {
    self.maybe_read_key(id).await.unwrap()
  }

  pub async fn maybe_read_medoid(&self) -> Option<Id> {
    self
      .maybe_read_raw(DbKeyT::Medoid)
      .await
      .map(|raw| rmp_serde::from_slice(&raw).unwrap())
  }

  pub async fn read_node<T: Dtype>(&self, id: Id) -> NodeData<T> {
    let raw = self.read_raw((DbKeyT::Node, id)).await;
    NodeData::deserialize(&raw)
  }
}
