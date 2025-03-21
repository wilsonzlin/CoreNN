use crate::cfg::CfgRaw;
use crate::common::Id;
use crate::compressor::pq::ProductQuantizer;
use bitcode::decode;
use bitcode::encode;
use bitcode::Encode;
use bytemuck::cast_slice;
use flume::r#async::RecvStream;
use futures::Stream;
use futures::StreamExt;
use half::f16;
use ndarray::Array1;
use rocksdb::BlockBasedOptions;
use rocksdb::Direction;
use rocksdb::IteratorMode;
use rocksdb::ReadOptions;
use rocksdb::WriteBatchWithTransaction;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::spawn_blocking;

#[derive(PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum DbKeyT {
  AddEdges, // (Id)
  // For simplicity, the entire Cfg is just one DB entry, given that all fields are optional.
  Cfg,
  Count,
  Deleted, // (Id)
  Id,      // (Vec<u8>)
  Key,     // (Id)
  NextId,
  Node, // (Id)
  PQModel,
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

  pub fn into_inner(self) -> WriteBatchWithTransaction<false> {
    self.batch
  }

  fn delete_raw(&mut self, k: impl DbKey) {
    self.batch.delete(k.bytes());
  }

  pub fn delete_additional_out_neighbors(&mut self, id: Id) {
    self.delete_raw((DbKeyT::AddEdges, id));
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

  pub fn delete_node(&mut self, id: Id) {
    self.delete_raw((DbKeyT::Node, id));
  }

  pub fn delete_pq_model(&mut self) {
    self.delete_raw(DbKeyT::PQModel);
  }

  fn write_raw(&mut self, k: impl DbKey, v: impl AsRef<[u8]>) {
    self.batch.put(k.bytes(), v);
  }

  fn write(&mut self, k: impl DbKey, v: &impl Encode) {
    self.write_raw(k, encode(v));
  }

  pub fn write_add_edges(&mut self, id: Id, neighbors: &Vec<Id>) {
    self.write((DbKeyT::AddEdges, id), neighbors);
  }

  pub fn write_cfg(&mut self, cfg: &CfgRaw) {
    // Use MessagePack for future compatibility/extensibility.
    self.write_raw(DbKeyT::Cfg, rmp_serde::to_vec(cfg).unwrap());
  }

  pub fn write_count(&mut self, count: usize) {
    self.write(DbKeyT::Count, &count);
  }

  pub fn write_deleted(&mut self, id: Id) {
    self.write_raw((DbKeyT::Deleted, id), Vec::new());
  }

  pub fn write_id(&mut self, key: &str, id: Id) {
    self.write((DbKeyT::Id, key), &id);
  }

  pub fn write_key(&mut self, id: Id, key: &str) {
    self.write_raw((DbKeyT::Key, id), key.as_bytes());
  }

  pub fn write_next_id(&mut self, id: Id) {
    self.write(DbKeyT::NextId, &id);
  }

  pub fn write_node(&mut self, id: Id, node: &NodeData) {
    self.write_raw((DbKeyT::Node, id), node.serialize());
  }

  pub fn write_pq_model(&mut self, pq: &ProductQuantizer<f32>) {
    self.write_raw(DbKeyT::PQModel, rmp_serde::to_vec(pq).unwrap());
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
#[derive(Clone)]
pub struct NodeData {
  pub neighbors: Vec<Id>,
  pub vector: Array1<f16>,
}

impl NodeData {
  pub fn serialize(&self) -> Vec<u8> {
    // Bitcode doesn't support f16 at this time, so we must convert to raw [u8] first.
    // WARNING: This decision is permanent, unless we want to break the disk format.
    let vec_raw: Vec<u8> = cast_slice(self.vector.as_slice().unwrap()).to_vec();
    encode(&(self.neighbors.clone(), vec_raw))
  }

  pub fn deserialize(raw: &[u8]) -> Self {
    let (neighbors, vec_raw): (Vec<Id>, Vec<u8>) = decode(raw).unwrap();
    let vector: Vec<f16> = cast_slice(&vec_raw).to_vec();
    let vector = Array1::from_vec(vector);
    Self { neighbors, vector }
  }
}

pub fn rocksdb_options() -> rocksdb::Options {
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
      // Optimize iterator for one-off table scan.
      let mut readopts = ReadOptions::default();
      readopts.set_pin_data(false);
      readopts.fill_cache(false);
      readopts.set_async_io(true);
      readopts.set_auto_readahead_size(true);
      for e in db.iterator_opt(
        IteratorMode::From(&[kt as u8], Direction::Forward),
        readopts,
      ) {
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

  pub fn iter_add_edges(&self) -> RecvStream<(Id, Vec<Id>)> {
    self.iter(DbKeyT::AddEdges, |v| decode(&v).unwrap())
  }

  pub fn iter_deleted(&self) -> impl Stream<Item = Id> + '_ {
    self.iter(DbKeyT::Deleted, |_| ()).map(|(id, _)| id)
  }

  pub fn iter_ids(&self) -> RecvStream<(Id, String)> {
    self.iter(DbKeyT::Id, |v| String::from_utf8(v.into_vec()).unwrap())
  }

  pub fn iter_nodes(&self) -> RecvStream<(Id, NodeData)> {
    self.iter(DbKeyT::Node, |v| NodeData::deserialize(&v))
  }

  async fn maybe_read_raw(&self, k: impl DbKey) -> Option<Vec<u8>> {
    let db = self.db.clone();
    let key = k.bytes();
    spawn_blocking(move || db.get(key).unwrap()).await.unwrap()
  }

  pub async fn read_add_edges(&self, id: Id) -> Vec<Id> {
    self
      .maybe_read_raw((DbKeyT::AddEdges, id))
      .await
      .map(|raw| decode(&raw).unwrap())
      .unwrap_or_default()
  }

  pub async fn read_cfg(&self) -> CfgRaw {
    self
      .maybe_read_raw(DbKeyT::Cfg)
      .await
      .map(|raw| rmp_serde::from_slice(&raw).unwrap())
      .unwrap_or_default()
  }

  pub async fn read_count(&self) -> usize {
    self
      .maybe_read_raw(DbKeyT::Count)
      .await
      .map(|raw| decode(&raw).unwrap())
      .unwrap_or_default()
  }

  pub async fn read_deleted(&self, id: Id) -> bool {
    self.maybe_read_raw((DbKeyT::Deleted, id)).await.is_some()
  }

  pub async fn maybe_read_id(&self, key: &str) -> Option<Id> {
    self
      .maybe_read_raw((DbKeyT::Id, key))
      .await
      .map(|raw| decode(&raw).unwrap())
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

  pub async fn read_next_id(&self) -> Id {
    self
      .maybe_read_raw(DbKeyT::NextId)
      .await
      .map(|raw| decode(&raw).unwrap())
      .unwrap_or_default()
  }

  pub async fn maybe_read_node(&self, id: Id) -> Option<NodeData> {
    self
      .maybe_read_raw((DbKeyT::Node, id))
      .await
      .map(|raw| NodeData::deserialize(&raw))
  }

  pub async fn read_node(&self, id: Id) -> NodeData {
    self.maybe_read_node(id).await.unwrap()
  }

  pub async fn maybe_read_pq_model(&self) -> Option<ProductQuantizer<f32>> {
    self
      .maybe_read_raw(DbKeyT::PQModel)
      .await
      .map(|raw| rmp_serde::from_slice(&raw).unwrap())
  }

  pub async fn read_pq_model(&self) -> ProductQuantizer<f32> {
    self.maybe_read_pq_model().await.unwrap()
  }
}
