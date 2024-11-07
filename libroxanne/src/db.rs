use crate::vamana::VamanaParams;
use bitcode::Decode;
use bitcode::Encode;
use libroxanne_search::Id;
use libroxanne_search::StdMetric;
use lmdb::Database;
use lmdb::DatabaseFlags;
use lmdb::Environment;
use lmdb::Transaction;
use lmdb::WriteFlags;
use serde::Serialize;
use std::fs::create_dir_all;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;
use std::thread::spawn;
use std::thread::JoinHandle;

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

// Use an arbitrarily huge value for map size; the default is 10 MiB.
const DB_MAP_SIZE_MAX: usize = 1024 * 1024 * 1024 * 1024 * 64;

/// Perform a large batch of writes in one go, avoiding excessive LMDB-handled transactions and locking.
pub struct DbWriter {
  send: Sender<(Vec<u8>, Vec<u8>)>,
  thread: JoinHandle<()>,
}

impl DbWriter {
  pub fn new(dir: PathBuf) -> Self {
    let (send, recv) = channel::<(Vec<u8>, Vec<u8>)>();
    let thread = spawn(move || {
      create_dir_all(&dir).unwrap();
      let db_env = Environment::new()
        .set_map_size(DB_MAP_SIZE_MAX)
        .open(&dir)
        .unwrap();
      let db = db_env.create_db(None, DatabaseFlags::empty()).unwrap();
      let mut txn = db_env.begin_rw_txn().unwrap();
      while let Some((k, v)) = recv.recv().ok() {
        txn.put(db, &k, &v, WriteFlags::empty()).unwrap();
      }
      txn.commit().unwrap();
    });
    Self { send, thread }
  }

  pub fn finish(self) {
    drop(self.send);
    self.thread.join().unwrap();
  }

  fn write_raw(&self, k: impl AsRef<str>, v: Vec<u8>) {
    let k = k.as_ref().to_string().into_bytes();
    self.send.send((k, v)).unwrap();
  }

  fn write(&self, k: impl AsRef<str>, v: &impl Serialize) {
    self.write_raw(k, rmp_serde::to_vec_named(v).unwrap());
  }

  pub fn write_cfg(&self, cfg: &VamanaParams) {
    self.write("cfg", cfg);
  }

  pub fn write_dim(&self, dim: usize) {
    self.write("dim", &dim);
  }

  pub fn write_medoid(&self, medoid: Id) {
    self.write("medoid", &medoid);
  }

  pub fn write_metric(&self, metric: StdMetric) {
    self.write("metric", &metric);
  }

  pub fn write_node(&self, id: Id, node: &NodeData) {
    self.write_raw(format!("node/{id}"), node.serialize());
  }
}

pub struct DbReader {
  db_env: Environment,
  db: Database,
}

impl DbReader {
  pub fn new(dir: impl AsRef<Path>) -> Self {
    let db_env = Environment::new()
      .set_map_size(DB_MAP_SIZE_MAX)
      .open(dir.as_ref())
      .unwrap();
    let db = db_env.open_db(None).unwrap();
    Self { db_env, db }
  }

  pub fn read_cfg(&self) -> VamanaParams {
    let txn = self.db_env.begin_ro_txn().unwrap();
    let raw = txn.get(self.db, &"cfg").unwrap();
    rmp_serde::from_slice(raw).unwrap()
  }

  pub fn read_dim(&self) -> usize {
    let txn = self.db_env.begin_ro_txn().unwrap();
    let raw = txn.get(self.db, &"dim").unwrap();
    rmp_serde::from_slice(raw).unwrap()
  }

  pub fn read_medoid(&self) -> Id {
    let txn = self.db_env.begin_ro_txn().unwrap();
    let raw = txn.get(self.db, &"medoid").unwrap();
    rmp_serde::from_slice(raw).unwrap()
  }

  pub fn read_metric(&self) -> StdMetric {
    let txn = self.db_env.begin_ro_txn().unwrap();
    let raw = txn.get(self.db, &"metric").unwrap();
    rmp_serde::from_slice(raw).unwrap()
  }

  pub fn read_node(&self, id: Id) -> NodeData {
    let txn = self.db_env.begin_ro_txn().unwrap();
    let raw = txn.get(self.db, &format!("node/{id}")).unwrap();
    NodeData::deserialize(raw)
  }
}
