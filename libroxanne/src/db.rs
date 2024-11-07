use bitcode::Decode;
use bitcode::Encode;
use libroxanne_search::Id;
use lmdb::DatabaseFlags;
use lmdb::Environment;
use lmdb::Transaction;
use lmdb::WriteFlags;
use serde::Serialize;
use std::fs::create_dir_all;
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
      // Use an arbitrarily huge value for map size; the default is 10 MiB.
      let db_env = Environment::new()
        .set_map_size(1024 * 1024 * 1024 * 1024 * 64)
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

  pub fn write(&self, k: impl AsRef<str>, v: &impl Serialize) {
    let k = k.as_ref().to_string().into_bytes();
    let v = rmp_serde::to_vec_named(v).unwrap();
    self.send.send((k, v)).unwrap();
  }

  pub fn finish(self) {
    drop(self.send);
    self.thread.join().unwrap();
  }
}
