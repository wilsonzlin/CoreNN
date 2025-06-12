use super::Store;
use super::WriteOp;
use crate::cfg::Cfg;
use crate::common::Id;
use crate::compressor::pq::ProductQuantizer;
use crate::vec::VecData;
use rmp_serde::to_vec_named;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait DbEntKeySfxTyp: Send + Sync {
  fn read(raw: &[u8]) -> Self;
  fn write(&self, out: &mut Vec<u8>);
}

impl DbEntKeySfxTyp for Id {
  fn read(raw: &[u8]) -> Self {
    Self::from_be_bytes(raw.try_into().unwrap())
  }

  fn write(&self, out: &mut Vec<u8>) {
    // Use big-endian to ensure lexicographic sorted is also numerically sorted.
    out.extend_from_slice(&self.to_be_bytes());
  }
}

impl DbEntKeySfxTyp for u64 {
  fn read(raw: &[u8]) -> Self {
    Self::from_be_bytes(raw.try_into().unwrap())
  }

  fn write(&self, out: &mut Vec<u8>) {
    // Use big-endian to ensure lexicographic sorted is also numerically sorted.
    out.extend_from_slice(&self.to_be_bytes());
  }
}

impl DbEntKeySfxTyp for String {
  fn read(raw: &[u8]) -> Self {
    String::from_utf8(raw.to_vec()).unwrap()
  }

  fn write(&self, out: &mut Vec<u8>) {
    out.extend_from_slice(self.as_bytes());
  }
}

impl DbEntKeySfxTyp for Vec<u8> {
  fn read(raw: &[u8]) -> Self {
    raw.to_vec()
  }

  fn write(&self, out: &mut Vec<u8>) {
    out.extend_from_slice(self);
  }
}

impl DbEntKeySfxTyp for () {
  fn read(_: &[u8]) -> Self {}

  fn write(&self, _: &mut Vec<u8>) {}
}

pub struct DbEntTyp<K: DbEntKeySfxTyp, V: DeserializeOwned + Send + Serialize + Sync> {
  pub prefix: u8,
  pub _key: PhantomData<K>,
  pub _val: PhantomData<V>,
}

impl<K: DbEntKeySfxTyp, V: DeserializeOwned + Send + Serialize + Sync> DbEntTyp<K, V> {
  pub fn key(&self, k: impl Borrow<K>) -> Vec<u8> {
    let mut key = vec![self.prefix];
    k.borrow().write(&mut key);
    key
  }

  pub fn parse_value(&self, raw: &[u8]) -> V {
    rmp_serde::from_slice(raw).unwrap()
  }

  pub fn read(&self, db: &Arc<dyn Store>, k: impl Borrow<K>) -> Option<V> {
    let key = self.key(k);
    let raw = db.get(key);
    raw.map(|raw| self.parse_value(&raw))
  }

  pub fn iter<'a>(
    &'a self,
    db: &'a Arc<dyn Store>,
  ) -> impl Iterator<Item = (K, V)> + Send + Sync + 'a {
    db.iter(self.prefix).map(|(k, v)| {
      let k = K::read(&k[1..]);
      let v = self.parse_value(&v);
      (k, v)
    })
  }

  pub fn put(&self, db: &Arc<dyn Store>, k: impl Borrow<K>, v: impl Borrow<V>) {
    let key = self.key(k);
    let val_raw = to_vec_named(v.borrow()).unwrap();
    db.put(key, val_raw);
  }

  pub fn delete(&self, db: &Arc<dyn Store>, k: impl Borrow<K>) {
    let key = self.key(k);
    db.delete(key);
  }

  pub fn batch_put(&self, txn: &mut Vec<WriteOp>, k: impl Borrow<K>, v: impl Borrow<V>) {
    let key = self.key(k);
    let val_raw = to_vec_named(v.borrow()).unwrap();
    txn.push(WriteOp::Put(key, val_raw));
  }

  pub fn batch_delete(&self, txn: &mut Vec<WriteOp>, k: impl Borrow<K>) {
    let key = self.key(k);
    txn.push(WriteOp::Delete(key));
  }
}

macro_rules! db_ent {
  ($name:ident, $prefix:expr, $key:ty, $val:ty) => {
    pub static $name: DbEntTyp<$key, $val> = DbEntTyp {
      prefix: $prefix,
      _key: PhantomData,
      _val: PhantomData,
    };
  };
}

// This is stored separately from node data as it's updated more frequently and we do a full scan every load.
db_ent!(ADD_EDGES, 1, Id, Vec<Id>);
db_ent!(CFG, 2, (), Cfg);
db_ent!(DELETED, 3, Id, ());
db_ent!(KEY_TO_ID, 4, String, Id);
db_ent!(ID_TO_KEY, 5, Id, String);
db_ent!(NODE, 6, Id, DbNodeData);
db_ent!(PQ_MODEL, 7, (), ProductQuantizer<f32>);

// We store both in one DB entry to leverage one disk page read to get both, as specified in the DiskANN paper. (If we store them as separate DB entries, they are unlikely to be stored in the same disk page.)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DbNodeData {
  // Every time the node is updated, we increment this version.
  // Right now it's used to concurrently read/insert from/into DB/cache without global locking.
  pub version: u64,
  pub neighbors: Vec<Id>,
  // We'll often share this around separate from the node data, so use separate inner Arc.
  pub vector: Arc<VecData>,
}

impl DbNodeData {
  pub fn into_vector(self) -> VecData {
    Arc::into_inner(self.vector).unwrap()
  }
}
