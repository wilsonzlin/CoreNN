use itertools::Itertools;
use schema::DbEntKeySfxTyp;
use schema::DbEntTyp;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::borrow::Borrow;

pub mod in_memory;
pub mod rocksdb;
pub mod schema;

pub enum WriteOp {
  Put(Vec<u8>, Vec<u8>),
  Delete(Vec<u8>),
}

pub trait Store: Send + Sync {
  fn get(&self, key: Vec<u8>) -> Option<Vec<u8>>;
  fn put(&self, key: Vec<u8>, value: Vec<u8>);
  fn delete(&self, key: Vec<u8>);

  fn iter(&self, prefix: u8)
    -> Box<dyn Iterator<Item = (Box<[u8]>, Box<[u8]>)> + Send + Sync + '_>;

  fn multi_get(&self, keys: Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>>;
  fn write(&self, ops: Vec<WriteOp>);
}

impl dyn Store {
  pub fn read_ents<'a, 'b, K, V, KeysItem, Keys>(
    &self,
    ent: &'static DbEntTyp<K, V>,
    keys: Keys,
  ) -> impl IntoIterator<Item = Option<V>> + 'b
  where
    K: DbEntKeySfxTyp,
    V: DeserializeOwned + Send + Serialize + Sync,
    KeysItem: Borrow<K>,
    Keys: IntoIterator<Item = KeysItem> + 'a,
  {
    let raws = self.multi_get(keys.into_iter().map(|k| ent.key(k)).collect_vec());
    raws
      .into_iter()
      .map(|raw| raw.map(|raw| ent.parse_value(&raw)))
  }
}
