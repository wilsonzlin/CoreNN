use super::Store;
use super::WriteOp;
use dashmap::DashMap;

pub struct InMemoryStore {
  data: DashMap<Vec<u8>, Vec<u8>>,
}

impl InMemoryStore {
  pub fn new() -> Self {
    Self {
      data: DashMap::new(),
    }
  }
}

impl Store for InMemoryStore {
  fn get(&self, key: Vec<u8>) -> Option<Vec<u8>> {
    self.data.get(&key).map(|v| v.clone())
  }

  fn put(&self, key: Vec<u8>, value: Vec<u8>) {
    self.data.insert(key, value);
  }

  fn delete(&self, key: Vec<u8>) {
    self.data.remove(&key);
  }

  fn iter(
    &self,
    prefix: u8,
  ) -> Box<dyn Iterator<Item = (Box<[u8]>, Box<[u8]>)> + Send + Sync + '_> {
    Box::new(
      self
        .data
        .iter()
        .filter(move |e| e.key().starts_with(&[prefix]))
        .map(|e| {
          (
            e.key().clone().into_boxed_slice(),
            e.value().clone().into_boxed_slice(),
          )
        }),
    )
  }

  fn multi_get(&self, keys: Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>> {
    keys.into_iter().map(|key| self.get(key)).collect()
  }

  fn write(&self, ops: Vec<WriteOp>) {
    for op in ops {
      match op {
        WriteOp::Put(key, value) => self.put(key, value),
        WriteOp::Delete(key) => self.delete(key),
      }
    }
  }
}
