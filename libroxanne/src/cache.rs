use crate::common::Id;
use crate::compressor::Compressor;
use crate::compressor::CV;
use crate::store::schema::DbNodeData;
use crate::store::schema::NODE;
use crate::store::Store;
use dashmap::DashMap;
use dashmap::Entry;
use itertools::Itertools;
use std::iter::zip;
use std::sync::Arc;

/// In-memory cache of DbNodeData or CV values.
/// Both lazily read and parse DB node entries, so this is a common struct with common code.
/// However, refer to NodeCache/CVCache and new_node_cache/new_cv_cache; this common type is more internal deduplication,
/// than establishing an intended public common API/concept/utility.
pub struct Cache<T: Cacheable> {
  db: Arc<dyn Store>,
  cache: DashMap<Id, T>,
  transformer: Box<dyn CacheableTransformer<T>>,
}

impl<T: Cacheable> Cache<T> {
  fn new(db: Arc<dyn Store>, transformer: impl CacheableTransformer<T> + 'static) -> Self {
    Self {
      db,
      cache: DashMap::new(),
      transformer: Box::new(transformer),
    }
  }

  pub fn contains(&self, id: Id) -> bool {
    self.cache.contains_key(&id)
  }

  pub fn insert(&self, id: Id, node: T) {
    match self.cache.entry(id) {
      Entry::Vacant(e) => {
        e.insert(node);
      }
      Entry::Occupied(mut e) => {
        if !T::VERSIONED || e.get().version() < node.version() {
          e.insert(node);
        }
      }
    }
  }

  pub fn remove(&self, id: Id) {
    self.cache.remove(&id);
  }

  pub fn get(&self, id: Id) -> Option<T> {
    self.cache.get(&id).map(|n| n.clone())
  }

  pub fn multi_get(&self, ids: &[Id]) -> Vec<Option<T>> {
    let to_read_ids = ids
      .iter()
      .copied()
      .filter(|&id| !self.contains(id))
      .collect_vec();
    let db_nodes = self.db.read_ents(&NODE, to_read_ids.iter());
    for (id, node) in zip(to_read_ids, db_nodes) {
      let Some(node) = node else {
        continue;
      };
      self.insert(id, self.transformer.transform(node));
    }
    ids.iter().map(|&id| self.get(id)).collect_vec()
  }
}

pub type NodeCache = Cache<Arc<DbNodeData>>;
pub type CVCache = Cache<CV>;

pub fn new_node_cache(db: Arc<dyn Store>) -> NodeCache {
  Cache::new(db, NoTransform)
}

pub fn new_cv_cache(db: Arc<dyn Store>, compressor: Arc<dyn Compressor>) -> CVCache {
  Cache::new(db, compressor)
}

/// This is an internal trait, used only for the internal implementation of Cache.
/// It's deduplication, not part of the public API.
/// Internally: this allows Cache to process DbNodeData and CVs using the same code.
pub trait Cacheable: Clone {
  const VERSIONED: bool;
  fn version(&self) -> u64 {
    0
  }
}

impl Cacheable for Arc<DbNodeData> {
  const VERSIONED: bool = true;

  fn version(&self) -> u64 {
    self.version
  }
}

impl Cacheable for CV {
  const VERSIONED: bool = false;
}

/// This is an internal trait, used only for the internal implementation of Cache.
/// It's deduplication, not part of the public API.
/// Internally: this allows Cache to process CVs specifically while retaining 99% of the same code.
pub trait CacheableTransformer<T: Cacheable>: Send + Sync {
  fn transform(&self, node: DbNodeData) -> T;
}

struct NoTransform;

impl CacheableTransformer<Arc<DbNodeData>> for NoTransform {
  fn transform(&self, node: DbNodeData) -> Arc<DbNodeData> {
    Arc::new(node)
  }
}
