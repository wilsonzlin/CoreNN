#![feature(f16)]
#![feature(path_add_extension)]

use ahash::HashSet;
use ahash::HashSetExt;
use arbitrary_lock::ArbitraryLock;
use bf::BruteForceIndex;
use blob::BlobStore;
use cfg::RoxanneDbCfg;
use common::nan_to_num;
use common::Dtype;
use common::Id;
use common::Metric;
use common::PointDist;
use common::PrecomputedDists;
use dashmap::DashMap;
use dashmap::DashSet;
use dashmap::Entry;
use db::Db;
use db::DbIndexMode;
use db::DbTransaction;
use db::NodeData;
use in_memory::InMemoryIndex;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use parking_lot::Mutex;
use parking_lot::RwLock;
use pq::ProductQuantizer;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use search::GreedySearchable;
use search::Query;
use std::cmp::max;
use std::cmp::min;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use vamana::Vamana;
use vamana::VamanaParams;

pub mod bf;
pub mod blob;
pub mod cfg;
pub mod common;
pub mod db;
pub mod hnsw;
pub mod in_memory;
pub mod pq;
pub mod queue;
pub mod search;
pub mod vamana;

pub enum RoxanneDbError {
  Busy,
}

struct PqState<T: Dtype> {
  db: Arc<Db>,
  pq_vecs: DashMap<Id, Array1<u8>>,
  pq: ProductQuantizer<T>,
}

impl<T: Dtype> PqState<T> {
  pub fn get_pq_vec(&self, id: Id) -> Array1<T> {
    // This holds a write lock on the entry, so multiple callers won't do repeated redundant calculations.
    match self.pq_vecs.entry(id) {
      Entry::Occupied(o) => {
        let pq_vec = o.get();
        self.pq.decode_1(&pq_vec.view())
      }
      Entry::Vacant(v) => {
        let node = self.db.read_node(id);
        let vec = Array1::from_vec(node.vector);
        let pq_vec = self.pq.encode_1(&vec.view());
        // Insert into cache.
        v.insert(pq_vec);
        vec
      }
    }
  }
}

enum Mode<T: Dtype> {
  InMemory {
    graph: Arc<DashMap<Id, Vec<Id>>>,
    vectors: Arc<DashMap<Id, Array1<T>>>,
  },
  LTI {
    pq: PqState<T>,
  },
}

struct Index<T: Dtype> {
  // We use a brute force index for the first few points, because:
  // - we want a decent sized sample to calculate the medoid
  // - it's fast enough to query
  // - inserting into a Vamana graph with zero or only a few nodes has complications (e.g. RobustPrune may actually make a node unreachable)
  bf: RwLock<Option<Arc<BruteForceIndex<T>>>>,
  db: Arc<Db>,
  mode: RwLock<Mode<T>>,
  // This changes when we transition from BF to InMemory.
  medoid: AtomicUsize,
  metric: Metric<T>,
  vamana_params: VamanaParams,
  additional_out_neighbors: DashMap<Id, Vec<Id>>,
  additional_edge_count: AtomicUsize,
  // Only used for a temporary short period during updater_thread when inserting new vectors.
  temp_vecs: DashMap<Id, Array1<T>>,
  temp_out_neighbors: DashMap<Id, Vec<Id>>,
}

impl<'a, T: Dtype> GreedySearchable<'a, T> for Index<T> {
  type FullVec = Array1<T>;
  type Neighbors = Vec<Id>;
  type Point = Array1<T>;

  fn medoid(&self) -> Id {
    self.medoid.load(Ordering::Relaxed)
  }

  fn metric(&self) -> Metric<T> {
    self.metric
  }

  fn get_point(&'a self, id: Id) -> Self::Point {
    if let Some(v) = self.temp_vecs.get(&id) {
      return v.clone();
    }
    match &*self.mode.read() {
      Mode::InMemory { vectors, .. } => vectors.get(&id).unwrap().clone(),
      Mode::LTI { pq } => pq.get_pq_vec(id),
    }
  }

  fn get_out_neighbors(&'a self, id: Id) -> (Self::Neighbors, Option<Self::FullVec>) {
    let temp = self.temp_out_neighbors.get(&id).map(|e| e.clone());
    let mut res = match &*self.mode.read() {
      Mode::InMemory { graph, .. } => {
        // TODO Avoid clones.
        (
          temp.unwrap_or_else(|| graph.get(&id).unwrap().clone()),
          None,
        )
      }
      Mode::LTI { .. } => {
        let n = self.db.read_node(id);
        (temp.unwrap_or(n.neighbors), Some(Array1::from(n.vector)))
      }
    };
    if let Some(add) = self.additional_out_neighbors.get(&id) {
      res.0.extend(add.iter());
    }
    res
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }
}

// We only implement Vamana to be able to use compute_robust_pruned during updater_thread.
// We don't use set_* as the update process is more sophisticated (see updater_thread).
impl<'a, T: Dtype> Vamana<'a, T> for Index<T> {
  fn params(&self) -> &vamana::VamanaParams {
    &self.vamana_params
  }

  fn set_point(&self, _id: Id, _point: Array1<T>) {
    unreachable!()
  }

  fn set_out_neighbors(&self, _id: Id, _neighbors: Vec<Id>) {
    unreachable!()
  }
}

pub struct RoxanneDb<T: Dtype> {
  blobs: BlobStore,
  cfg: RoxanneDbCfg,
  db: Arc<Db>,
  deleted: DashSet<Id>,
  index: Index<T>,
  key_locks: ArbitraryLock<String, Mutex<()>>,
  next_id: AtomicUsize,
  update_sender: Sender<(Vec<Id>, Vec<Array1<T>>, Sender<()>)>,
}

impl<T: Dtype> RoxanneDb<T> {
  // Why do all updates from a single thread (i.e. serialized), instead of only the compaction process?
  // Because our Vamana implementation doesn't support parallel updates (it does batching instead), so a lot of complexity to split out insertion (thread-safe) and compaction (single thread) ultimately ends up being pointless. It's safer to get correct; if we need to optimize, we can profile in the future.
  // NOTE: Many operations may seem incorrect due to only acquiring read locks (so it seems like changes could happen beneath our feet and we're not correctly looking at a stable consistent snapshot/view of the entire system/data), but remember that all updates are processed serially within this sole thread only.
  fn updater_thread(
    self: &Arc<RoxanneDb<T>>,
    receiver: Receiver<(Vec<Id>, Vec<Array1<T>>, Sender<()>)>,
  ) {
    while let Ok(mut batch) = receiver.recv() {
      // Collect more if available.
      let mut batch_signals = Vec::new();
      while let Ok(e) = receiver.try_recv() {
        batch.0.extend(e.0);
        batch.1.extend(e.1);
        batch_signals.push(e.2);
      }
      let (batch_ids, batch_vecs, _) = batch;

      // BruteForceIndex is wrapped in Arc so we can work on it while dropping the lock (so we can potentially replace it).
      if let Some(index) = self.index.bf.read().clone() {
        batch_ids.par_iter().zip(batch_vecs).for_each(|(&id, v)| {
          index.insert(id, v);
        });
        if index.len() > self.cfg.brute_force_index_cap {
          let ids = index.vectors().iter().map(|e| *e.key()).collect_vec();
          let vecs = index
            .vectors()
            .iter()
            .map(|e| e.value().clone())
            .collect_vec();
          let ann = InMemoryIndex::builder(ids.clone(), vecs)
            .degree_bound(self.cfg.degree_bound)
            .distance_threshold(self.cfg.distance_threshold)
            .metric(self.index.metric)
            .update_batch_size(self.cfg.update_batch_size)
            .update_search_list_cap(self.cfg.update_search_list_cap)
            .build();
          let mut txn = DbTransaction::new();
          txn.write_index_mode(DbIndexMode::InMemory);
          txn.write_medoid(ann.medoid());
          for id in ids {
            txn.delete_brute_force_vec(id);
            txn.write_node(id, &NodeData {
              neighbors: ann.graph.get(&id).unwrap().clone(),
              vector: ann.vectors.get(&id).unwrap().to_vec(),
            });
          }
          txn.commit(&self.db);
          *self.index.bf.write() = None;
          self.index.medoid.store(ann.medoid(), Ordering::Relaxed);
          *self.index.mode.write() = Mode::InMemory {
            graph: ann.graph,
            vectors: ann.vectors,
          };
        };
        continue;
      };

      let txn = Mutex::new(DbTransaction::new());
      let touched_nodes = DashSet::new();
      batch_ids
        .into_par_iter()
        .zip(&batch_vecs)
        .for_each(|(id, v)| {
          // CORRECTNESS: later compute_robust_pruned calls from other nodes (when adding backedges) will fetch this vector, so insert this now.
          self.index.temp_vecs.insert(id, v.clone());
          let mut candidates = HashSet::new();
          self.index.greedy_search(
            Query::Vec(&v.view()),
            1,
            self.index.vamana_params.update_search_list_cap,
            1,
            self.index.medoid.load(Ordering::Relaxed),
            |n| !self.deleted.contains(&n.id),
            Some(&mut candidates),
            None,
            None,
          );
          let neighbors = self
            .index
            .compute_robust_pruned(Query::Vec(&v.view()), candidates);
          // We need to insert now because once we add to additional_out_neighbors, some queries may reach this new node and request its neighbors (and we haven't inserted it into the index yet).
          // CORRECTNESS: our new `neighbors` can only contain current nodes and none from this insertion batch, so it's safe to expand any of the nodes in `neighbors` right now.
          self.index.temp_out_neighbors.insert(id, neighbors.clone());
          for &j in neighbors.iter() {
            self
              .index
              .additional_out_neighbors
              .entry(j)
              .or_default()
              .push(id);
            self
              .index
              .additional_edge_count
              .fetch_add(1, Ordering::Relaxed);
            touched_nodes.insert(j);
          }
          txn.lock().write_node(id, &NodeData {
            neighbors,
            vector: v.to_vec(),
          });
        });
      touched_nodes.into_par_iter().for_each(|id| {
        let (mut new_neighbors, full_vec) = self.index.get_out_neighbors(id);
        let add_neighbors = self.index.additional_out_neighbors.get(&id).unwrap();
        new_neighbors.extend(add_neighbors.iter());
        if new_neighbors.len() > self.cfg.max_degree_bound {
          // At this point, compute_robust_pruned will likely look up the vectors for our newly inserted vectors, which is why we have temp_vecs.
          new_neighbors = self
            .index
            .compute_robust_pruned(Query::Id(id), new_neighbors);
          let mut txn = txn.lock();
          txn.write_node(id, &NodeData {
            neighbors: new_neighbors,
            // If LTI, full_vec will be Some; if in-memory, use get_point.
            vector: full_vec
              .unwrap_or_else(|| self.index.get_point(id))
              .to_vec(),
          });
          txn.delete_additional_out_neighbors(id);
          // Technically, this affects queries because we haven't committed the new NodeData yet; in reality, it's minor (very short period) and shouldn't matter.
          self.index.additional_edge_count.fetch_sub(
            self
              .index
              .additional_out_neighbors
              .remove(&id)
              .unwrap()
              .1
              .len(),
            Ordering::Relaxed,
          );
        } else {
          let mut txn = txn.lock();
          txn.write_additional_out_neighbors(id, &add_neighbors);
        }
      });
      txn.into_inner().commit(&self.db);
      self.index.temp_vecs.clear();
      self.index.temp_out_neighbors.clear();
      for c in batch_signals {
        c.send(()).unwrap();
      }

      if self.deleted.len() < self.cfg.merge_threshold_deletes
        && self.index.additional_edge_count.load(Ordering::Relaxed)
          < self.cfg.merge_threshold_additional_edges
      {
        // No need to merge yet.
        continue;
      };

      // From now on, we must work with a consistent, immutable snapshot of deleted elements.
      let deleted = self.deleted.iter().map(|e| *e).collect::<HashSet<_>>();

      let new_mode = match &*self.index.mode.read() {
        Mode::InMemory { vectors, .. } => {
          // Since we've never built a LTI before, we need to build the PQ now.
          let ss = min(vectors.len(), self.cfg.pq_sample_size);
          let mut mat = Array2::zeros((ss, self.cfg.dim));
          for (i, vec) in vectors
            .iter()
            .choose_multiple(&mut thread_rng(), ss)
            .into_iter()
            .enumerate()
          {
            mat.row_mut(i).assign(&*vec);
          }
          let pq = ProductQuantizer::train(&mat.view(), self.cfg.pq_subspaces);
          self.blobs.write_pq_model(&pq);
          // Free memory now.
          drop(mat);

          let mut txn = DbTransaction::new();
          txn.write_index_mode(DbIndexMode::LongTerm);
          txn.commit(&self.db);

          Some(Mode::LTI {
            pq: PqState {
              db: self.db.clone(),
              pq_vecs: DashMap::new(),
              pq,
            },
          })
        }
        Mode::LTI { .. } => None,
      };

      // In RocksDB, iterators view a snapshot of the entire DB at the time of iterator creation, so we can safely modify DB entries during iteration. https://github.com/facebook/rocksdb/wiki/Iterator
      self
        .db
        .iter_nodes::<T>()
        .par_bridge()
        .for_each(|(id, node)| {
          let mut deleted_neighbors = Vec::new();
          let mut new_neighbors = DashSet::new();
          for &n in node.neighbors.iter() {
            if deleted.contains(&n) {
              deleted_neighbors.push(n);
            } else {
              new_neighbors.insert(n);
            };
          }
          let add = self.index.additional_out_neighbors.remove(&id).map(|e| e.1);
          if add.is_none() && deleted_neighbors.is_empty() {
            // Node is untouched.
            return;
          };
          if let Some(add) = add {
            self
              .index
              .additional_edge_count
              .fetch_sub(add.len(), Ordering::Relaxed);
            new_neighbors.extend(add);
          }
          // TODO I/O parallelism using Rayon: no no.
          deleted_neighbors.par_iter().for_each(|&n_id| {
            for n in self.db.read_node::<T>(n_id).neighbors {
              if !deleted.contains(&n) {
                new_neighbors.insert(n);
              };
            }
          });
          let new_neighbors = self
            .index
            .compute_robust_pruned(Query::Id(id), new_neighbors);

          let mut txn = DbTransaction::new();
          txn.write_node(id, &NodeData {
            neighbors: new_neighbors,
            vector: node.vector,
          });
          txn.delete_additional_out_neighbors(id);
          txn.commit(&self.db);
        });

      let mut txn = DbTransaction::new();
      for &id in deleted.iter() {
        txn.delete_deleted(id);
        txn.delete_additional_out_neighbors(id);
        txn.delete_node(id);
      }
      txn.commit(&self.db);

      for id in deleted {
        self.deleted.remove(&id);
      }

      if let Some(m) = new_mode {
        *self.index.mode.write() = m;
      }
    }
  }
}

impl<T: Dtype> RoxanneDb<T> {
  pub fn open(dir: PathBuf) -> Arc<RoxanneDb<T>> {
    let blobs = BlobStore::open(dir.join("blobs"));
    let cfg: RoxanneDbCfg =
      toml::from_str(&fs::read_to_string(dir.join("roxanne.toml")).unwrap()).unwrap();
    let metric = cfg.metric.get_fn::<T>();
    let vamana_params = VamanaParams {
      degree_bound: cfg.degree_bound,
      distance_threshold: cfg.distance_threshold,
      update_batch_size: cfg.update_batch_size,
      update_search_list_cap: cfg.update_search_list_cap,
    };

    let db = Arc::new(Db::open(dir.join("db")));
    let mut next_id = 0;
    let deleted = DashSet::new();
    for id in db.iter_deleted() {
      next_id = max(id + 1, next_id);
      deleted.insert(id);
    }
    for (id, _) in db.iter_ids() {
      next_id = max(id + 1, next_id);
    }

    let additional_out_neighbors = DashMap::new();
    let additional_edge_count = AtomicUsize::new(0);
    db.iter_additional_out_neighbors()
      .par_bridge()
      .for_each(|(id, add)| {
        additional_edge_count.fetch_add(add.len(), Ordering::Relaxed);
        additional_out_neighbors.insert(id, add);
      });

    let mode = db.read_index_mode();

    let index = Index {
      bf: RwLock::new(if mode == DbIndexMode::BruteForce {
        let bf = BruteForceIndex::new(metric);
        db.iter_brute_force_vecs::<T>()
          .par_bridge()
          .for_each(|(id, vec)| {
            bf.insert(id, Array1::from_vec(vec));
          });
        Some(Arc::new(bf))
      } else {
        None
      }),
      db: db.clone(),
      // In BruteForce mode (e.g. init), there is no medoid.
      medoid: db.maybe_read_medoid().unwrap_or(Id::MAX).into(),
      metric,
      vamana_params,
      additional_out_neighbors,
      additional_edge_count,
      temp_vecs: Default::default(),
      temp_out_neighbors: Default::default(),
      mode: RwLock::new(if mode == DbIndexMode::LongTerm {
        let pq = blobs.read_pq_model();
        Mode::LTI {
          pq: PqState {
            db: db.clone(),
            pq,
            pq_vecs: DashMap::new(),
          },
        }
      } else {
        // This also handles BruteForce mode (this'll be empty.)
        let graph = Arc::new(DashMap::new());
        let vectors = Arc::new(DashMap::new());
        db.iter_nodes().par_bridge().for_each(|(id, n)| {
          graph.insert(id, n.neighbors);
          vectors.insert(id, Array1::from_vec(n.vector));
        });
        Mode::InMemory { graph, vectors }
      }),
    };

    let (send, recv) = std::sync::mpsc::channel();
    let roxanne = Arc::new(RoxanneDb {
      blobs,
      cfg,
      db,
      deleted,
      index,
      key_locks: ArbitraryLock::new(),
      next_id: AtomicUsize::new(next_id),
      update_sender: send,
    });
    thread::spawn({
      let roxanne = roxanne.clone();
      move || roxanne.updater_thread(recv)
    });
    roxanne
  }

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    match self.index.bf.read().as_ref() {
      Some(bf) => bf.query(query, k, |n| !self.deleted.contains(&n)),
      None => self.index.greedy_search(
        Query::Vec(query),
        k,
        self.cfg.query_search_list_cap,
        self.cfg.beam_width,
        self.index.medoid(),
        |n| !self.deleted.contains(&n.id),
        None,
        None,
        None,
      ),
    }
  }

  // Considerations:
  // - We don't want to keep piling on to a brute force index (e.g. while compactor struggles to keep up) and degrade query performance exponentially.
  // - We don't want to keep buffering vectors in memory, acknowledging inserts but in reality there's backpressure while the graph is being updated.
  // - Both above points means that we should have some mechanism of returning a "busy" signal to the caller, instead of just accepting into unbounded memory or blocking the caller.
  // - We don't want to perform any compaction within the function (instead running it in a background thread), as otherwise that negatively affects the insertion call latency for the one unlucky caller.
  // Require a Map to ensure there are no duplicate keys (which would otherwise complicate insertion).
  pub fn insert(&self, entries: BTreeMap<String, Array1<T>>) -> Result<(), RoxanneDbError> {
    let n = entries.len();

    let ids_base = self.next_id.fetch_add(n, Ordering::Relaxed);
    let ids = (0..n).map(|i| ids_base + i).collect_vec();

    let mut entries = entries.into_iter().collect_vec();

    entries.par_iter_mut().enumerate().for_each(|(i, (k, v))| {
      // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
      v.mapv_inplace(nan_to_num);

      let id = ids_base + i;
      let locker = self.key_locks.get(k.clone());
      let _lock = locker.lock();
      // TODO FIX/DOCUMENT: For simplicity and performance reasons, the deletion of an existing vector with the same key is **not** atomic; always retry inserts. The alternative seems to be to hold a lock on the key for the entirety of its insert in updater_thread.
      let mut txn = DbTransaction::new();
      if let Some(ex_id) = self.db.maybe_read_id(&k) {
        txn.write_deleted(ex_id);
        txn.delete_key(ex_id);
        self.deleted.insert(id);
      };
      // CORRECTNESS: even though we haven't actually inserted anything at ID, it's safe to commit the key <-> ID link now, as:
      // - We don't support retrieving vectors by key.
      // - If it gets marked as deleted later, that's fine as our algorithm doesn't expect delete-marked nodes to exist (even during merging).
      // - No result will come back with this ID.
      txn.write_key(id, &k);
      txn.write_id(&k, id);
      txn.commit(&self.db);
    });

    let (tx, rx) = std::sync::mpsc::channel();
    self
      .update_sender
      .send((ids, entries.into_iter().map(|e| e.1).collect(), tx))
      .unwrap();
    rx.recv().unwrap();
    Ok(())
  }

  pub fn delete(&self, key: &str) -> Result<(), RoxanneDbError> {
    let locker = self.key_locks.get(key.to_string());
    let lock = locker.lock();
    let mut txn = DbTransaction::new();
    let id = self.db.maybe_read_id(key);
    if let Some(id) = id {
      txn.write_deleted(id);
      txn.delete_id(key);
      txn.delete_key(id);
    };
    txn.commit(&self.db);
    drop(lock);

    if let Some(id) = id {
      self.deleted.insert(id);
    };

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use crate::cfg::RoxanneDbCfg;
  use crate::RoxanneDb;
  use itertools::Itertools;
  use ndarray::Array1;
  use rand::thread_rng;
  use rand::Rng;
  use std::fs;
  use std::path::PathBuf;

  #[test]
  fn test_roxanne() {
    // Parameters.
    let dir = PathBuf::from("/dev/shm/roxannedb-test");
    let dim = 512;
    let pq_subspaces = 64;
    let gen = || {
      Array1::from_vec(
        (0..dim)
          .map(|_| thread_rng().gen_range(-1.0..1.0))
          .collect_vec(),
      )
    };

    // Create and open DB.
    if fs::exists(&dir).unwrap() {
      fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir(&dir).unwrap();
    fs::write(
      dir.join("roxanne.toml"),
      toml::to_string(&RoxanneDbCfg {
        dim,
        pq_subspaces,
        ..Default::default()
      })
      .unwrap(),
    )
    .unwrap();
    let db = RoxanneDb::open(dir);

    // First test: an empty DB should provide no results.
    let res = db.query(&gen().view(), 100);
    assert_eq!(res.len(), 0);
  }
}
