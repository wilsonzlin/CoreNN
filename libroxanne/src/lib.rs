#![feature(duration_millis_float)]
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
pub mod util;

#[derive(Debug)]
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
  temp_nodes: DashMap<Id, (Vec<Id>, Array1<T>)>,
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
    if let Some(temp) = self.temp_nodes.get(&id) {
      // TODO Avoid clone.
      return temp.1.clone();
    }
    match &*self.mode.read() {
      Mode::InMemory { vectors, .. } => vectors.get(&id).unwrap().clone(),
      Mode::LTI { pq } => pq.get_pq_vec(id),
    }
  }

  fn get_out_neighbors(&'a self, id: Id) -> (Self::Neighbors, Option<Self::FullVec>) {
    if let Some(temp) = self.temp_nodes.get(&id) {
      // We don't need to add additional_out_neighbors, as temp_nodes is a full override.
      // We don't need to return a full vector as it's not from the disk.
      // TODO Avoid clone.
      return (temp.0.clone(), None);
    }
    let mut res = match &*self.mode.read() {
      Mode::InMemory { graph, .. } => {
        // TODO Avoid clones.
        (graph.get(&id).unwrap().clone(), None)
      }
      Mode::LTI { .. } => {
        let n = self.db.read_node(id);
        (n.neighbors, Some(Array1::from(n.vector)))
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
      let mut batch_signals = vec![batch.2];
      while let Ok(e) = receiver.try_recv() {
        batch.0.extend(e.0);
        batch.1.extend(e.1);
        batch_signals.push(e.2);
      }
      let batch_n = batch.0.len();
      let (batch_ids, batch_vecs, _) = batch;
      tracing::debug!(n = batch_n, "received batch");

      // If we are in brute force index mode, then handle differently (it's not a graph, so update process is different).
      // BruteForceIndex is wrapped in Arc so we can work on it while dropping the lock (so we can potentially replace it).
      // WARNING: Do not inline this to the `if` RHS; in Rust, given `a.b.c`, `a` is dropped but `b` is not, so if we inline this, we'll hold the lock.
      let bf = self.index.bf.read().clone();
      if let Some(index) = bf {
        tracing::debug!(current_size = index.len(), "brute force index exists");
        batch_ids.par_iter().zip(batch_vecs).for_each(|(&id, v)| {
          index.insert(id, v);
        });
        if index.len() > self.cfg.brute_force_index_cap {
          tracing::info!(
            vectors = index.len(),
            "transitioning brute force index to in-memory index"
          );
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
          tracing::info!(vectors = index.len(), "built optimized in-memory index");
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
          tracing::info!(vectors = index.len(), "transitioned to in-memory index");
        };
        for c in batch_signals {
          c.send(()).unwrap();
        }
        continue;
      };

      let txn = Mutex::new(DbTransaction::new());
      let touched_nodes = DashSet::new();
      batch_ids
        .into_par_iter()
        .zip(batch_vecs)
        .for_each(|(id, v)| {
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
          match &*self.index.mode.read() {
            Mode::InMemory { graph, vectors } => {
              graph.insert(id, neighbors.clone());
              vectors.insert(id, v.clone());
            }
            Mode::LTI { pq } => {
              // We need to insert now because once we add to additional_out_neighbors, some queries may reach this new node and request its neighbors (and we haven't inserted it into the index yet). Also, later compute_robust_pruned calls from other nodes (when adding backedges) will fetch this vector, so insert into `temp_nodes` now.
              // CORRECTNESS: our new `neighbors` can only contain current nodes and none from this insertion batch, so it's safe to expand any of the nodes in `neighbors` right now.
              self
                .index
                .temp_nodes
                .insert(id, (neighbors.clone(), v.clone()));
              // We may as well pre-populate the cache now instead of reading from disk again later.
              pq.pq_vecs.insert(id, pq.pq.encode_1(&v.view()));
            }
          }
          for &j in neighbors.iter() {
            // `id` cannot have existed in any existing out-neighbors of any node as it has just been created, so we don't need to check if it doesn't already exist in `out(j)` first.
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
      let flushed_adds_count = AtomicUsize::new(0);
      let touched_count = touched_nodes.len();
      touched_nodes.into_par_iter().for_each(|id| {
        let (mut new_neighbors, full_vec) = self.index.get_out_neighbors(id);
        // Clone so we can remove later if necessary (otherwise we'll deadlock).
        let add_neighbors = self
          .index
          .additional_out_neighbors
          .get(&id)
          .unwrap()
          .clone();
        new_neighbors.extend(add_neighbors.iter());
        if new_neighbors.len() <= self.cfg.max_degree_bound {
          // Technically, we can always update the graph node's out neighbors if we're using in-memory index, but for a consistent code path we'll stick with using additional_out_neighbors even for in-memory.
          let mut txn = txn.lock();
          txn.write_additional_out_neighbors(id, &add_neighbors);
        } else {
          flushed_adds_count.fetch_add(1, Ordering::Relaxed);
          // At this point, compute_robust_pruned will likely look up the vectors for our newly inserted vectors, which is why we have temp_nodes.
          new_neighbors = self
            .index
            .compute_robust_pruned(Query::Id(id), new_neighbors);
          let mut txn = txn.lock();
          txn.write_node(id, &NodeData {
            neighbors: new_neighbors.clone(),
            // If LTI, full_vec will be Some; if in-memory, use get_point.
            vector: full_vec
              .unwrap_or_else(|| self.index.get_point(id))
              .to_vec(),
          });
          txn.delete_additional_out_neighbors(id);
          drop(txn);
          if let Mode::InMemory { graph, .. } = &*self.index.mode.read() {
            graph.insert(id, new_neighbors);
          };
          // Technically, this affects LTI queries because we haven't committed the new NodeData yet; in reality, it's minor (very short period) and shouldn't matter.
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
        }
      });
      txn.into_inner().commit(&self.db);
      self.index.temp_nodes.clear();
      for c in batch_signals {
        c.send(()).unwrap();
      }
      tracing::debug!(
        n = batch_n,
        flushed_adds = flushed_adds_count.load(Ordering::Relaxed),
        touched = touched_count,
        "inserted vectors"
      );

      // Opportunity to transition from in-memory to long term index.
      let new_mode = match &*self.index.mode.read() {
        Mode::InMemory { vectors, .. } => {
          if vectors.len() <= self.cfg.in_memory_index_cap {
            // The subsequent code is StreamingMerge, which we don't (yet) support for in memory indices (only the on-disk LTI).
            // TODO Should we support it?
            continue;
          };
          tracing::info!("transitioning in-memory index to long term index");
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
          tracing::info!(
            sample_inputs = ss,
            subspaces = self.cfg.pq_subspaces,
            "trained PQ"
          );
          // Free memory now.
          drop(mat);

          let mut txn = DbTransaction::new();
          txn.write_index_mode(DbIndexMode::LongTerm);
          txn.commit(&self.db);

          // We may as well populate the cache now, instead of reading from disk again later.
          let pq_vecs = DashMap::new();
          vectors.par_iter().for_each(|e| {
            let (&id, vec) = e.pair();
            pq_vecs.insert(id, pq.encode_1(&vec.view()));
          });

          // We hold a read lock in this match arm, so we can't update yet.
          Some(Mode::LTI {
            pq: PqState {
              db: self.db.clone(),
              pq_vecs,
              pq,
            },
          })
        }
        _ => None,
      };
      if let Some(m) = new_mode {
        *self.index.mode.write() = m;
        tracing::info!("transitioned to long term index");
        continue;
      };

      if self.deleted.len() < self.cfg.merge_threshold_deletes
        && self.index.additional_edge_count.load(Ordering::Relaxed)
          < self.cfg.merge_threshold_additional_edges
      {
        // No need to merge yet.
        continue;
      };

      // From now on, we must work with a consistent snapshot of deleted elements.
      let deleted = self
        .deleted
        .iter()
        .map(|e| *e)
        .filter(|&e| e != self.index.medoid.load(Ordering::Relaxed))
        .collect::<HashSet<_>>();

      tracing::info!(
        deleted = deleted.len(),
        additional_edges = self.index.additional_edge_count.load(Ordering::Relaxed),
        "merging",
      );

      // In RocksDB, iterators view a snapshot of the entire DB at the time of iterator creation, so we can safely modify DB entries during iteration. https://github.com/facebook/rocksdb/wiki/Iterator
      let touched = AtomicUsize::new(0);
      self
        .db
        .iter_nodes::<T>()
        .par_bridge()
        .for_each(|(id, node)| {
          if deleted.contains(&id) {
            return;
          };

          let mut deleted_neighbors = Vec::new();
          let new_neighbors = DashSet::new();
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
          touched.fetch_add(1, Ordering::Relaxed);
          if let Some(add) = add {
            self
              .index
              .additional_edge_count
              .fetch_sub(add.len(), Ordering::Relaxed);
            for n in add {
              if !deleted.contains(&n) {
                new_neighbors.insert(n);
              };
            }
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

      let Mode::LTI { pq } = &*self.index.mode.read() else {
        unreachable!();
      };
      for &id in deleted.iter() {
        self.deleted.remove(&id);
        pq.pq_vecs.remove(&id);
        if let Some(add) = self.index.additional_out_neighbors.remove(&id) {
          self
            .index
            .additional_edge_count
            .fetch_sub(add.1.len(), Ordering::Relaxed);
        };
      }
      tracing::info!(
        touched = touched.load(Ordering::Relaxed),
        deleted = deleted.len(),
        "merge complete",
      );
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
      temp_nodes: DashMap::new(),
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

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<(String, f64)> {
    let res = match self.index.bf.read().as_ref() {
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
    };
    // TODO Using rayon for I/O, tsk tsk tsk.
    res
      .into_par_iter()
      // A node may have been deleted during the query (already collected, so didn't get filtered), or literally just after the end of the query but before here.
      // TODO DOCUMENT: it's possible to get less than k for the above reason.
      .filter_map(|r| self.db.maybe_read_key(r.id).map(|k| (k, r.dist)))
      .collect()
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
    let id = self.db.maybe_read_id(key);
    if let Some(id) = id {
      let mut txn = DbTransaction::new();
      txn.write_deleted(id);
      txn.delete_id(key);
      txn.delete_key(id);
      txn.commit(&self.db);
    };
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
  use crate::common::metric_euclidean;
  use crate::common::StdMetric;
  use crate::db::DbIndexMode;
  use crate::in_memory::calc_approx_medoid;
  use crate::Mode;
  use crate::RoxanneDb;
  use ahash::HashSet;
  use dashmap::DashSet;
  use itertools::Itertools;
  use ndarray::Array1;
  use ordered_float::OrderedFloat;
  use rand::seq::IteratorRandom;
  use rand::thread_rng;
  use rand::Rng;
  use rayon::iter::IntoParallelIterator;
  use rayon::iter::IntoParallelRefIterator;
  use rayon::iter::ParallelBridge;
  use rayon::iter::ParallelIterator;
  use std::fs;
  use std::iter::once;
  use std::path::PathBuf;
  use std::sync::atomic::Ordering;
  use std::thread::sleep;
  use std::time::Duration;
  use std::time::Instant;

  #[test]
  fn test_roxanne() {
    tracing_subscriber::fmt()
      .with_max_level(tracing::Level::DEBUG)
      .init();

    // Parameters.
    let dir = PathBuf::from("/dev/shm/roxannedb-test");
    let dim = 512;
    let degree_bound = 40;
    let max_degree_bound = 80;
    let search_list_cap = 100;
    let pq_subspaces = 256;
    let gen = || {
      Array1::from_vec(
        (0..dim)
          .map(|_| thread_rng().gen_range(-1.0..1.0))
          .collect_vec(),
      )
    };
    let n = 10_000;
    let vecs = (0..n).map(|_| gen()).collect_vec();
    let nn = (0..n)
      .into_par_iter()
      .map(|i| {
        (0..n)
          .map(|j| (j, metric_euclidean(&vecs[i].view(), &vecs[j].view())))
          .sorted_unstable_by_key(|e| OrderedFloat(e.1))
          .map(|e| e.0)
          .collect_vec()
      })
      .collect::<Vec<_>>();
    tracing::info!("calculated nearest neighbors");

    // Create and open DB.
    if fs::exists(&dir).unwrap() {
      fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir(&dir).unwrap();
    fs::write(
      dir.join("roxanne.toml"),
      toml::to_string(&RoxanneDbCfg {
        brute_force_index_cap: 1000,
        degree_bound,
        dim,
        in_memory_index_cap: 2500,
        max_degree_bound,
        merge_threshold_deletes: 200,
        metric: StdMetric::L2,
        query_search_list_cap: search_list_cap,
        update_batch_size: num_cpus::get(),
        update_search_list_cap: search_list_cap,
        pq_subspaces,
        ..Default::default()
      })
      .unwrap(),
    )
    .unwrap();
    let db = RoxanneDb::open(dir);
    tracing::info!("opened database");

    // First test: an empty DB should provide no results.
    let res = db.query(&gen().view(), 100);
    assert_eq!(res.len(), 0);

    let deleted = DashSet::new();
    let assert_accuracy = |below_i: usize, exp_acc: f64| {
      let start = Instant::now();
      let mut correct = 0;
      let mut total = 0;
      for i in 0..below_i {
        let res = db.query(&vecs[i].view(), 100);
        let got = res
          .iter()
          .map(|e| e.0.parse::<usize>().unwrap())
          .collect::<HashSet<_>>();
        let want = nn[i]
          .iter()
          .cloned()
          .filter(|&n| n < below_i && !deleted.contains(&n))
          .take(100)
          .collect::<HashSet<_>>();
        total += want.len();
        correct += want.intersection(&got).count();
      }
      let accuracy = correct as f64 / total as f64;
      let exec_ms = start.elapsed().as_millis_f64();
      tracing::info!(accuracy, exec_ms, queries = below_i, "accuracy");
      assert!(accuracy >= exp_acc);
    };

    // Insert below brute force index cap.
    db.insert([("0".to_string(), vecs[0].clone())].into_iter().collect())
      .unwrap();
    let res = db.query(&vecs[0].view(), 100);
    assert_eq!(res.len(), 1);
    assert_eq!(&res[0].0, "0");

    // Still below brute force index cap.
    db.insert((1..734).map(|i| (i.to_string(), vecs[i].clone())).collect())
      .unwrap();
    // It's tempting to directly compare against `nn[i]` given BF's 100% accuracy, but it's still complicated due to 1) non-deterministic DB internal IDs assigned, and 2) unstable sorting (i.e. two same dists).
    assert_accuracy(734, 1.0);
    tracing::info!("inserted 734 so far");

    // Delete some keys.
    // Deleting non-existent keys should do nothing.
    for i in [5, 90, 734, 735, 900, 10002] {
      db.delete(&i.to_string()).unwrap();
    }
    assert_eq!(db.deleted.len(), 2);
    deleted.insert(5);
    deleted.insert(90);

    // Still below brute force index cap.
    db.insert(
      (734..1000)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    assert_accuracy(1000, 1.0);
    tracing::info!("inserted 1000 so far");
    // Ensure the updater_thread isn't doing anything.
    sleep(Duration::from_secs(1));

    assert_eq!(db.deleted.len(), 2);
    assert_eq!(db.next_id.load(Ordering::Relaxed), 1000);
    assert_eq!(db.index.bf.read().clone().unwrap().len(), 1000);
    assert_eq!(db.index.additional_out_neighbors.len(), 0);
    assert_eq!(db.index.temp_nodes.len(), 0);
    match &*db.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 0);
        assert_eq!(vectors.len(), 0);
      }
      Mode::LTI { .. } => unreachable!(),
    };

    // Now, it should build the in-memory index.
    db.insert(
      (1000..1313)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    tracing::info!("inserted 1313 so far");
    let expected_medoid_i = calc_approx_medoid(
      &(0..1313).map(|i| (i, vecs[i].clone())).collect(),
      metric_euclidean,
      10_000,
      None,
    );
    let expected_medoid_key = expected_medoid_i.to_string();
    tracing::info!("calculated expected medoid");
    let expected_medoid_id = db.db.maybe_read_id(&expected_medoid_key).unwrap();
    assert_eq!(db.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(db.index.bf.read().is_none());
    assert_eq!(db.index.additional_out_neighbors.len(), 0);
    assert_eq!(db.index.temp_nodes.len(), 0);
    match &*db.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 1313);
        assert_eq!(vectors.len(), 1313);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(db.db.iter_brute_force_vecs::<f32>().count(), 0);
    assert_eq!(db.db.maybe_read_medoid(), Some(expected_medoid_id));
    assert_eq!(db.db.read_index_mode(), DbIndexMode::InMemory);
    assert_eq!(db.db.iter_nodes::<f32>().count(), 1313);
    let missing_nodes = db
      .db
      .iter_nodes::<f32>()
      .par_bridge()
      .map(|(internal_id, n)| {
        // We don't know the internal ID so we can't just skip if it matches our ID.
        let Some(key) = db.db.maybe_read_key(internal_id) else {
          return 1;
        };
        let i = key.parse::<usize>().unwrap();
        assert_eq!(n.vector, vecs[i].to_vec());
        0
      })
      .sum::<usize>();
    assert_eq!(missing_nodes, deleted.len());

    // Delete some keys.
    // Yes, some of these will be duplicates, which we want to also test.
    // Also delete the medoid. Medoids are never permanently deleted, which will be important later once we test merging.
    let to_delete = (0..40)
      .map(|_| thread_rng().gen_range(0..1313))
      .chain(once(expected_medoid_i))
      .collect::<Vec<_>>();
    to_delete.par_iter().for_each(|&i| {
      db.delete(&i.to_string()).unwrap();
      deleted.insert(i); // Handles already deleted, duplicate delete requests.
    });
    assert_eq!(db.deleted.len(), deleted.len());

    // Still under in-memory index cap.
    db.insert(
      (1313..2090)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    tracing::info!("inserted 2090 so far");
    // Medoid must not have changed.
    assert_eq!(db.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(db.index.bf.read().is_none());
    assert_eq!(db.index.temp_nodes.len(), 0);
    match &*db.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 2090);
        assert_eq!(vectors.len(), 2090);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(db.db.iter_nodes::<f32>().count(), 2090);
    assert_accuracy(2090, 0.95);

    // Now, it should transition to LTI.
    db.insert(
      (2090..2671)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    // Do another insert to wait on the updater_thread to complete the transition.
    db.insert(
      (2671..2722)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    assert_eq!(db.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(db.index.bf.read().is_none());
    assert_eq!(db.index.temp_nodes.len(), 0);
    match &*db.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq } => {
        assert_eq!(pq.pq_vecs.len(), 2722);
      }
    };
    assert_eq!(db.db.iter_brute_force_vecs::<f32>().count(), 0);
    assert_eq!(db.db.maybe_read_medoid(), Some(expected_medoid_id));
    assert_eq!(db.db.read_index_mode(), DbIndexMode::LongTerm);
    assert_eq!(db.db.iter_nodes::<f32>().count(), 2722);
    assert_accuracy(2722, 0.85);

    // Trigger merge due to excessive deletes.
    let to_delete = (0..2722)
      .filter(|&i| !deleted.contains(&i))
      .choose_multiple(&mut thread_rng(), 200);
    to_delete.par_iter().for_each(|&i| {
      db.delete(&i.to_string()).unwrap();
      deleted.insert(i);
    });
    assert_eq!(db.deleted.len(), deleted.len());
    tracing::info!(n = to_delete.len(), "deleted vectors");
    assert_accuracy(2722, 0.84);
    // Do insert to trigger the updater_thread to start the merge.
    db.insert(
      (2722..2799)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    // Do another insert to wait for the updater_thread to finish the merge.
    db.insert(
      [("2799".to_string(), vecs[2799].clone())]
        .into_iter()
        .collect(),
    )
    .unwrap();
    // The medoid is never permanently deleted, so add 1.
    let expected_post_merge_nodes = 2800 - deleted.len() + 1;
    match &*db.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq } => {
        assert_eq!(pq.pq_vecs.len(), expected_post_merge_nodes);
      }
    };
    assert_eq!(db.db.iter_nodes::<f32>().count(), expected_post_merge_nodes);
    // The medoid is never permanently deleted.
    assert_eq!(db.deleted.len(), 1);
    assert_eq!(db.db.iter_deleted().count(), 1);
    // NOTE: We don't update our `deleted` as they are deleted, it's not the same as the soft-delete markers `db.deleted`.
    assert_accuracy(2800, 0.84);

    // Finally, insert all remaining vectors.
    db.insert(
      (2800..n)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .unwrap();
    tracing::info!("inserted all vectors");
    // Do final query.
    assert_accuracy(n, 0.8);
  }
}
