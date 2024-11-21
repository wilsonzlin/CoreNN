#![feature(f16)]
#![feature(path_add_extension)]

use ahash::HashSet;
use ahash::HashSetExt;
use arbitrary_lock::ArbitraryLock;
use bf::BruteForceIndex;
use blob::BlobStore;
use common::nan_to_num;
use common::Dtype;
use common::Id;
use common::Metric;
use common::PointDist;
use common::PrecomputedDists;
use common::StdMetric;
use dashmap::DashMap;
use dashmap::DashSet;
use dashmap::Entry;
use db::Db;
use db::DbTransaction;
use db::NodeData;
use in_memory::InMemoryIndex;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use parking_lot::RwLock;
use pq::ProductQuantizer;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use search::GreedySearchable;
use search::Query;
use serde::Deserialize;
use serde::Serialize;
use std::cmp::max;
use std::cmp::min;
use std::collections::BTreeMap;
use std::fs;
use std::iter::zip;
use std::mem::replace;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use vamana::Vamana;
use vamana::VamanaParams;

pub mod bf;
pub mod blob;
pub mod common;
pub mod db;
pub mod hnsw;
pub mod in_memory;
pub mod pq;
pub mod queue;
pub mod search;
pub mod vamana;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RoxanneDbCfg {
  pub beam_width: usize,
  pub brute_force_index_cap: usize,
  pub degree_bound: usize,
  pub dim: usize,
  pub distance_threshold: f64,
  pub max_temp_indices: usize,
  pub metric: StdMetric,
  pub pq_sample_size: usize,
  pub pq_subspaces: usize,
  pub query_search_list_cap: usize,
  pub temp_index_cap: usize,
  pub update_batch_size: usize,
  pub update_search_list_cap: usize,
}

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
    // This holds a write lock on the entry, so multiple callers won't clash-write to DB.
    match self.pq_vecs.entry(id) {
      Entry::Occupied(o) => {
        let pq_vec = o.get();
        self.pq.decode_1(&pq_vec.view())
      }
      Entry::Vacant(v) => {
        let node = self.db.read_node(id);
        let vec = Array1::from_vec(node.vector);
        let pq_vec = self.pq.encode_1(&vec.view());
        // Persist.
        let mut txn = DbTransaction::new();
        txn.write_pq_vec(id, pq_vec.to_vec());
        txn.commit(&self.db);
        // Insert into cache.
        v.insert(pq_vec);
        vec
      }
    }
  }
}

enum TempIndex<T: Dtype> {
  ANN(InMemoryIndex<T>),
  BF(BruteForceIndex<T>),
}

struct LongTermIndex<T: Dtype> {
  db: Arc<Db>,
  pq: PqState<T>,
  medoid: Id,
  metric: Metric<T>,
  vamana_params: VamanaParams,
  // These are specifically used only by the updater thread for StreamingMerge, and is cleared after the process. It's safe to let these affect regular queries during StreamingMerge though.
  override_neighbors: DashMap<Id, Vec<Id>>,
  override_vectors: DashMap<Id, Array1<T>>,
}

impl<'a, T: Dtype> GreedySearchable<'a, T> for LongTermIndex<T> {
  type FullVec = Array1<T>;
  type Neighbors = Vec<Id>;
  type Point = Array1<T>;

  fn medoid(&self) -> Id {
    self.medoid
  }

  fn metric(&self) -> Metric<T> {
    self.metric
  }

  fn get_point(&'a self, id: Id) -> Self::Point {
    self
      .override_vectors
      .get(&id)
      .map(|e| e.value().clone())
      .unwrap_or_else(|| self.pq.get_pq_vec(id))
  }

  fn get_out_neighbors(&'a self, id: Id) -> (Self::Neighbors, Option<Self::FullVec>) {
    let n = self.db.read_node(id);
    let over = self.override_neighbors.get(&id).map(|e| e.value().clone());
    (over.unwrap_or(n.neighbors), Some(Array1::from(n.vector)))
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }
}

// We only implement Vamana to be able to use compute_robust_pruned during StreamingMerge.
// We don't use set_* as the update process is more sophisticated (see StreamingMerge).
impl<'a, T: Dtype> Vamana<'a, T> for LongTermIndex<T> {
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

struct TempIndices<T: Dtype> {
  frozen: Vec<InMemoryIndex<T>>,
  current: TempIndex<T>,
}

fn should_compact<T: Dtype>(cur: &TempIndex<T>, cfg: &RoxanneDbCfg) -> bool {
  match cur {
    TempIndex::ANN(idx) => idx.len() >= cfg.temp_index_cap,
    TempIndex::BF(idx) => idx.len() >= cfg.brute_force_index_cap,
  }
}

pub struct RoxanneDb<T: Dtype> {
  blobs: BlobStore,
  busy_signal: AtomicBool,
  cfg: RoxanneDbCfg,
  db: Arc<Db>,
  deleted: DashSet<Id>,
  key_locks: ArbitraryLock<String, parking_lot::Mutex<()>>,
  long_term_index: RwLock<Option<LongTermIndex<T>>>,
  metric: Metric<T>,
  next_id: AtomicUsize,
  temp_indices: RwLock<TempIndices<T>>,
  update_sender: std::sync::mpsc::Sender<(Vec<Id>, Vec<Array1<T>>)>,
}

impl<T: Dtype> RoxanneDb<T> {
  // Why do all updates from a single thread (i.e. serialized), instead of only the compaction process?
  // Because our Vamana implementation doesn't support parallel updates (it does batching instead), so a lot of complexity to split out insertion (thread-safe) and compaction (single thread) ultimately ends up being pointless. It's safer to get correct; if we need to optimize, we can profile in the future.
  // NOTE: Many operations may seem incorrect due to only acquiring read locks, but remember that all updates are processed serially within this sole thread only.
  fn updater_thread(
    self: &Arc<RoxanneDb<T>>,
    receiver: std::sync::mpsc::Receiver<(Vec<Id>, Vec<Array1<T>>)>,
  ) {
    while let Ok(mut batch) = receiver.recv() {
      // Collect more if available.
      while let Ok(e) = receiver.try_recv() {
        batch.0.extend(e.0);
        batch.1.extend(e.1);
      }

      let rlock = self.temp_indices.read();

      // Insert into current TempIndex.
      let (batch_ids, batch_vecs) = batch;
      match &rlock.current {
        TempIndex::ANN(idx) => {
          idx.insert(batch_ids.clone(), batch_vecs.clone());
        }
        TempIndex::BF(index) => {
          for (id, vec) in zip(batch_ids.clone(), batch_vecs.clone()) {
            index.insert(id, vec);
          }
        }
      };
      if !should_compact(&rlock.current, &self.cfg) {
        continue;
      };

      // From now on, we must work with a consistent, immutable set of deleted elements.
      let deleted = self.deleted.iter().map(|e| *e).collect_vec();

      enum PostAction<T: Dtype> {
        InsertIntoFrozen,
        ReplaceWithAnn(InMemoryIndex<T>),
      }

      // It's safe to only acquire a read lock, as no one else can insert except us (so no one will be inserting while we transform the current TempIndex and as a result lose data).
      let post_action = match &rlock.current {
        TempIndex::BF(bf) => {
          let id_to_vec = bf.vectors();
          let ids = id_to_vec.iter().map(|e| *e.key()).collect_vec();
          let vecs = id_to_vec.iter().map(|e| e.value().clone()).collect_vec();
          let ann = InMemoryIndex::builder(ids, vecs)
            .degree_bound(self.cfg.degree_bound)
            .distance_threshold(self.cfg.distance_threshold)
            .metric(self.metric)
            .update_batch_size(self.cfg.update_batch_size)
            .update_search_list_cap(self.cfg.update_search_list_cap)
            .build();
          PostAction::ReplaceWithAnn(ann)
        }
        TempIndex::ANN(ann) => {
          let ids = ann.graph.iter().map(|e| *e.key()).collect_vec();
          let index_id = rlock.frozen.len();
          self.blobs.write_temp_index(index_id, &ann.into());
          let mut txn = DbTransaction::new();
          // We must record the latest temp index count/ID as the index itself is stored on the filesystem which we cannot atomically commit. Consider that if we succeed in writing the index file but fail to commit these deletions to the DB, these WAL vectors will be in the WAL (or latest RW temp index) and in a frozen index (because we'll just read all index files in that folder).
          txn.write_temp_index_count(index_id + 1);
          for id in ids {
            txn.delete_write_ahead_log_vector(id);
          }
          txn.commit(&self.db);
          PostAction::InsertIntoFrozen
        }
      };
      drop(rlock);

      let mut wlock = self.temp_indices.write();
      match post_action {
        PostAction::InsertIntoFrozen => {
          let ex = replace(
            &mut wlock.current,
            TempIndex::BF(BruteForceIndex::new(self.metric)),
          );
          let TempIndex::ANN(ann) = ex else {
            unreachable!();
          };
          wlock.frozen.push(ann);
        }
        PostAction::ReplaceWithAnn(in_memory_index) => {
          wlock.current = TempIndex::ANN(in_memory_index);
        }
      };
      drop(wlock);

      let rlock = self.temp_indices.read();
      if rlock.frozen.len() < self.cfg.max_temp_indices {
        continue;
      };

      // TODO Can we relax this a bit more and not block all updates for so long?
      self.busy_signal.store(true, Ordering::Relaxed);

      // TODO Can we avoid memory copying here? Even if we have to do hacks like Arc. Otherwise, we're doubling memory requirements of in-memory temp indices.
      let mut ids = Vec::new();
      let mut vecs = Vec::new();
      let mut ingest_point = |id: Id, vec: &Array1<T>| {
        if deleted.contains(&id) {
          return;
        }
        ids.push(id);
        vecs.push(vec.clone());
      };
      for f in rlock.frozen.iter() {
        for e in f.vectors.iter() {
          ingest_point(*e.key(), e.value());
        }
      }
      match &rlock.current {
        TempIndex::ANN(ann) => {
          for e in ann.vectors.iter() {
            ingest_point(*e.key(), e.value());
          }
        }
        TempIndex::BF(bf) => {
          for e in bf.vectors() {
            ingest_point(*e.key(), e.value());
          }
        }
      };

      if self.long_term_index.read().is_none() {
        // Since we've never built a LTI before, we need to build the PQ now.
        let ss = min(ids.len(), self.cfg.pq_sample_size);
        let mut mat = Array2::zeros((ss, self.cfg.dim));
        for (i, vec) in vecs.choose_multiple(&mut thread_rng(), ss).enumerate() {
          mat.row_mut(i).assign(vec);
        }
        let pq = ProductQuantizer::train(&mat.view(), self.cfg.pq_subspaces);
        self.blobs.write_pq_model(&pq);
        // Free memory now.
        drop(mat);

        // Build LTI. Given temp indices should be only a fraction of memory budget, we can build in memory using full vectors for speed and accuracy. (The graph and vectors don't exist on disk yet anyway.)
        let graph = InMemoryIndex::builder(ids.clone(), vecs.clone())
          .degree_bound(self.cfg.degree_bound)
          .distance_threshold(self.cfg.distance_threshold)
          .metric(self.metric)
          .update_batch_size(self.cfg.update_batch_size)
          .update_search_list_cap(self.cfg.update_search_list_cap)
          .build();

        // Persist to DB.
        let mut txn = DbTransaction::new();
        txn.write_has_long_term_index();
        txn.write_temp_index_count(0);
        for &id in ids.iter().chain(deleted.iter()) {
          // For many, neither of these will exist, but for defensive coding delete them anyway.
          txn.delete_deleted(id);
          txn.delete_write_ahead_log_vector(id);
        }
        txn.write_medoid(graph.medoid);
        // TODO Excessive memory usage by cloning and buffering all NodeDatas in memory (transaction)?
        for id in ids {
          txn.write_node(id, &NodeData {
            neighbors: graph.graph.get(&id).unwrap().clone(),
            vector: graph.vectors.get(&id).unwrap().to_vec(),
          });
        }
        txn.commit(&self.db);

        *self.temp_indices.write() = TempIndices {
          frozen: Vec::new(),
          current: TempIndex::BF(BruteForceIndex::new(self.metric)),
        };
        // Technically, there is a tiny tiny window here for a query to come in and get no results. But it's better than having the possibility of duplicate IDs in results and needing a filter after every query.
        *self.long_term_index.write() = Some(LongTermIndex {
          db: self.db.clone(),
          medoid: graph.medoid,
          metric: self.metric,
          override_neighbors: Default::default(),
          override_vectors: Default::default(),
          vamana_params: VamanaParams {
            degree_bound: self.cfg.degree_bound,
            distance_threshold: self.cfg.distance_threshold,
            update_batch_size: self.cfg.update_batch_size,
            update_search_list_cap: self.cfg.update_search_list_cap,
          },
          pq: PqState {
            db: self.db.clone(),
            pq_vecs: DashMap::new(),
            pq,
          },
        });
      } else {
        // StreamingMerge.
        // TODO FUTURE: Persist to secondary intermediate LTI on disk so as to not use possibly lots of memory.
        let lock = self.long_term_index.read();
        let lti = lock.as_ref().unwrap();

        // Phase 1: process deletions.
        // Why not do phase 1 and 3 at the same time? Likely because we could be deleting a huge chunk of the graph such that phase 2 inserts will insert edges that would be poor after deletion. (Deleting now means inserts will use the graph as it exists post-deletion.)
        if !deleted.is_empty() {
          self
            .db
            .iter_nodes::<T>()
            .par_bridge()
            .for_each(|(id, node)| {
              let mut deleted_neighbors = Vec::new();
              let candidates = DashSet::new();
              for &n in node.neighbors.iter() {
                if deleted.contains(&n) {
                  deleted_neighbors.push(n);
                } else {
                  candidates.insert(n);
                };
              }
              if deleted_neighbors.is_empty() {
                return;
              };
              // TODO I/O parallelism using Rayon: no no.
              deleted_neighbors.par_iter().for_each(|&n_id| {
                for n in self.db.read_node::<T>(n_id).neighbors {
                  if !deleted.contains(&n) {
                    candidates.insert(n);
                  };
                }
              });
              let new_candidates = lti.compute_robust_pruned(id, candidates);
              lti.override_neighbors.insert(id, new_candidates);
            });
        }

        // Phase 2: insert but don't add backedges.
        let pending_backedges = DashMap::<Id, Vec<Id>>::new();
        ids.par_iter().zip(&vecs).for_each(|(&id, vec)| {
          // We need to insert the vectors into the LTI graph (at least temporarily),
          // so that compute_robust_pruned works, both now and in phase 3.
          lti.override_vectors.insert(id, vec.clone());
          let mut candidates = HashSet::new();
          lti.greedy_search(
            Query::Id(id),
            1,
            self.cfg.update_search_list_cap,
            self.cfg.beam_width,
            lti.medoid,
            |_| true,
            Some(&mut candidates),
            None,
            None,
          );
          let new_neighbors = lti.compute_robust_pruned(id, candidates);
          for &j in new_neighbors.iter() {
            pending_backedges.entry(j).or_default().push(id);
          }
          // This is safe to insert mid-iteration as the graph currently has no edges to these new nodes.
          lti.override_neighbors.insert(id, new_neighbors);
        });

        // Phase 3: add backedges.
        pending_backedges.into_par_iter().for_each(|(id, to_add)| {
          let mut new_neighbors = lti.get_out_neighbors(id).0;
          for j in to_add {
            if !new_neighbors.contains(&j) {
              new_neighbors.push(j);
            };
          }
          if new_neighbors.len() > self.cfg.degree_bound {
            new_neighbors = lti.compute_robust_pruned(id, new_neighbors);
          }
          // This is safe to insert mid-iteration, as in phase 3 we just run compute_robust_pruned which doesn't look at neighbors of any other nodes.
          lti.override_neighbors.insert(id, new_neighbors);
        });

        // We're done: update the LTI on disk.
        let mut txn = DbTransaction::new();
        txn.write_temp_index_count(0);
        for &id in ids.iter().chain(deleted.iter()) {
          // For many, neither of these will exist, but for defensive coding delete them anyway.
          txn.delete_deleted(id);
          txn.delete_write_ahead_log_vector(id);
        }
        for e in lti.override_neighbors.iter() {
          let (&id, neighbors) = e.pair();
          let vector = lti
            .override_vectors
            .get(&id)
            .map(|e| e.to_vec())
            .unwrap_or_else(|| self.db.read_node(id).vector);
          txn.write_node(id, &NodeData {
            neighbors: neighbors.clone(),
            vector,
          });
        }
        txn.commit(&self.db);

        *self.temp_indices.write() = TempIndices {
          frozen: Vec::new(),
          current: TempIndex::BF(BruteForceIndex::new(self.metric)),
        };
        lti.override_neighbors.clear();
        lti.override_vectors.clear();
      };
      for id in deleted {
        self.deleted.remove(&id);
      }

      self.busy_signal.store(false, Ordering::Relaxed);
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
    let temp_indices = RwLock::new(TempIndices {
      frozen: (0..db.read_temp_index_count())
        .into_par_iter()
        .map(|id| {
          blobs
            .read_temp_index(id)
            .to_index(metric, vamana_params.clone())
        })
        .collect::<Vec<_>>(),
      current: {
        let (ids, vecs): (Vec<_>, Vec<_>) = db.iter_write_ahead_log_vectors::<T>().unzip();
        if ids.len() < cfg.brute_force_index_cap {
          let bf = BruteForceIndex::new(metric);
          for (id, vec) in zip(ids, vecs) {
            bf.insert(id, vec);
          }
          TempIndex::BF(bf)
        } else {
          let ann = InMemoryIndex::builder(ids, vecs)
            .degree_bound(cfg.degree_bound)
            .distance_threshold(cfg.distance_threshold)
            .metric(metric)
            .update_batch_size(cfg.update_batch_size)
            .update_search_list_cap(cfg.update_search_list_cap)
            .build();
          TempIndex::ANN(ann)
        }
      },
    });
    let long_term_index = RwLock::new(db.read_has_long_term_index().then(|| LongTermIndex {
      db: db.clone(),
      medoid: db.read_medoid(),
      metric,
      override_neighbors: Default::default(),
      override_vectors: Default::default(),
      vamana_params,
      pq: PqState {
        db: db.clone(),
        pq_vecs: DashMap::new(),
        pq: blobs.read_pq_model(),
      },
    }));
    let (send, recv) = std::sync::mpsc::channel();
    let roxanne = Arc::new(RoxanneDb {
      blobs,
      busy_signal: AtomicBool::new(false),
      cfg,
      db,
      deleted,
      key_locks: ArbitraryLock::new(),
      long_term_index,
      metric,
      next_id: AtomicUsize::new(next_id),
      temp_indices,
      update_sender: send,
    });
    std::thread::spawn({
      let roxanne = roxanne.clone();
      move || roxanne.updater_thread(recv)
    });
    roxanne
  }

  fn greedy_search<'a, 'q>(
    &'a self,
    g: &'a impl GreedySearchable<'a, T>,
    q: &'q ArrayView1<T>,
    k: usize,
  ) -> Vec<PointDist> {
    g.greedy_search(
      Query::Vec(q),
      k,
      self.cfg.query_search_list_cap,
      self.cfg.beam_width,
      g.medoid(),
      |n| !self.deleted.contains(&n.id),
      None,
      None,
      None,
    )
  }

  fn greedy_search_temp_rw_index(
    &self,
    indices: &TempIndices<T>,
    q: &ArrayView1<T>,
    k: usize,
  ) -> Vec<PointDist> {
    match &indices.current {
      TempIndex::ANN(idx) => self.greedy_search(idx, q, k),
      TempIndex::BF(index) => index.query(q, k, |id| !self.deleted.contains(&id)),
    }
  }

  fn greedy_search_temp_ro_indices(
    &self,
    indices: &TempIndices<T>,
    q: &ArrayView1<T>,
    k: usize,
  ) -> Vec<PointDist> {
    indices
      .frozen
      .par_iter()
      .flat_map(|idx| self.greedy_search(idx, q, k))
      .collect::<Vec<_>>()
  }

  fn greedy_search_long_term_index(&self, q: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    self
      .long_term_index
      .read()
      .as_ref()
      .map(|idx| self.greedy_search(idx, q, k))
      .unwrap_or_default()
  }

  pub fn query(&self, query: &ArrayView1<T>, k: usize) -> Vec<PointDist> {
    let lock = self.temp_indices.read();
    let results = Mutex::new(Vec::new());
    rayon::join(
      || {
        rayon::join(
          || {
            results
              .lock()
              .extend(self.greedy_search_temp_rw_index(&lock, query, k))
          },
          || {
            results
              .lock()
              .extend(self.greedy_search_temp_ro_indices(&lock, query, k))
          },
        )
      },
      || {
        results
          .lock()
          .append(&mut self.greedy_search_long_term_index(query, k))
      },
    );
    let mut results = results.into_inner();
    results.sort_unstable_by_key(|e| OrderedFloat(e.dist));
    results.truncate(k);
    results
  }

  // Considerations:
  // - We don't want to keep piling on to a brute force index (e.g. while compactor struggles to keep up) and degrade query performance exponentially.
  // - We don't want to keep buffering vectors in memory, acknowledging inserts but in reality there's backpressure while the graph is being updated.
  // - Both above points means that we should have some mechanism of returning a "busy" signal to the caller, instead of just accepting into unbounded memory or blocking the caller.
  // - We don't want to perform any compaction within the function (instead running it in a background thread), as otherwise that negatively affects the insertion call latency for the one unlucky caller.
  // Require a Map to ensure there are no duplicates (which would otherwise complicate insertion).
  pub fn insert(&self, entries: BTreeMap<String, Array1<T>>) -> Result<(), RoxanneDbError> {
    if self.busy_signal.load(Ordering::Relaxed) {
      return Err(RoxanneDbError::Busy);
    };

    let n = entries.len();

    let ids_base = self.next_id.fetch_add(n, Ordering::Relaxed);
    let ids = (0..n).map(|i| ids_base + i).collect_vec();

    let mut entries = entries.into_iter().collect_vec();

    let ex_ids = Mutex::new(Vec::new());
    entries.par_iter_mut().enumerate().for_each(|(i, (k, v))| {
      // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
      v.mapv_inplace(nan_to_num);

      let id = ids_base + i;
      let locker = self.key_locks.get(k.clone());
      let _lock = locker.lock();
      let mut txn = DbTransaction::new();
      if let Some(ex_id) = self.db.maybe_read_id(&k) {
        txn.write_deleted(ex_id);
        txn.delete_key(ex_id);
        ex_ids.lock().push(ex_id);
      };
      txn.write_key(id, &k);
      txn.write_id(&k, id);
      txn.write_write_ahead_log_vector(id, v.as_slice().unwrap());
      txn.commit(&self.db);
    });

    self
      .update_sender
      .send((ids, entries.into_iter().map(|e| e.1).collect()))
      .unwrap();
    for id in ex_ids.into_inner() {
      self.deleted.insert(id);
    }

    Ok(())
  }

  pub fn delete(&self, key: &str) -> Result<(), RoxanneDbError> {
    if self.busy_signal.load(Ordering::Relaxed) {
      return Err(RoxanneDbError::Busy);
    };

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
