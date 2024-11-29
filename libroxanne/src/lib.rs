#![feature(async_closure)]
#![feature(duration_millis_float)]
#![feature(f16)]
#![feature(path_add_extension)]
#![warn(clippy::future_not_send)]
#![allow(async_fn_in_trait)]

use crate::util::AsyncConcurrentIteratorExt;
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
use db::Db;
use db::DbIndexMode;
use db::DbTransaction;
use db::NodeData;
use flume::Receiver;
use flume::Sender;
use futures::stream::iter;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
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
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use search::GreedySearchParams;
use search::GreedySearchable;
use search::GreedySearchableAsync;
use search::INeighbors;
use search::IVec;
use search::Query;
use signal_future::SignalFuture;
use signal_future::SignalFutureController;
use std::cmp::max;
use std::cmp::min;
use std::collections::BTreeMap;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use util::ArcMap;
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
pub mod util;
pub mod vamana;

#[derive(Debug)]
pub enum RoxanneDbError {
  Busy,
}

enum Mode<T: Dtype> {
  InMemory {
    graph: Arc<DashMap<Id, Vec<Id>>>,
    vectors: Arc<DashMap<Id, Array1<T>>>,
  },
  LTI {
    pq_vecs: DashMap<Id, Array1<u8>>,
    pq: ProductQuantizer<T>,
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

enum IndexPoint<'a, T> {
  LTI(Array1<T>),
  Temp(dashmap::mapref::one::Ref<'a, Id, (Vec<Id>, Array1<T>)>),
  // I'd prefer to use MappedRwLockReadGuard but we're blocked on https://github.com/Amanieu/parking_lot/issues/289.
  InMemory(ArcMap<DashMap<Id, Array1<T>>, dashmap::mapref::one::Ref<'a, Id, Array1<T>>>),
}

impl<'a, T: Dtype> IVec<T> for IndexPoint<'a, T> {
  fn view(&self) -> ArrayView1<T> {
    match self {
      IndexPoint::LTI(v) => v.view(),
      IndexPoint::Temp(v) => v.value().1.view(),
      IndexPoint::InMemory(v) => v.value().view(),
    }
  }
}

enum IndexNeighborsBase<'a, T> {
  Temp(dashmap::mapref::one::Ref<'a, Id, (Vec<Id>, Array1<T>)>),
  // I'd prefer to use MappedRwLockReadGuard but we're blocked on https://github.com/Amanieu/parking_lot/issues/289.
  InMemory(ArcMap<DashMap<Id, Vec<Id>>, dashmap::mapref::one::Ref<'a, Id, Vec<Id>>>),
  LTI(Vec<Id>),
}

struct IndexNeighbors<'a, T> {
  base: IndexNeighborsBase<'a, T>,
  add: Option<dashmap::mapref::one::Ref<'a, Id, Vec<Id>>>,
}

impl<'a, T> IndexNeighbors<'a, T> {
  fn iter_add(&self) -> impl Iterator<Item = Id> + '_ {
    match &self.add {
      Some(e) => e.value().as_slice(),
      None => &[],
    }
    .iter()
    .map(|e| *e)
  }
}

impl<'a, T> INeighbors for IndexNeighbors<'a, T> {
  fn iter(&self) -> impl Iterator<Item = Id> {
    match &self.base {
      IndexNeighborsBase::InMemory(e) => e.value().as_slice(),
      IndexNeighborsBase::LTI(v) => v.as_slice(),
      IndexNeighborsBase::Temp(e) => e.value().0.as_slice(),
    }
    .iter()
    .map(|e| *e)
    .chain(self.iter_add())
  }
}

impl<T: Dtype> GreedySearchable<T> for Index<T> {
  type FullVec = Array1<T>;
  type Neighbors<'a> = IndexNeighbors<'a, T>;
  type Point<'a> = IndexPoint<'a, T>;

  fn medoid(&self) -> Id {
    self.medoid.load(Ordering::Relaxed)
  }

  fn metric(&self) -> Metric<T> {
    self.metric
  }

  fn get_point<'a>(&'a self, id: Id) -> Self::Point<'a> {
    if let Some(temp) = self.temp_nodes.get(&id) {
      return IndexPoint::Temp(temp);
    }
    match &*self.mode.read() {
      Mode::InMemory { vectors, .. } => {
        IndexPoint::InMemory(ArcMap::map(vectors.clone(), |v| v.get(&id).unwrap()))
      }
      Mode::LTI { pq, pq_vecs } => IndexPoint::LTI({
        let pq_vec = pq_vecs.get(&id).unwrap();
        pq.decode_1(&pq_vec.view())
      }),
    }
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }
}

impl<T: Dtype> GreedySearchableAsync<T> for Index<T> {
  async fn get_out_neighbors_async<'a>(
    &'a self,
    id: Id,
  ) -> (Self::Neighbors<'a>, Option<Self::FullVec>) {
    if let Some(temp) = self.temp_nodes.get(&id) {
      // We don't need to add additional_out_neighbors, as temp_nodes is a full override.
      // We don't need to return a full vector as it's not from the disk.
      return (
        IndexNeighbors {
          base: IndexNeighborsBase::Temp(temp),
          add: None,
        },
        None,
      );
    }
    // Don't use `match` as we can't hold lock across await in Mode::LTI branch.
    if let Mode::InMemory { graph, .. } = &*self.mode.read() {
      return (
        IndexNeighbors {
          base: IndexNeighborsBase::InMemory(ArcMap::map(Arc::clone(&graph), |g| {
            g.get(&id).unwrap()
          })),
          add: self.additional_out_neighbors.get(&id),
        },
        None,
      );
    };
    let n = self.db.read_node(id).await;
    (
      IndexNeighbors {
        base: IndexNeighborsBase::LTI(n.neighbors),
        add: self.additional_out_neighbors.get(&id),
      },
      Some(Array1::from(n.vector)),
    )
  }
}

// We only implement Vamana to be able to use compute_robust_pruned during updater_thread.
// We don't impl VamanaSync and use set_* as the update process is more sophisticated (see updater_thread).
impl<T: Dtype> Vamana<T> for Index<T> {
  fn params(&self) -> &vamana::VamanaParams {
    &self.vamana_params
  }
}

pub struct RoxanneDb<T: Dtype> {
  blobs: BlobStore,
  cfg: RoxanneDbCfg,
  db: Arc<Db>,
  deleted: DashSet<Id>,
  index: Index<T>,
  // We must use Tokio's lock as we will hold it across await. (parking_lot does not support this.)
  key_locks: ArbitraryLock<String, tokio::sync::Mutex<()>>,
  next_id: AtomicUsize,
  update_sender: Sender<(Vec<Id>, Vec<Array1<T>>, SignalFutureController<()>)>,
}

impl<T: Dtype> RoxanneDb<T> {
  // Why do all updates from a single thread (i.e. serialized), instead of only the compaction process?
  // Because our Vamana implementation doesn't support parallel updates (it does batching instead), so a lot of complexity to split out insertion (thread-safe) and compaction (single thread) ultimately ends up being pointless. It's safer to get correct; if we need to optimize, we can profile in the future.
  // NOTE: Many operations in this method may seem incorrect due to only acquiring read locks (so it seems like changes could happen beneath our feet and we're not correctly looking at a stable consistent snapshot/view of the entire system/data), but remember that all updates are processed serially within this sole single-threaded/serially-executed function only.
  async fn updater_thread(
    self: Arc<Self>,
    receiver: Receiver<(Vec<Id>, Vec<Array1<T>>, SignalFutureController<()>)>,
  ) {
    let cfg = &self.cfg;
    let db = &self.db;
    let idx = &self.index;
    while let Ok(mut msg) = receiver.recv_async().await {
      // Collect more if available.
      let mut signals = vec![msg.2];
      while let Ok(e) = receiver.try_recv() {
        msg.0.extend(e.0);
        msg.1.extend(e.1);
        signals.push(e.2);
      }
      let insert_n = msg.0.len();
      let (insert_ids, insert_vecs, _) = msg;
      tracing::debug!(n = insert_n, "processing inserts");

      // If we are in brute force index mode, then handle differently (it's not a graph, so update process is different).
      // BruteForceIndex is wrapped in Arc so we can work on it while dropping the lock (so we can potentially replace it).
      // WARNING: Do not inline this to the `if` RHS; in Rust, given `a.b.c`, `a` is dropped but `b` is not, so if we inline this, we'll hold the lock.
      let bf = idx.bf.read().clone();
      if let Some(bf) = bf {
        for (&id, v) in zip(&insert_ids, insert_vecs) {
          bf.insert(id, v);
        }
        if bf.len() > cfg.brute_force_index_cap {
          tracing::warn!(
            vectors = bf.len(),
            "transitioning brute force index to in-memory index"
          );
          let started = Instant::now();
          let (ids, vecs): (Vec<_>, Vec<_>) = bf
            .vectors()
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .unzip();
          let ann = InMemoryIndex::builder(ids.clone(), vecs)
            .degree_bound(cfg.degree_bound)
            .distance_threshold(cfg.distance_threshold)
            .metric(idx.metric)
            .update_batch_size(cfg.update_batch_size)
            .update_search_list_cap(cfg.update_search_list_cap)
            .build();
          tracing::info!(vectors = bf.len(), "built optimized in-memory index");
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
          txn.commit(db).await;
          idx.medoid.store(ann.medoid(), Ordering::Relaxed);
          *idx.mode.write() = Mode::InMemory {
            graph: ann.graph,
            vectors: ann.vectors,
          };
          // Update this last to avoid very subtle race condition where `bf` is None but `mode` isn't `InMemory` yet.
          *idx.bf.write() = None;
          tracing::info!(
            vectors = bf.len(),
            exec_ms = started.elapsed().as_millis_f64(),
            "transitioned to in-memory index"
          );
        };
        for c in signals {
          c.signal(());
        }
        continue;
      };

      if let Mode::LTI { pq, pq_vecs } = &*idx.mode.read() {
        insert_ids
          .par_iter()
          .zip(&insert_vecs)
          .for_each(|(&id, v)| {
            // We may as well pre-populate the cache now instead of reading from disk again later.
            pq_vecs.insert(id, pq.encode_1(&v.view()));
          });
      };

      // We must use Tokio's lock as we need to hold across await (unsupported by parking_lot).
      let txn = Arc::new(tokio::sync::Mutex::new(DbTransaction::new()));
      // Use batching as otherwise it's possible to build a poor graph. For example, the existing graph might have 5 nodes, and we're inserting 7,000; without batching, all edges can only be from those 5 nodes to the 7,000, hardly enough. Use batching to ensure there are enough high-quality optimized edges between our newly inserted nodes.
      for batch_no in 0..insert_n.div_ceil(cfg.update_batch_size) {
        let batch_touched = Arc::new(DashSet::new());
        let start = batch_no * cfg.update_batch_size;
        let end = min(insert_n, (batch_no + 1) * cfg.update_batch_size);
        (start..end)
          // We have to clone as we will move into tokio::spawn which requires 'static.
          .map(|i| (insert_ids[i], insert_vecs[i].clone()))
          .map(|(id, v)| {
            let roxanne = self.clone();
            let batch_touched = batch_touched.clone();
            // Spawn as we'll do CPU-heavy `compute_robust_pruned` and `pq.encode_1` (.for_each_concurrent is not sufficient as it's single-threaded).
            tokio::spawn(async move {
              let idx = &roxanne.index;
              let mut candidates = HashSet::new();
              idx
                .greedy_search_async(GreedySearchParams {
                  query: Query::Vec(&v.view()),
                  k: 1,
                  search_list_cap: idx.vamana_params.update_search_list_cap,
                  beam_width: 1,
                  start: idx.medoid.load(Ordering::Relaxed),
                  filter: |n| !roxanne.deleted.contains(&n),
                  out_visited: Some(&mut candidates),
                  out_metrics: None,
                  ground_truth: None,
                })
                .await;
              // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
              let neighbors = idx.compute_robust_pruned(Query::Vec(&v.view()), candidates);
              // We need to insert now because once we add to additional_out_neighbors, some queries may reach this new node and request its neighbors (and we haven't inserted it into the index yet). Also, later compute_robust_pruned calls from other nodes (when adding backedges) will fetch this vector, so insert into `temp_nodes` now.
              // We won't update InMemory or DB just yet, as this new node's neighbors could change again in future batches.
              // CORRECTNESS: our new `neighbors` can only contain existing nodes and nodes from previous batches, and none from this insertion batch, so it's safe to expand any of the nodes in `neighbors` right now (i.e. get_out_neighbors/get_point will return them).
              idx.temp_nodes.insert(id, (neighbors.clone(), v));
              for j in neighbors.iter() {
                // `id` cannot have existed in any existing out-neighbors of any node as it has just been created, so we don't need to check if it doesn't already exist in `out(j)` first.
                idx.additional_out_neighbors.entry(j).or_default().push(id);
                idx.additional_edge_count.fetch_add(1, Ordering::Relaxed);
                batch_touched.insert(j);
              }
            })
          })
          .collect::<FuturesUnordered<_>>()
          .for_each(async |e| e.unwrap())
          .await;
        Arc::into_inner(batch_touched)
          .unwrap()
          .into_iter()
          .map(|id| {
            let roxanne = self.clone();
            let txn = txn.clone();
            // Spawn as we'll do CPU-heavy compute_robust_pruned (.for_each_concurrent is not sufficient as it's single-threaded).
            tokio::spawn(async move {
              let idx = &roxanne.index;
              // We must use clone instead of `get_mut` to update in place, as we'll hold a mut lock on a shard in the temp_nodes DashMap and then compute_robust_pruned will try to read from temp_nodes and deadlock.
              // We could separate into `temp_vecs` and `temp_neighbors`, but in general it's not a good idea to hold a write lock for a long time (and compute_robust_pruned takes a long time).
              let temp_node = idx.temp_nodes.get(&id).map(|e| e.value().0.clone());
              if let Some(mut new_neighbors) = temp_node {
                // This is a node in the current inserts but processed in a previous batch, so this is a different code path as there's nothing to read from or write to disk (yet).
                for j in idx.additional_out_neighbors.remove(&id).unwrap().1 {
                  new_neighbors.push(j);
                  idx.additional_edge_count.fetch_sub(1, Ordering::Relaxed);
                }
                if new_neighbors.len() > roxanne.cfg.max_degree_bound {
                  new_neighbors = idx.compute_robust_pruned(Query::Id(id), new_neighbors);
                }
                idx.temp_nodes.get_mut(&id).unwrap().0 = new_neighbors;
              } else {
                let (new_neighbors, full_vec) = idx.get_out_neighbors_async(id).await;
                let mut new_neighbors = new_neighbors.into_vec();
                // Clone so we can remove later if necessary (otherwise we'll deadlock).
                let add_neighbors = idx.additional_out_neighbors.get(&id).unwrap().clone();
                new_neighbors.extend(add_neighbors.iter());
                // Technically, we can always update the graph node's out neighbors if we're using in-memory index, but for a consistent code path we'll stick with using additional_out_neighbors even for in-memory.
                if new_neighbors.len() <= roxanne.cfg.max_degree_bound {
                  let mut txn = txn.lock().await;
                  txn.write_additional_out_neighbors(id, &add_neighbors);
                } else {
                  // At this point, compute_robust_pruned will likely look up the vectors for our newly inserted vectors, which is why we have temp_nodes.
                  new_neighbors = idx.compute_robust_pruned(Query::Id(id), new_neighbors);
                  {
                    let mut txn = txn.lock().await;
                    txn.write_node(id, &NodeData {
                      neighbors: new_neighbors.clone(),
                      // If LTI, full_vec will be Some; if in-memory, use get_point.
                      vector: full_vec
                        .map(|v| v.into_raw_vec())
                        // TODO Avoid cloning. (write_node should take a `&[T]`.)
                        .unwrap_or_else(|| idx.get_point(id).into_vec()),
                    });
                    txn.delete_additional_out_neighbors(id);
                  };
                  if let Mode::InMemory { graph, .. } = &*idx.mode.read() {
                    graph.insert(id, new_neighbors);
                  };
                  // Technically, this affects LTI queries because we haven't committed the new NodeData yet; in reality, it's minor (very short period) and shouldn't matter.
                  idx.additional_edge_count.fetch_sub(
                    idx.additional_out_neighbors.remove(&id).unwrap().1.len(),
                    Ordering::Relaxed,
                  );
                }
              }
            })
          })
          .collect::<FuturesUnordered<_>>()
          .for_each(async |e| e.unwrap())
          .await;
      }
      let mut txn = Arc::into_inner(txn).unwrap().into_inner();
      for (id, v) in zip(insert_ids, insert_vecs) {
        // Don't remove from temp_nodes yet as we haven't committed to disk yet.
        let neighbors = idx.temp_nodes.get(&id).unwrap().0.clone();
        txn.write_node(id, &NodeData {
          // TODO Avoid cloning. (write_node should take a `&[Id]`.)
          neighbors: neighbors.clone(),
          // TODO Avoid cloning. (write_node should take a `&[T]`.)
          vector: v.to_vec(),
        });
        if let Mode::InMemory { graph, vectors } = &*idx.mode.read() {
          graph.insert(id, neighbors);
          vectors.insert(id, v);
        };
      }
      txn.commit(db).await;
      idx.temp_nodes.clear();
      for c in signals {
        c.signal(());
      }
      tracing::debug!(n = insert_n, "inserted vectors");

      // Opportunity to transition from in-memory to long term index.
      // Clone (cheap Arc) so we can release lock and not have to hold it across .await.
      let in_memory_vecs = match &*idx.mode.read() {
        Mode::InMemory { vectors, .. } => Some(vectors.clone()),
        _ => None,
      };
      if let Some(vectors) = in_memory_vecs {
        if vectors.len() <= cfg.in_memory_index_cap {
          // The subsequent code is StreamingMerge, which we don't (yet) support for in memory indices (only the on-disk LTI).
          // TODO Should we support it?
          continue;
        };
        tracing::warn!("transitioning in-memory index to long term index");
        // Since we've never built a LTI before, we need to build the PQ now.
        let ss = min(vectors.len(), cfg.pq_sample_size);
        let mut mat = Array2::zeros((ss, cfg.dim));
        for (i, vec) in vectors
          .iter()
          .choose_multiple(&mut thread_rng(), ss)
          .into_iter()
          .enumerate()
        {
          mat.row_mut(i).assign(&*vec);
        }
        let pq = ProductQuantizer::train(&mat.view(), cfg.pq_subspaces);
        self.blobs.write_pq_model(&pq).await;
        tracing::info!(
          sample_inputs = ss,
          subspaces = cfg.pq_subspaces,
          "trained PQ"
        );
        // Free memory now.
        drop(mat);

        let mut txn = DbTransaction::new();
        txn.write_index_mode(DbIndexMode::LongTerm);
        txn.commit(db).await;

        // We may as well populate the cache now, instead of reading from disk again later.
        let pq_vecs = DashMap::new();
        vectors.par_iter().for_each(|e| {
          let (&id, vec) = e.pair();
          pq_vecs.insert(id, pq.encode_1(&vec.view()));
        });

        *idx.mode.write() = Mode::LTI { pq_vecs, pq };
        tracing::info!("transitioned to long term index");
        continue;
      };

      if self.deleted.len() < cfg.merge_threshold_deletes
        && idx.additional_edge_count.load(Ordering::Relaxed) < cfg.merge_threshold_additional_edges
      {
        // No need to merge yet.
        continue;
      };

      // From now on, we must work with a consistent snapshot of deleted elements.
      let deleted = self
        .deleted
        .iter()
        .map(|e| *e)
        .filter(|&e| e != idx.medoid.load(Ordering::Relaxed))
        .collect::<HashSet<_>>();

      tracing::warn!(
        deleted = deleted.len(),
        additional_edges = idx.additional_edge_count.load(Ordering::Relaxed),
        "merging",
      );

      // In RocksDB, iterators view a snapshot of the entire DB at the time of iterator creation, so we can safely modify DB entries during iteration. https://github.com/facebook/rocksdb/wiki/Iterator
      let touched = AtomicUsize::new(0);
      // We iterate all nodes, as we don't know which nodes have edges to a node in `deleted` (backedges do exist but are regularly pruned). This is the approach in the FreshDiskANN paper.
      db.iter_nodes::<T>()
        .for_each_concurrent(None, async |(id, node)| {
          if deleted.contains(&id) {
            return;
          };

          let mut deleted_neighbors = Vec::new();
          let new_neighbors = DashSet::new();
          for n in node.neighbors.iter() {
            if deleted.contains(&n) {
              deleted_neighbors.push(n);
            } else {
              new_neighbors.insert(n);
            };
          }
          let add = idx.additional_out_neighbors.remove(&id).map(|e| e.1);
          if add.is_none() && deleted_neighbors.is_empty() {
            // Node is untouched.
            return;
          };
          touched.fetch_add(1, Ordering::Relaxed);
          if let Some(add) = add {
            idx
              .additional_edge_count
              .fetch_sub(add.len(), Ordering::Relaxed);
            for n in add {
              if !deleted.contains(&n) {
                new_neighbors.insert(n);
              };
            }
          }
          iter(deleted_neighbors)
            .for_each_concurrent(None, async |n_id| {
              for n in db.read_node::<T>(n_id).await.neighbors {
                if !deleted.contains(&n) {
                  new_neighbors.insert(n);
                };
              }
            })
            .await;

          // Spawn as we'll do CPU-heavy compute_robust_pruned (our current .for_each_concurrent is not sufficient as it's single-threaded).
          let roxanne = self.clone();
          tokio::spawn(async move {
            // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
            let new_neighbors = roxanne
              .index
              .compute_robust_pruned(Query::Id(id), new_neighbors);

            let mut txn = DbTransaction::new();
            txn.write_node(id, &NodeData {
              neighbors: new_neighbors,
              vector: node.vector,
            });
            txn.delete_additional_out_neighbors(id);
            txn.commit(&roxanne.db).await;
          })
          .await
          .unwrap();
        })
        .await;

      let mut txn = DbTransaction::new();
      for &id in deleted.iter() {
        txn.delete_deleted(id);
        txn.delete_additional_out_neighbors(id);
        txn.delete_node(id);
      }
      txn.commit(db).await;

      let Mode::LTI { pq_vecs, .. } = &*idx.mode.read() else {
        unreachable!();
      };
      for &id in deleted.iter() {
        self.deleted.remove(&id);
        pq_vecs.remove(&id);
        if let Some(add) = idx.additional_out_neighbors.remove(&id) {
          idx
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
  pub async fn open(dir: PathBuf) -> Arc<RoxanneDb<T>> {
    let blobs = BlobStore::open(dir.join("blobs")).await;
    let cfg_raw = tokio::fs::read_to_string(dir.join("roxanne.toml"))
      .await
      .unwrap();
    let cfg: RoxanneDbCfg = toml::from_str(&cfg_raw).unwrap();
    let metric = cfg.metric.get_fn::<T>();
    let vamana_params = VamanaParams {
      degree_bound: cfg.degree_bound,
      distance_threshold: cfg.distance_threshold,
      update_batch_size: cfg.update_batch_size,
      update_search_list_cap: cfg.update_search_list_cap,
    };

    let db = Arc::new(Db::open(dir.join("db")).await);
    let next_id = Mutex::new(0);
    let deleted = DashSet::new();
    db.iter_deleted()
      .for_each(async |id| {
        deleted.insert(id);
        let mut next_id = next_id.lock();
        *next_id = max(id + 1, *next_id);
      })
      .await;
    db.iter_ids()
      .for_each(async |(id, _)| {
        let mut next_id = next_id.lock();
        *next_id = max(id + 1, *next_id);
      })
      .await;
    let next_id = next_id.into_inner();

    let additional_out_neighbors = DashMap::new();
    let additional_edge_count = AtomicUsize::new(0);
    db.iter_additional_out_neighbors()
      .for_each(async |(id, add)| {
        additional_edge_count.fetch_add(add.len(), Ordering::Relaxed);
        additional_out_neighbors.insert(id, add);
      })
      .await;

    let mode = db.read_index_mode().await;

    let index = Index {
      bf: RwLock::new(if mode == DbIndexMode::BruteForce {
        let bf = BruteForceIndex::new(metric);
        db.iter_brute_force_vecs::<T>()
          .for_each(async |(id, vec)| {
            bf.insert(id, Array1::from_vec(vec));
          })
          .await;
        Some(Arc::new(bf))
      } else {
        None
      }),
      db: db.clone(),
      // In BruteForce mode (e.g. init), there is no medoid.
      medoid: db.maybe_read_medoid().await.unwrap_or(Id::MAX).into(),
      metric,
      vamana_params,
      additional_out_neighbors,
      additional_edge_count,
      temp_nodes: DashMap::new(),
      mode: RwLock::new(if mode == DbIndexMode::LongTerm {
        let pq = blobs.read_pq_model().await;
        // TODO Build pq_vecs from existing nodes.
        Mode::LTI {
          pq,
          pq_vecs: DashMap::new(),
        }
      } else {
        // This also handles BruteForce mode (this'll be empty.)
        let graph = Arc::new(DashMap::new());
        let vectors = Arc::new(DashMap::new());
        db.iter_nodes()
          .for_each(async |(id, n)| {
            graph.insert(id, n.neighbors);
            vectors.insert(id, Array1::from_vec(n.vector));
          })
          .await;
        Mode::InMemory { graph, vectors }
      }),
    };

    let (send, recv) = flume::unbounded();
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
    tokio::spawn({
      let roxanne = roxanne.clone();
      async move {
        roxanne.updater_thread(recv).await;
      }
    });
    roxanne
  }

  pub async fn query<'a>(&'a self, query: &'a ArrayView1<'a, T>, k: usize) -> Vec<(String, f64)> {
    // We must (cheaply) clone to avoid holding lock across await.
    let bf = self.index.bf.read().clone();
    let res = if let Some(bf) = bf {
      // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
      bf.query(query, k, |n| !self.deleted.contains(&n))
    } else {
      self
        .index
        .greedy_search_async(GreedySearchParams {
          query: Query::Vec(query),
          k,
          search_list_cap: self.cfg.query_search_list_cap,
          beam_width: self.cfg.beam_width,
          start: self.index.medoid(),
          filter: |n| !self.deleted.contains(&n),
          out_visited: None,
          out_metrics: None,
          ground_truth: None,
        })
        .await
    };
    let keys = res
      .iter()
      .map_concurrent(|r| self.db.maybe_read_key(r.id))
      .collect::<Vec<_>>()
      .await;
    zip(keys, res)
      // A node may have been deleted during the query (already collected, so didn't get filtered), or literally just after the end of the query but before here.
      // TODO DOCUMENT: it's possible to get less than k for the above reason.
      .filter_map(|(k, r)| k.map(|k| (k, r.dist)))
      .collect()
  }

  // Considerations:
  // - We don't want to keep piling on to a brute force index (e.g. while compactor struggles to keep up) and degrade query performance exponentially.
  // - We don't want to keep buffering vectors in memory, acknowledging inserts but in reality there's backpressure while the graph is being updated.
  // - Both above points means that we should have some mechanism of returning a "busy" signal to the caller, instead of just accepting into unbounded memory or blocking the caller.
  // - We don't want to perform any compaction within the function (instead running it in a background thread), as otherwise that negatively affects the insertion call latency for the one unlucky caller.
  // Require a Map to ensure there are no duplicate keys (which would otherwise complicate insertion).
  pub async fn insert(&self, entries: BTreeMap<String, Array1<T>>) -> Result<(), RoxanneDbError> {
    let n = entries.len();

    let ids_base = self.next_id.fetch_add(n, Ordering::Relaxed);
    let ids = (0..n).map(|i| ids_base + i).collect_vec();

    let vectors = zip(&ids, entries)
      .map_concurrent(|(&id, (k, v))| {
        async move {
          let locker = self.key_locks.get(k.clone());
          let _lock = locker.lock().await;
          // TODO FIX/DOCUMENT: For simplicity and performance reasons, the deletion of an existing vector with the same key is **not** atomic; always retry inserts. The alternative seems to be to hold a lock on the key for the entirety of its insert in updater_thread.
          let mut txn = DbTransaction::new();
          if let Some(ex_id) = self.db.maybe_read_id(&k).await {
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
          txn.commit(&self.db).await;

          // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
          v.mapv(nan_to_num)
        }
      })
      .collect::<Vec<_>>()
      .await;

    let (signal, ctl) = SignalFuture::new();
    self
      .update_sender
      .send_async((ids, vectors, ctl))
      .await
      .unwrap();
    signal.await;
    Ok(())
  }

  pub async fn delete(&self, key: &str) -> Result<(), RoxanneDbError> {
    let locker = self.key_locks.get(key.to_string());
    let lock = locker.lock().await;
    let id = self.db.maybe_read_id(key).await;
    if let Some(id) = id {
      let mut txn = DbTransaction::new();
      txn.write_deleted(id);
      txn.delete_id(key);
      txn.delete_key(id);
      txn.commit(&self.db).await;
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
  use crate::util::AsyncConcurrentIteratorExt;
  use crate::util::AsyncConcurrentStreamExt;
  use crate::Mode;
  use crate::RoxanneDb;
  use ahash::HashSet;
  use dashmap::DashSet;
  use futures::stream::iter;
  use futures::StreamExt;
  use itertools::Itertools;
  use ndarray::Array1;
  use ordered_float::OrderedFloat;
  use rand::seq::IteratorRandom;
  use rand::thread_rng;
  use rand::Rng;
  use rayon::iter::IntoParallelIterator;
  use rayon::iter::ParallelIterator;
  use std::fs;
  use std::iter::once;
  use std::path::PathBuf;
  use std::sync::atomic::Ordering;
  use std::thread::sleep;
  use std::time::Duration;
  use std::time::Instant;

  // Unlike tokio::main, tokio::test by default uses only one thread: https://docs.rs/tokio/latest/tokio/attr.test.html#multi-threaded-runtime.
  #[tokio::test(flavor = "multi_thread")]
  async fn test_roxanne() {
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
    let rx = RoxanneDb::open(dir).await;
    tracing::info!("opened database");

    // First test: an empty DB should provide no results.
    let res = rx.query(&gen().view(), 100).await;
    assert_eq!(res.len(), 0);

    let deleted = DashSet::new();
    macro_rules! assert_accuracy {
      ($below_i:expr, $exp_acc:expr) => {{
        tracing::debug!(n = $below_i, "running queries");
        let mode = rx.db.read_index_mode().await;
        let start = Instant::now();
        let mut correct = 0;
        let mut total = 0;
        let results = (0..$below_i)
          .map_concurrent(async |i| {
            tokio::spawn({
              let rx = rx.clone();
              let vec = vecs[i].clone();
              async move { rx.query(&vec.view(), 100).await }
            })
            .await
            .unwrap()
          })
          .collect_vec()
          .await;
        for (i, res) in results.into_iter().enumerate() {
          let got = res
            .iter()
            .map(|e| e.0.parse::<usize>().unwrap())
            .collect::<HashSet<_>>();
          let want = nn[i]
            .iter()
            .cloned()
            .filter(|&n| n < $below_i && !deleted.contains(&n))
            .take(100)
            .collect::<HashSet<_>>();
          total += want.len();
          correct += want.intersection(&got).count();
        }
        let accuracy = correct as f64 / total as f64;
        let exec_ms = start.elapsed().as_millis_f64();
        tracing::info!(
          mode = format!("{:?}", mode),
          accuracy,
          qps = $below_i as f64 / exec_ms * 1000.0,
          "accuracy"
        );
        assert!(accuracy >= $exp_acc);
      }};
    }

    // Insert below brute force index cap.
    rx.insert([("0".to_string(), vecs[0].clone())].into_iter().collect())
      .await
      .unwrap();
    let res = rx.query(&vecs[0].view(), 100).await;
    assert_eq!(res.len(), 1);
    assert_eq!(&res[0].0, "0");

    // Still below brute force index cap.
    rx.insert((1..734).map(|i| (i.to_string(), vecs[i].clone())).collect())
      .await
      .unwrap();
    // It's tempting to directly compare against `nn[i]` given BF's 100% accuracy, but it's still complicated due to 1) non-deterministic DB internal IDs assigned, and 2) unstable sorting (i.e. two same dists).
    // Accuracy can be very slightly less than 1.0 due to non-deterministic sorting of equal dist points.
    assert_accuracy!(734, 0.99);
    tracing::info!("inserted 734 so far");

    // Delete some keys.
    // Deleting non-existent keys should do nothing.
    for i in [5, 90, 734, 735, 900, 10002] {
      rx.delete(&i.to_string()).await.unwrap();
    }
    assert_eq!(rx.deleted.len(), 2);
    deleted.insert(5);
    deleted.insert(90);

    // Still below brute force index cap.
    rx.insert(
      (734..1000)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    // Accuracy can be very slightly less than 1.0 due to non-deterministic sorting of equal dist points.
    assert_accuracy!(1000, 0.99);
    tracing::info!("inserted 1000 so far");
    // Ensure the updater_thread isn't doing anything.
    sleep(Duration::from_secs(1));

    assert_eq!(rx.deleted.len(), 2);
    assert_eq!(rx.next_id.load(Ordering::Relaxed), 1000);
    assert_eq!(rx.index.bf.read().clone().unwrap().len(), 1000);
    assert_eq!(rx.index.additional_out_neighbors.len(), 0);
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 0);
        assert_eq!(vectors.len(), 0);
      }
      Mode::LTI { .. } => unreachable!(),
    };

    // Now, it should build the in-memory index.
    rx.insert(
      (1000..1313)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
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
    let expected_medoid_id = rx.db.maybe_read_id(&expected_medoid_key).await.unwrap();
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.additional_out_neighbors.len(), 0);
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 1313);
        assert_eq!(vectors.len(), 1313);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(rx.db.iter_brute_force_vecs::<f32>().count().await, 0);
    assert_eq!(rx.db.maybe_read_medoid().await, Some(expected_medoid_id));
    assert_eq!(rx.db.read_index_mode().await, DbIndexMode::InMemory);
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 1313);
    let missing_nodes = rx
      .db
      .iter_nodes::<f32>()
      .filter_map(async |(internal_id, n)| {
        // We don't know the internal ID so we can't just skip if it matches our ID.
        let Some(key) = rx.db.maybe_read_key(internal_id).await else {
          return Some(());
        };
        let i = key.parse::<usize>().unwrap();
        assert_eq!(n.vector, vecs[i].to_vec());
        None
      })
      .count()
      .await;
    assert_eq!(missing_nodes, deleted.len());

    // Delete some keys.
    // Yes, some of these will be duplicates, which we want to also test.
    // Also delete the medoid. Medoids are never permanently deleted, which will be important later once we test merging.
    let to_delete = (0..40)
      .map(|_| thread_rng().gen_range(0..1313))
      .chain(once(expected_medoid_i))
      .collect::<Vec<_>>();
    iter(&to_delete)
      .for_each_concurrent(None, async |&i| {
        rx.delete(&i.to_string()).await.unwrap();
        deleted.insert(i); // Handles already deleted, duplicate delete requests.
      })
      .await;
    assert_eq!(rx.deleted.len(), deleted.len());

    // Still under in-memory index cap.
    rx.insert(
      (1313..2090)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    tracing::info!("inserted 2090 so far");
    // Medoid must not have changed.
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 2090);
        assert_eq!(vectors.len(), 2090);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 2090);
    assert_accuracy!(2090, 0.95);

    // Now, it should transition to LTI.
    rx.insert(
      (2090..2671)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    // Do another insert to wait on the updater_thread to complete the transition.
    rx.insert(
      (2671..2722)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq_vecs, .. } => {
        assert_eq!(pq_vecs.len(), 2722);
      }
    };
    assert_eq!(rx.db.iter_brute_force_vecs::<f32>().count().await, 0);
    assert_eq!(rx.db.maybe_read_medoid().await, Some(expected_medoid_id));
    assert_eq!(rx.db.read_index_mode().await, DbIndexMode::LongTerm);
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 2722);
    assert_accuracy!(2722, 0.85);

    // Trigger merge due to excessive deletes.
    let to_delete = (0..2722)
      .filter(|&i| !deleted.contains(&i))
      .choose_multiple(&mut thread_rng(), 200);
    iter(&to_delete)
      .for_each_concurrent(None, async |&i| {
        rx.delete(&i.to_string()).await.unwrap();
        deleted.insert(i);
      })
      .await;
    assert_eq!(rx.deleted.len(), deleted.len());
    tracing::info!(n = to_delete.len(), "deleted vectors");
    assert_accuracy!(2722, 0.84);
    // Do insert to trigger the updater_thread to start the merge.
    rx.insert(
      (2722..2799)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    // Do another insert to wait for the updater_thread to finish the merge.
    rx.insert(
      [("2799".to_string(), vecs[2799].clone())]
        .into_iter()
        .collect(),
    )
    .await
    .unwrap();
    // The medoid is never permanently deleted, so add 1.
    let expected_post_merge_nodes = 2800 - deleted.len() + 1;
    match &*rx.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq_vecs, .. } => {
        assert_eq!(pq_vecs.len(), expected_post_merge_nodes);
      }
    };
    assert_eq!(
      rx.db.iter_nodes::<f32>().count().await,
      expected_post_merge_nodes
    );
    // The medoid is never permanently deleted.
    assert_eq!(rx.deleted.len(), 1);
    assert_eq!(rx.db.iter_deleted().count().await, 1);
    // NOTE: We don't update our `deleted` as they are deleted, it's not the same as the soft-delete markers `db.deleted`.
    assert_accuracy!(2800, 0.84);

    // Finally, insert all remaining vectors.
    rx.insert(
      (2800..n)
        .map(|i| (i.to_string(), vecs[i].clone()))
        .collect(),
    )
    .await
    .unwrap();
    tracing::info!("inserted all vectors");
    // Do final query.
    // We expect a dramatic drop in accuracy as we're inserting uniformly random data (i.e. noise) that cannot be compressed or fitted to, and we've now inserted ~300% more data so our PQ model is now very poor.
    assert_accuracy!(n, 0.77);
  }
}
