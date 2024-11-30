use crate::bf::BruteForceIndex;
use crate::cfg::RoxanneDbCfg;
use crate::common::to_calc;
use crate::common::Dtype;
use crate::common::DtypeCalc;
use crate::common::Id;
use crate::db::DbIndexMode;
use crate::db::DbTransaction;
use crate::db::NodeData;
use crate::in_memory::InMemoryIndex;
use crate::pq::ProductQuantizer;
use crate::search::GreedySearchParams;
use crate::search::GreedySearchable;
use crate::search::GreedySearchableAsync;
use crate::search::INeighbors;
use crate::search::IPoint;
use crate::search::Points;
use crate::search::Query;
use crate::util::AsyncConcurrentIteratorExt;
use crate::vamana::Vamana;
use crate::Index;
use crate::Mode;
use crate::RoxanneDb;
use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashMap;
use dashmap::DashSet;
use flume::Receiver;
use futures::stream::iter;
use futures::StreamExt;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::Array2;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use signal_future::SignalFutureController;
use std::cmp::min;
use std::iter::zip;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

pub enum Update<T> {
  Insert(Vec<(String, Array1<T>)>, SignalFutureController<()>),
  Delete(String, SignalFutureController<()>),
}

struct Updates<T: Dtype> {
  keys_to_delete_first: Vec<String>,
  insert_keys: Vec<String>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<T>>,
  signals: Vec<SignalFutureController<()>>,
}

impl<T: Dtype> Updates<T> {
  /// Awaits the next update, then collects any further buffered updates. Returns None if the channel has closed. Otherwise, returns organized data about the updates.
  pub async fn receive(receiver: &Receiver<Update<T>>, next_id: &mut Id) -> Option<Self> {
    let mut to_insert = HashMap::new();
    let mut to_delete = HashSet::new();
    let mut signals = vec![];
    let mut collect_msg = |msg: Update<T>| {
      match msg {
        Update::Insert(ents, signal) => {
          for (k, v) in ents {
            // If there are duplicates within the same insert request or update batch, we can skip inserting the previous ones.
            to_insert.insert(k.clone(), v);
            // Delete any existing entry.
            to_delete.insert(k);
          }
          signals.push(signal);
        }
        Update::Delete(key, signal) => {
          to_insert.remove(&key);
          // We still need to remove any existing one, even if a new insert was also requested with the same key (which we've now removed).
          to_delete.insert(key);
          signals.push(signal);
        }
      }
    };
    collect_msg(receiver.recv_async().await.ok()?);
    // Collect more if available.
    while let Ok(msg) = receiver.try_recv() {
      collect_msg(msg);
    }
    let (insert_keys, insert_vecs): (Vec<_>, Vec<_>) = to_insert.into_iter().unzip();
    let insert_n = insert_keys.len();
    let insert_ids = (0..insert_n).map(|i| *next_id + i).collect_vec();
    *next_id += insert_n;
    Some(Updates {
      keys_to_delete_first: to_delete.into_iter().collect(),
      insert_keys,
      insert_ids,
      insert_vecs,
      signals,
    })
  }
}

fn brute_force_index_mode_update<T: Dtype, C: DtypeCalc>(
  txn: &mut DbTransaction,
  bf: &BruteForceIndex<T, C>,
  cfg: &RoxanneDbCfg,
  idx: &Index<T, C>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<T>>,
) {
  for (&id, v) in zip(&insert_ids, &insert_vecs) {
    bf.insert(id, v.clone());
  }
  if bf.len() <= cfg.brute_force_index_cap {
    for (id, v) in zip(insert_ids, insert_vecs) {
      txn.write_node(id, &NodeData {
        neighbors: Vec::new(),
        vector: v.to_vec(),
      });
    }
    return;
  };

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
  txn.write_index_mode(DbIndexMode::InMemory);
  txn.write_medoid(ann.medoid());
  for id in ids {
    txn.write_node(id, &NodeData {
      neighbors: ann.graph.get(&id).unwrap().clone(),
      vector: ann.vectors.get(&id).unwrap().to_vec(),
    });
  }
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
}

async fn graph_index_update<T: Dtype, C: DtypeCalc>(
  txn: &Arc<tokio::sync::Mutex<DbTransaction>>,
  rx: &Arc<RoxanneDb<T, C>>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<T>>,
) {
  let cfg = &rx.cfg;
  let idx = &rx.index;
  let insert_n = insert_ids.len();

  if let Mode::LTI { pq, pq_vecs } = &*idx.mode.read() {
    insert_ids
      .par_iter()
      .zip(&insert_vecs)
      .for_each(|(&id, v)| {
        pq_vecs.insert(id, pq.encode_1(&to_calc(&v.view()).view()));
      });
  };

  // Use batching as otherwise it's possible to build a poor graph. For example, the existing graph might have 5 nodes, and we're inserting 7,000; without batching, all edges can only be from those 5 nodes to the 7,000, hardly enough. Use batching to ensure there are enough high-quality optimized edges between our newly inserted nodes.
  for batch_no in 0..insert_n.div_ceil(cfg.update_batch_size) {
    let batch_touched = Arc::new(DashSet::new());
    let start = batch_no * cfg.update_batch_size;
    let end = min(insert_n, (batch_no + 1) * cfg.update_batch_size);
    (start..end)
      // We have to clone as we will move into tokio::spawn which requires 'static.
      .map(|i| {
        (
          rx.clone(),
          batch_touched.clone(),
          insert_ids[i],
          insert_vecs[i].clone(),
        )
      })
      // Spawn as we'll do CPU-heavy `compute_robust_pruned` and `pq.encode_1` (.for_each_concurrent is not sufficient as it's single-threaded).
      .spawn_for_each(|(rx, batch_touched, id, v)| async move {
        let idx = &rx.index;
        let mut candidates = HashSet::new();
        idx
          .greedy_search_async(GreedySearchParams {
            query: Query::Vec(&v.view()),
            k: 1,
            search_list_cap: idx.vamana_params.update_search_list_cap,
            beam_width: 1,
            start: idx.medoid.load(Ordering::Relaxed),
            filter: |n| !rx.deleted.contains(&n),
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
        for j in neighbors.neighbors() {
          // `id` cannot have existed in any existing out-neighbors of any node as it has just been created, so we don't need to check if it doesn't already exist in `out(j)` first.
          idx.additional_out_neighbors.entry(j).or_default().push(id);
          idx.additional_edge_count.fetch_add(1, Ordering::Relaxed);
          batch_touched.insert(j);
        }
      })
      .await;
    Arc::into_inner(batch_touched)
      .unwrap()
      .into_iter()
      .map(|id| (rx.clone(), txn.clone(), id))
      // Spawn as we'll do CPU-heavy compute_robust_pruned (.for_each_concurrent is not sufficient as it's single-threaded).
      .spawn_for_each(|(rx, txn, id)| async move {
        let idx = &rx.index;
        // We must use clone instead of `get_mut` to update in place, as we'll hold a mut lock on a shard in the temp_nodes DashMap and then compute_robust_pruned will try to read from temp_nodes and deadlock.
        // We could separate into `temp_vecs` and `temp_neighbors`, but in general it's not a good idea to hold a write lock for a long time (and compute_robust_pruned takes a long time).
        let temp_node = idx.temp_nodes.get(&id).map(|e| e.value().0.clone());
        if let Some(mut new_neighbors) = temp_node {
          // This is a node in the current inserts but processed in a previous batch, so this is a different code path as there's nothing to read from or write to disk (yet).
          for j in idx.additional_out_neighbors.remove(&id).unwrap().1 {
            new_neighbors.push(j);
            idx.additional_edge_count.fetch_sub(1, Ordering::Relaxed);
          }
          if new_neighbors.len() > rx.cfg.max_degree_bound {
            new_neighbors = idx.compute_robust_pruned(Query::Id(id), new_neighbors);
          }
          idx.temp_nodes.get_mut(&id).unwrap().0 = new_neighbors;
        } else {
          let (new_neighbors, full_vec) = idx.get_out_neighbors_async(id).await;
          let mut new_neighbors = new_neighbors.into_vec();
          // Clone so we can remove later if necessary (otherwise we'll deadlock).
          let add_neighbors = idx.additional_out_neighbors.get(&id).unwrap().clone();
          new_neighbors.extend(add_neighbors.neighbors());
          // Technically, we can always update the graph node's out neighbors if we're using in-memory index, but for a consistent code path we'll stick with using additional_out_neighbors even for in-memory.
          if new_neighbors.len() <= rx.cfg.max_degree_bound {
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
      .await;
  }

  let mut txn = txn.lock().await;
  for (id, v) in zip(insert_ids, insert_vecs) {
    // Don't remove from temp_nodes yet as we haven't committed to disk yet.
    let neighbors = idx.temp_nodes.get(&id).unwrap().0.clone();
    txn.write_node(id, &NodeData {
      neighbors: neighbors.clone(),
      vector: v.to_vec(),
    });
    if let Mode::InMemory { graph, vectors } = &*idx.mode.read() {
      graph.insert(id, neighbors);
      vectors.insert(id, v);
    };
  }
}

async fn maybe_transition_to_lti<T: Dtype, C: DtypeCalc>(
  rx: &RoxanneDb<T, C>,
  vectors: &DashMap<Id, Array1<T>>,
) {
  let cfg = &rx.cfg;
  if vectors.len() <= cfg.in_memory_index_cap {
    return;
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
    mat.row_mut(i).assign(&to_calc(&vec.view()));
  }
  let pq = ProductQuantizer::train(&mat.view(), cfg.pq_subspaces);
  rx.blobs.write_pq_model(&pq).await;
  tracing::info!(
    sample_inputs = ss,
    subspaces = cfg.pq_subspaces,
    "trained PQ"
  );
  // Free memory now.
  drop(mat);

  let mut txn = DbTransaction::new();
  txn.write_index_mode(DbIndexMode::LongTerm);
  txn.commit(&rx.db).await;

  let pq_vecs = DashMap::new();
  vectors.par_iter().for_each(|e| {
    let (&id, vec) = e.pair();
    pq_vecs.insert(id, pq.encode_1(&to_calc(&vec.view()).view()));
  });

  *rx.index.mode.write() = Mode::LTI { pq_vecs, pq };
  tracing::info!("transitioned to long term index");
}

async fn maybe_merge<T: Dtype, C: DtypeCalc>(rx: &Arc<RoxanneDb<T, C>>) {
  let cfg = &rx.cfg;
  let db = &rx.db;
  let idx = &rx.index;

  if rx.deleted.len() < cfg.merge_threshold_deletes
    && idx.additional_edge_count.load(Ordering::Relaxed) < cfg.merge_threshold_additional_edges
  {
    // No need to merge yet.
    return;
  };

  // From now on, we must work with a consistent snapshot of deleted elements.
  let deleted = rx
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
      for n in node.neighbors.neighbors() {
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
      let rx = rx.clone();
      tokio::spawn(async move {
        // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
        let new_neighbors = rx.index.compute_robust_pruned(Query::Id(id), new_neighbors);

        let mut txn = DbTransaction::new();
        txn.write_node(id, &NodeData {
          neighbors: new_neighbors,
          vector: node.vector,
        });
        txn.delete_additional_out_neighbors(id);
        txn.commit(&rx.db).await;
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
    rx.deleted.remove(&id);
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

// Why do all updates from a single thread (i.e. serialized), instead of only the compaction process?
// Because our Vamana implementation doesn't support parallel updates (it does batching instead), so a lot of complexity to split out insertion (thread-safe) and compaction (single thread) ultimately ends up being pointless. It's safer to get correct; if we need to optimize, we can profile in the future.
// NOTE: Many operations in this method may seem incorrect due to only acquiring read locks (so it seems like changes could happen beneath our feet and we're not correctly looking at a stable consistent snapshot/view of the entire system/data), but remember that all updates are processed serially within this sole single-threaded/serially-executed function only.
pub async fn updater_thread<T: Dtype, C: DtypeCalc>(
  rx: Arc<RoxanneDb<T, C>>,
  receiver: Receiver<Update<T>>,
  mut next_id: Id,
) {
  let cfg = &rx.cfg;
  let db = &rx.db;
  let idx = &rx.index;
  while let Some(u) = Updates::receive(&receiver, &mut next_id).await {
    let Updates {
      keys_to_delete_first,
      insert_keys,
      insert_ids,
      insert_vecs,
      signals,
    } = u;
    let insert_n = insert_ids.len();

    // We must use Tokio's lock as we need to hold across await later in graph_index_update (unsupported by parking_lot).
    let txn = Arc::new(tokio::sync::Mutex::new(DbTransaction::new()));
    let actually_deleted = AtomicUsize::new(0);
    iter(&keys_to_delete_first)
      .for_each_concurrent(None, async |key| {
        if let Some(ex_id) = db.maybe_read_id(key).await {
          let mut txn = txn.lock().await;
          txn.write_deleted(ex_id);
          txn.delete_id(key);
          txn.delete_key(ex_id);
          // Insert into rx.deleted now so later vector insertion graph searches doesn't pick this as a neighbor.
          if rx.deleted.insert(ex_id) {
            actually_deleted.fetch_add(1, Ordering::Relaxed);
          }
        };
      })
      .await;
    for (key, &id) in zip(&insert_keys, &insert_ids) {
      let mut txn = txn.lock().await;
      txn.write_key(id, key);
      txn.write_id(key, id);
    }

    // If we are in brute force index mode, then handle differently (it's not a graph, so update process is different).
    // BruteForceIndex is wrapped in Arc so we can work on it while dropping the lock (so we can potentially replace it).
    // WARNING: Do not inline this to the `if` RHS; in Rust, given `a.b.c`, `a` is dropped but `b` is not, so if we inline this, we'll hold the lock.
    let bf = idx.bf.read().clone();
    if let Some(bf) = bf {
      let mut txn = txn.lock().await;
      brute_force_index_mode_update(&mut txn, &bf, cfg, idx, insert_ids, insert_vecs);
    } else {
      graph_index_update(&txn, &rx, insert_ids, insert_vecs).await;
    };
    Arc::into_inner(txn).unwrap().into_inner().commit(db).await;
    // This must be done after transaction has committed and its changes are visible.
    idx.temp_nodes.clear();
    for c in signals {
      c.signal(());
    }
    tracing::debug!(
      inserted = insert_n,
      deleted = actually_deleted.load(Ordering::Relaxed),
      "committed updates"
    );

    // Opportunity to transition from in-memory to long term index.
    // Clone (cheap Arc) so we can release lock and not have to hold it across .await.
    let in_memory_vecs = match &*idx.mode.read() {
      Mode::InMemory { vectors, .. } => Some(vectors.clone()),
      _ => None,
    };
    if let Some(in_memory_vecs) = in_memory_vecs {
      maybe_transition_to_lti(&rx, &*in_memory_vecs).await;
    } else {
      // This is a separate `else` branch as we'll do StreamingMerge, which we don't (yet) support for in memory indices (only the on-disk LTI).
      // TODO Should we do this also for in memory mode?
      maybe_merge(&rx).await;
    };
  }
}
