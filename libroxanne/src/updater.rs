use crate::cfg::Cfg;
use crate::cfg::CompressionMode;
use crate::common::Id;
use crate::compressor::pq::ProductQuantizer;
use crate::compressor::trunc::TruncCompressor;
use crate::compressor::Compressor;
use crate::db::DbTransaction;
use crate::db::NodeData;
use crate::util::unarc;
use crate::util::AsyncConcurrentIteratorExt;
use crate::util::AtomUsz;
use crate::Cache;
use crate::Mode;
use crate::Roxanne;
use ahash::HashSet;
use ahash::HashSetExt;
use dashmap::DashMap;
use dashmap::DashSet;
use flume::Receiver;
use futures::stream::iter;
use futures::StreamExt;
use half::f16;
use itertools::Itertools;
use ndarray::Array1;
use signal_future::SignalFutureController;
use tracing::debug;
use std::iter::zip;
use std::sync::Arc;

pub enum Update {
  Insert(String, Array1<f16>, SignalFutureController<()>),
  Delete(String, SignalFutureController<()>),
}

struct Updates {
  keys_to_delete_first: Vec<String>,
  insert_keys: Vec<String>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<f16>>,
  signals: Vec<SignalFutureController<()>>,
}

/// Awaits the next update, then collects any further buffered updates. Returns None if the channel has closed. Otherwise, returns organized data about the updates.
async fn collect_updates(receiver: &Receiver<Update>, next_id: &AtomUsz, cfg: &Cfg) -> Option<Updates> {
  // DashMap instead of HashMap is a workaround for collect_msg otherwise borrowing to_insert mutably, so we can't get its length outside of the closure.
  let to_insert = DashMap::new();
  let mut to_delete = HashSet::new();
  let mut signals = vec![];
  let mut collect_msg = |msg: Update| {
    match msg {
      Update::Insert(k, v, signal) => {
        // If there are duplicates within the same insert request or update batch, we can skip inserting the previous ones.
        to_insert.insert(k.clone(), v);
        // Delete any existing entry.
        to_delete.insert(k);
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
    // Use batching as otherwise it's possible to build a poor graph. For example, the existing graph might have 5 nodes, and we're inserting 7,000; without batching, all edges can only be from those 5 nodes to the 7,000, hardly enough. Use batching to ensure there are enough high-quality optimized edges between our newly inserted nodes.
    // Since we do batching here, we may as well not collect more than batch_size, otherwise we'll just do complex batching-within-batching.
    if to_insert.len() >= cfg.update_batch_size() {
      break;
    };
  }
  let (insert_keys, insert_vecs): (Vec<_>, Vec<_>) = to_insert.into_iter().unzip();
  let insert_n = insert_keys.len();
  let insert_ids = (0..insert_n).map(|i| next_id.get() + i).collect_vec();
  next_id.inc(insert_n);
  Some(Updates {
    keys_to_delete_first: to_delete.into_iter().collect(),
    insert_keys,
    insert_ids,
    insert_vecs,
    signals,
  })
}

async fn update_persisted_graph_first_time(
  txn: &Arc<tokio::sync::Mutex<DbTransaction>>,
  rox: &Arc<Roxanne>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<f16>>,
) {
  let dim = insert_vecs[0].len();
  debug!(dim, n = insert_ids.len(), "first graph update");
  rox.dim.set(dim);

  txn.lock().await.write_node(0, &NodeData {
    neighbors: insert_ids.clone(),
    vector: insert_vecs[0].clone(),
  });
  for (i, (&id, v)) in zip(&insert_ids, insert_vecs).enumerate() {
    let mut neighbors = insert_ids.clone();
    neighbors.remove(i);
    txn.lock().await.write_node(id, &NodeData {
      neighbors,
      vector: v,
    });
  }
}

async fn update_persisted_graph(
  txn: &Arc<tokio::sync::Mutex<DbTransaction>>,
  rox: &Arc<Roxanne>,
  insert_ids: Vec<Id>,
  insert_vecs: Vec<Array1<f16>>,
  is_first: bool,
) -> (Arc<DashMap<Id, NodeData>>, Arc<DashMap<Id, Vec<Id>>>) {
  if is_first {
    update_persisted_graph_first_time(txn, rox, insert_ids, insert_vecs).await;
    return Default::default();
  };

  let backedges = Arc::new(DashMap::<Id, Vec<Id>>::new());

  zip(insert_ids, insert_vecs)
    .map(|(id, v)| (rox.clone(), backedges.clone(), txn.clone(), id, v.clone()))
    // Spawn as we'll do CPU-heavy calls so they should be spread across CPU cores.
    .spawn_for_each(|(rox, backedges, txn, id, v)| async move {
      let candidates = rox
        .search(&v.view(), 1, rox.cfg.update_search_list_cap())
        .await
        .1;
      // This is CPU bound, not I/O, so don't run in spawn_blocking.
      let neighbors = rox.prune_candidates(&v.view(), candidates).await;
      // We don't, but we *can* safely persist immediately: only existing nodes will be in `neighbors` so no non-existent node expansions occur if this is hit.
      txn.lock().await.write_node(id, &NodeData {
        neighbors: neighbors.clone(),
        vector: v,
      });
      for j in neighbors {
        backedges.entry(j).or_default().push(id);
      }
    })
    .await;

  // We cannot update in-memory data until DB is consistent.
  // So, we collect updates to do after transaction commit.
  let node_data_updates = Arc::new(DashMap::new());
  let add_edges_updates = Arc::new(DashMap::new());

  unarc(backedges)
    .into_iter()
    .map(|(id, backneighbors)| {
      (
        rox.clone(),
        txn.clone(),
        node_data_updates.clone(),
        add_edges_updates.clone(),
        id,
        backneighbors,
      )
    })
    .spawn_for_each(
      |(rox, txn, node_data_updates, add_edges_updates, id, backneighbors)| async move {
        // We intentionally prune before, not after, adding new backneighbors.
        // Otherwise, we'd have to handle these add_edges to new nodes that aren't yet available in the DB.
        // From experience, that makes the system much more complex, subtle, and error-prone, as it breaks the simple source-of-truth and consistency invariants.
        // This keeps the code and design simple and correct, a worthwhile trade-off.
        // It may seem wasteful to not prune with new neighbors together in one go, but the next time this node's touched will just add more add_edges again anyway.
        let mut add_edges = rox
          .add_edges
          .get(&id)
          .map(|e| e.clone())
          .unwrap_or_default();
        if add_edges.len() + backneighbors.len() >= rox.cfg.max_add_edges() {
          // We need to prune this neighbor.
          let NodeData {
            mut neighbors,
            vector,
          } = rox.db.read_node(id).await;
          neighbors.extend_from_slice(&add_edges);
          neighbors = rox.prune_candidates(&vector.view(), neighbors).await;
          let new_node = NodeData { neighbors, vector };
          txn.lock().await.write_node(id, &new_node);
          node_data_updates.insert(id, new_node);
          add_edges.clear();
        }
        add_edges.extend(backneighbors);
        txn.lock().await.write_add_edges(id, &add_edges);
        add_edges_updates.insert(id, add_edges);
      },
    )
    .await;

  (node_data_updates, add_edges_updates)
}

async fn maybe_enable_compression(rox: &Roxanne) {
  if rox.count.get() <= rox.cfg.compression_threshold() {
    return;
  };
  tracing::warn!(threshold = rox.cfg.compression_threshold(), n = rox.count.get(), "enabling compression");

  let compressor: Box<dyn Compressor> = match rox.cfg.compression_mode() {
    CompressionMode::PQ => {
      let pq = ProductQuantizer::<f32>::train_from_roxanne(rox).await;
      let mut txn = DbTransaction::new();
      txn.write_pq_model(&pq);
      txn.commit(&rox.db).await;
      Box::new(pq)
    }
    CompressionMode::Trunc => Box::new(TruncCompressor::new(rox.cfg.trunc_dims())),
  };

  *rox.mode.write().await = Mode::Compressed(compressor, Cache::new());
  tracing::info!("enabled compression");
}

async fn maybe_compact(rox: &Arc<Roxanne>) {
  let cfg = &rox.cfg;
  let db = &rox.db;

  if rox.deleted.len() < cfg.compaction_threshold_deletes() {
    // No need to merge yet.
    return;
  };

  // From now on, we must work with a consistent snapshot of deleted elements.
  let deleted = rox.deleted.iter().map(|e| *e).collect::<HashSet<_>>();

  tracing::warn!(deleted = deleted.len(), "compacting");

  // In RocksDB, iterators view a snapshot of the entire DB at the time of iterator creation, so we can safely modify DB entries during iteration. https://github.com/facebook/rocksdb/wiki/Iterator
  let touched = AtomUsz::new(0);
  // We iterate all nodes, as we don't know which nodes have edges to a node in `deleted` (backedges do exist but are regularly pruned). This is the approach in the FreshDiskANN paper.
  db.iter_nodes()
    .for_each_concurrent(None, async |(id, node)| {
      if deleted.contains(&id) {
        return;
      };

      let mut deleted_neighbors = Vec::new();
      let new_neighbors = DashSet::new();
      for n in node.neighbors {
        if deleted.contains(&n) {
          deleted_neighbors.push(n);
        } else {
          new_neighbors.insert(n);
        };
      }
      let add = rox.add_edges.remove(&id).map(|e| e.1);
      if add.is_none() && deleted_neighbors.is_empty() {
        // Node is untouched.
        return;
      };
      touched.inc(1);
      if let Some(add) = add {
        for n in add {
          if !deleted.contains(&n) {
            new_neighbors.insert(n);
          };
        }
      }
      iter(deleted_neighbors)
        .for_each_concurrent(None, async |n_id| {
          for n in db.read_node(n_id).await.neighbors {
            if !deleted.contains(&n) {
              new_neighbors.insert(n);
            };
          }
        })
        .await;

      // Spawn as we'll do CPU-heavy compute_robust_pruned (our current .for_each_concurrent is not sufficient as it's single-threaded).
      let rox = rox.clone();
      tokio::spawn(async move {
        // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
        // TODO Evict from in-memory cache.
        let new_neighbors = rox
          .prune_candidates(&node.vector.view(), new_neighbors)
          .await;

        let mut txn = DbTransaction::new();
        txn.write_node(id, &NodeData {
          neighbors: new_neighbors,
          vector: node.vector,
        });
        txn.delete_additional_out_neighbors(id);
        txn.commit(&rox.db).await;
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

  for &id in deleted.iter() {
    rox.add_edges.remove(&id);
    // This is to free memory, not for correctness.
    match &*rox.mode.read().await {
      Mode::Compressed(_, cache) => {
        cache.evict(id);
      }
      Mode::Uncompressed(cache) => {
        cache.evict(id);
      }
    };
    rox.deleted.remove(&id);
  }
  tracing::info!(
    touched = touched.get(),
    deleted = deleted.len(),
    "compaction complete",
  );
}

// Why do all updates from a single thread (i.e. serialized), instead of only the compaction process?
// Because our Vamana implementation doesn't support parallel updates (it does batching instead), so a lot of complexity to split out insertion (thread-safe) and compaction (single thread) ultimately ends up being pointless. It's safer to get correct; if we need to optimize, we can profile in the future.
// NOTE: Many operations in this method may seem incorrect due to only acquiring read locks (so it seems like changes could happen beneath our feet and we're not correctly looking at a stable consistent snapshot/view of the entire system/data), but remember that all updates are processed serially within this sole single-threaded/serially-executed function only.
pub async fn updater_thread(rox: Arc<Roxanne>, receiver: Receiver<Update>) {
  while let Some(u) = collect_updates(&receiver, &rox.next_id, &rox.cfg).await {
    let Updates {
      keys_to_delete_first,
      insert_keys,
      insert_ids,
      insert_vecs,
      signals,
    } = u;
    let insert_n = insert_ids.len();
    // When first, when need to update `dim` and clone into entry point with ID 0.
    let is_first = rox.count.get() == 0;
    debug!(insert = insert_n, "received updates");

    // Use Tokio's lock as we need to hold across await.
    let txn = Arc::new(tokio::sync::Mutex::new(DbTransaction::new()));
    let actually_deleted = AtomUsz::new(0);
    iter(&keys_to_delete_first)
      .for_each_concurrent(None, async |key| {
        let Some(ex_id) = rox.db.maybe_read_id(key).await else {
          return;
        };
        // Mark as in-memory deleted now so later vector insertion graph searches don't pick this as a neighbor.
        if !rox.deleted.insert(ex_id) {
          // DB node exists but already marked as soft-deleted.
          return;
        }
        let mut txn = txn.lock().await;
        txn.write_deleted(ex_id);
        txn.delete_id(key);
        txn.delete_key(ex_id);
        actually_deleted.inc(1);
        rox.count.dec(1);
      })
      .await;
    for (key, &id) in zip(&insert_keys, &insert_ids) {
      let mut txn = txn.lock().await;
      txn.write_key(id, key);
      txn.write_id(key, id);
      rox.count.inc(1);
    }

    let (node_data_updates, add_edges_updates) =
      update_persisted_graph(&txn, &rox, insert_ids, insert_vecs, is_first).await;
    unarc(txn).into_inner().commit(&rox.db).await;
    // Now the disk is up-to-date and consistent; we need to align in-memory data.
    let is_uncompressed = match &*rox.mode.read().await {
      Mode::Uncompressed(cache) => {
        // This currently doesn't race with Roxanne::get_point, where it clobbers our insert due to its async DB read.
        // new_nodes only has existing nodes, so get_point will never try to fetch from DB.
        for (id, node) in unarc(node_data_updates) {
          assert!(cache.insert(id, node));
        }
        true
      }
      _ => false,
    };
    for (id, edges) in unarc(add_edges_updates) {
      rox.add_edges.insert(id, edges);
    }
    for c in signals {
      c.signal(());
    }
    tracing::debug!(
      inserted = insert_n,
      deleted = actually_deleted.get(),
      "committed updates"
    );

    // Opportunity to enable compression.
    if is_uncompressed {
      maybe_enable_compression(&rox).await;
    }

    // Compact if enough deleted.
    maybe_compact(&rox).await;
  }
}
