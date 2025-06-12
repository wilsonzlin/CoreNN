use crate::store::schema::DbNodeData;
use crate::store::schema::ADD_EDGES;
use crate::store::schema::DELETED;
use crate::store::schema::ID_TO_KEY;
use crate::store::schema::NODE;
use crate::util::AtomUsz;
use crate::CoreNN;
use crate::Mode;
use ahash::HashSet;
use dashmap::DashSet;
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::sync::Arc;

pub fn compact(corenn: &CoreNN) {
  // From now on, we must work with a consistent snapshot of deleted elements.
  let deleted = corenn.deleted.iter().map(|e| *e).collect::<HashSet<_>>();

  tracing::warn!(deleted = deleted.len(), "compacting");

  let touched = AtomUsz::new(0);
  // We iterate all nodes, as we don't know which nodes have edges to a node in `deleted` (backedges do exist but are regularly pruned). This is the approach in the FreshDiskANN paper.
  // In RocksDB, iterators view a snapshot of the entire DB at the time of iterator creation, so we can safely modify DB entries during iteration. https://github.com/facebook/rocksdb/wiki/Iterator
  // However, for the same reason, we don't iterate nodes, so we can:
  // - Get the latest node at the time of traversal (it may have been a while since the start of the iterator, but RocksDB will give us the stale snapshot value).
  // - Acquire a lock *before* reading the node, to avoid data races.
  ID_TO_KEY.iter(&corenn.db).par_bridge().for_each(|(id, _)| {
    if deleted.contains(&id) {
      return;
    };

    let lock = corenn.node_write_lock.get(id);
    let _g = lock.lock();

    let Some(node) = NODE.read(&corenn.db, id) else {
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
    let add = corenn.add_edges.remove(&id).map(|e| e.1);
    if add.is_none() && deleted_neighbors.is_empty() {
      // Node is untouched.
      return;
    };
    touched.inc();
    if let Some(add) = add {
      for n in add {
        if !deleted.contains(&n) {
          new_neighbors.insert(n);
        };
      }
    }
    for dn in corenn.get_nodes(&deleted_neighbors) {
      let Some(dn) = dn else {
        continue;
      };
      for &n in dn.neighbors.iter() {
        if !deleted.contains(&n) {
          new_neighbors.insert(n);
        };
      }
    }

    let new_neighbors =
      corenn.prune_candidates(&node.vector, &new_neighbors.into_iter().collect_vec());

    let mut txn = Vec::new();
    let new_node = DbNodeData {
      version: node.version + 1,
      neighbors: new_neighbors,
      vector: node.vector,
    };
    NODE.batch_put(&mut txn, id, &new_node);
    if let Mode::Uncompressed(cache) = &*corenn.mode.read() {
      cache.insert(id, Arc::new(new_node));
    };
    ADD_EDGES.batch_put(&mut txn, id, Vec::new());
    // We've already removed the add_edges entry.
    corenn.db.write(txn);
  });

  let mut txn = Vec::new();
  for &id in deleted.iter() {
    ADD_EDGES.batch_delete(&mut txn, id);
    DELETED.batch_delete(&mut txn, id);
    NODE.batch_delete(&mut txn, id);
  }
  corenn.db.write(txn);

  for &id in deleted.iter() {
    corenn.add_edges.remove(&id);
    // This is to free memory, not for correctness.
    match &*corenn.mode.read() {
      Mode::Compressed(_, cache) => {
        cache.remove(id);
      }
      Mode::Uncompressed(cache) => {
        cache.remove(id);
      }
    };
    corenn.deleted.remove(&id);
  }
  tracing::info!(
    touched = touched.get(),
    deleted = deleted.len(),
    "compaction complete",
  );
}
