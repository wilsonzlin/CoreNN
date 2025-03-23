#![allow(async_fn_in_trait)]
#![feature(async_closure)]
#![feature(duration_millis_float)]
#![feature(extract_if)]
#![feature(path_add_extension)]
#![warn(clippy::future_not_send)]

use crate::util::AsyncConcurrentIteratorExt;
use ahash::HashSet;
use ahash::HashSetExt;
use cfg::Cfg;
use cfg::CfgRaw;
use cfg::CompressionMode;
use common::nan_to_num;
use common::Id;
use common::Metric;
use common::PointDist;
use compressor::trunc::TruncCompressor;
use compressor::Compressor;
use dashmap::DashMap;
use dashmap::DashSet;
use db::Db;
use db::DbTransaction;
use db::NodeData;
use flume::Sender;
use futures::stream::iter;
use futures::StreamExt;
use half::f16;
use itertools::Itertools;
use ndarray::Array1;
use ndarray::ArrayView1;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use signal_future::SignalFuture;
use std::collections::VecDeque;
use std::convert::identity;
use std::future::Future;
use std::iter::zip;
use std::path::Path;
use std::sync::Arc;
use tracing::debug;
use updater::updater_thread;
use updater::Update;
use util::AsyncConcurrentStreamExt;
use util::AtomUsz;

pub mod cfg;
pub mod common;
pub mod compressor;
pub mod db;
pub mod updater;
pub mod util;

// This is somewhat complex:
// - We want to check presence in cache, exclusively to avoid multiple DB fetches for the same thing.
// - DashMap uses locks that cannot be held across await though, so merely holding Entry and then only fetching if vacant doesn't work.
// - We don't want exclusive lock over the entire cache, as that prevents concurrent DB reads.
// So the current simple approach is DashMap<Id, Arc<Lock<Data>>> — the [ArbitraryLock](https://crates.io/crates/arbitrary-lock) pattern except locked data is persisted.
// For simplicity, data must be Clone to avoid issues around holding locks and returning refs. I'm OK with the theoretical perf hit.
struct Cache<D: Clone>(DashMap<Id, Arc<tokio::sync::Mutex<Option<D>>>>);

impl<D: Clone> Cache<D> {
  pub fn new() -> Self {
    Self(DashMap::new())
  }

  pub async fn get_or_compute<F: Future<Output = D>>(&self, id: Id, func: impl FnOnce() -> F) -> D {
    let lock = self.0.entry(id).or_default().clone();
    let mut g = lock.lock().await;
    match &mut *g {
      Some(d) => d.clone(),
      None => {
        let d = func().await;
        g.insert(d).clone()
      }
    }
  }

  /// Returns true if an existing entry was replaced.
  pub fn insert(&self, id: Id, data: D) -> bool {
    self
      .0
      .insert(id, Arc::new(tokio::sync::Mutex::new(Some(data))))
      .is_some()
  }

  pub fn evict(&self, id: Id) {
    self.0.remove(&id);
  }
}

enum Mode {
  // Lazy cache of NodeData in-memory.
  // We don't prepopulate as that makes start time unnecessarily long.
  Uncompressed(Cache<NodeData>),
  // Second element is a cache of compressed vectors.
  // Caching isn't to save computation, it's to avoid DB roundtrip (same as Uncompressed).
  // In compressed mode, graph edges are always fetched from DB.
  Compressed(Box<dyn Compressor>, Cache<Vec<u8>>),
}

pub struct Roxanne {
  add_edges: DashMap<Id, Vec<Id>>,
  cfg: Cfg,
  count: AtomUsz,
  dim: AtomUsz, // Changes after first insert.
  db: Arc<Db>,
  deleted: DashSet<Id>,
  metric: Metric,
  mode: tokio::sync::RwLock<Mode>,
  next_id: AtomUsz,
  update_sender: Sender<Update>,
}

impl Roxanne {
  // For internal use only. No guarantees about DB schema or state.
  pub fn internal_db(&self) -> &Db {
    &self.db
  }

  pub fn get_cfg(&self) -> CfgRaw {
    self.cfg.raw.read().clone()
  }

  pub async fn set_cfg(&self, cfg: CfgRaw) {
    let mut txn = DbTransaction::new();
    txn.write_cfg(&cfg);
    txn.commit(&self.db).await;
    *self.cfg.raw.write() = cfg;
  }

  // TODO It's likely all callers of this have race conditions:
  // - The mode may change during the caller's execution.
  // - If compression suddenly enables, array dimensions and latent space will change.
  // - Array dims. changing will cause panics. Latent space changing will cause incorrect distances.
  async fn get_point(&self, id: Id) -> Array1<f16> {
    match &*self.mode.read().await {
      Mode::Uncompressed(cache) => {
        cache
          .get_or_compute(id, || self.db.read_node(id))
          .await
          .vector
      }
      Mode::Compressed(compressor, cache) => {
        let compressed = cache
          .get_or_compute(id, async || {
            let node = self.db.read_node(id).await;
            compressor.compress(&node.vector.view())
          })
          .await;
        compressor.decompress(&compressed)
      }
    }
  }

  async fn get_node(&self, id: Id) -> NodeData {
    match &*self.mode.read().await {
      Mode::Uncompressed(cache) => cache.get_or_compute(id, || self.db.read_node(id)).await,
      Mode::Compressed(..) => self.db.read_node(id).await,
    }
  }

  /// Calculate the distance between two DB nodes.
  async fn dist(&self, a: Id, b: Id) -> f32 {
    let (a, b) = tokio::join!(self.get_point(a), self.get_point(b));
    (self.metric)(&a.view(), &b.view())
  }

  /// Calculate the distance between a DB node and a query.
  async fn dist2(&self, a: Id, b: &ArrayView1<'_, f16>) -> f32 {
    (self.metric)(&self.get_point(a).await.view(), b)
  }

  async fn prune_candidates(
    &self,
    node: &ArrayView1<'_, f16>,
    candidate_ids: impl IntoIterator<Item = Id>,
  ) -> Vec<Id> {
    let max_edges = self.cfg.max_edges();
    let dist_thresh = self.cfg.distance_threshold();

    let mut candidates = candidate_ids
      .into_iter()
      .map_concurrent_unordered(async |candidate_id| PointDist {
        id: candidate_id,
        dist: self.dist2(candidate_id, node).await,
      })
      .collect_vec()
      .await
      .into_iter()
      .sorted_unstable_by_key(|s| OrderedFloat(s.dist))
      .collect::<VecDeque<_>>();

    let mut new_neighbors = Vec::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some(PointDist { id: p_star, .. }) = candidates.pop_front() {
      new_neighbors.push(p_star);
      if new_neighbors.len() == max_edges {
        break;
      }
      let should_retain = candidates
        .iter()
        .map_concurrent(async |s| {
          let s_to_p = s.dist;
          let s_to_p_star = self.dist(p_star, s.id).await;
          s_to_p <= s_to_p_star * dist_thresh
        })
        .collect_vec()
        .await;
      candidates = (0..candidates.len())
        .filter(|&i| should_retain[i])
        .map(|i| candidates[i])
        .collect();
    }
    new_neighbors
  }

  // TODO Ignore deleted.
  async fn search(
    &self,
    query: &ArrayView1<'_, f16>,
    k: usize,
    search_list_cap: usize,
  ) -> (Vec<PointDist>, DashSet<Id>) {
    // NOTE: This is intentionally simple over optimized.
    // Not the most optimal data structures or avoiding of malloc/memcpy.
    // And that's OK — simple makes this easier to understand and maintain.
    // The performance is still extremely fast — and probably fits in cache better and branches less.

    assert!(
      search_list_cap >= k,
      "search list capacity must be greater than or equal to k"
    );
    // Our list of seen nodes, always sorted by distance.
    // This is our result list, but also the candidate list for expansion.
    let mut search_list = Vec::<PointDist>::new();
    // Seen != expansion. We just want to prevent duplicate nodes from being added to the search list.
    // Use DashSet as we'll insert from for_each_concurrent.
    let seen = DashSet::new();
    // There's no need to expand the same node twice.
    let mut expanded = HashSet::new();

    // Start with the entry node.
    search_list.push(PointDist {
      id: 0,
      dist: self.dist2(0, query).await,
    });
    seen.insert(0);

    loop {
      // Pop and mark beam_width nodes for expansion.
      // We pop as we'll later re-rank then re-insert with updated dists.
      let to_expand = search_list
        .extract_if(|p| expanded.insert(p.id))
        .take(self.cfg.beam_width())
        .collect_vec();
      if to_expand.is_empty() {
        break;
      };

      // Concurrently fetch all at once — don't block on or serialize I/O.
      let fetched = to_expand
        .into_iter()
        .map_concurrent_unordered(async |node| {
          let data = self.get_node(node.id).await;
          (node, data.neighbors, data.vector)
        })
        .collect_vec()
        .await;

      // Add expanded neighbors to search list.
      let to_add = Mutex::new(Vec::<PointDist>::new());
      iter(fetched)
        .for_each_concurrent(None, async |(mut node, mut neighbors, full_vec)| {
          // Re-rank using full vector.
          node.dist = (self.metric)(&full_vec.view(), query);
          to_add.lock().push(node);

          // There may be additional neighbors.
          if let Some(add) = self.add_edges.get(&node.id) {
            neighbors.extend(add.as_slice());
          };
          iter(neighbors)
            .for_each_concurrent(None, async |neighbor| {
              // We've seen this node in a previous search iteration,
              // or in this iteration — but from another node's expansion.
              if !seen.insert(neighbor) {
                return;
              }
              let dist = self.dist2(neighbor, query).await;
              to_add.lock().push(PointDist { id: neighbor, dist });
            })
            .await;
        })
        .await;
      // WARNING: If you want to optimize by batching inserts, be careful:
      // Two source values to add could be inserted at the same position but between themselves are not sorted.
      // Remember to handle this scenario.
      for node in to_add.into_inner() {
        let pos = search_list
          .binary_search_by_key(&OrderedFloat(node.dist), |s| OrderedFloat(s.dist))
          .map_or_else(identity, identity);
        search_list.insert(pos, node);
      }

      // Without truncation, we'll search the entire graph.
      search_list.truncate(search_list_cap);
    }

    search_list.truncate(k);
    (search_list, seen)
  }

  pub async fn open(dir: impl AsRef<Path>) -> Arc<Roxanne> {
    let db = Arc::new(Db::open(dir).await);
    debug!("opened database");

    let cfg_raw: CfgRaw = db.read_cfg().await;
    let cfg = Cfg::new(cfg_raw);
    debug!("loaded config");

    let deleted = DashSet::new();
    db.iter_deleted()
      .for_each(async |id| {
        deleted.insert(id);
      })
      .await;
    debug!(count = deleted.len(), "loaded deleted");

    let add_edges = DashMap::new();
    db.iter_add_edges()
      .for_each(async |(id, add)| {
        add_edges.insert(id, add);
      })
      .await;
    debug!(nodes = add_edges.len(), "loaded additional edges");

    let count: AtomUsz = db.read_count().await.into();
    let next_id: AtomUsz = db.read_next_id().await.into();
    let dim = AtomUsz::new(0);
    if let Some(n) = db.iter_nodes().next().await {
      dim.set(n.1.vector.len());
    };
    let metric = cfg.metric().get_fn();
    debug!(
      dim = dim.get(),
      count = count.get(),
      next_id = next_id.get(),
      "loaded state"
    );

    let mode = if count.get() > cfg.compression_threshold() {
      let compressor: Option<Box<dyn Compressor>> = match cfg.compression_mode() {
        CompressionMode::PQ => db.maybe_read_pq_model().await.map(|pq| {
          let compressor: Box<dyn Compressor> = Box::new(pq);
          compressor
        }),
        CompressionMode::Trunc => Some(Box::new(TruncCompressor::new(cfg.trunc_dims()))),
      };
      match compressor {
        Some(c) => Mode::Compressed(c, Cache::new()),
        None => Mode::Uncompressed(Cache::new()),
      }
    } else {
      Mode::Uncompressed(Cache::new())
    };

    let (update_sender, update_receiver) = flume::unbounded();
    let rox = Arc::new(Roxanne {
      add_edges,
      cfg,
      count,
      dim,
      db: db.clone(),
      deleted,
      metric,
      next_id,
      update_sender,
      mode: tokio::sync::RwLock::new(mode),
    });

    // Spawn updater thread.
    let (oneshot_tx, oneshot_rx) = tokio::sync::oneshot::channel::<Arc<Roxanne>>();
    tokio::spawn({
      async move {
        let roxanne = oneshot_rx.await.unwrap();
        updater_thread(roxanne, update_receiver).await;
      }
    });
    let Ok(_) = oneshot_tx.send(rox.clone()) else {
      unreachable!();
    };
    debug!("spawned updater thread");

    rox
  }

  pub async fn query<'ref_, 'array>(
    &self,
    query: &'ref_ ArrayView1<'array, f16>,
    k: usize,
  ) -> Vec<(String, f32)> {
    let res = self
      .search(query, k, self.cfg.query_search_list_cap())
      .await
      .0;
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

  pub async fn insert(&self, entries: impl IntoIterator<Item = (String, Array1<f16>)>) {
    iter(entries)
      .for_each_concurrent(None, async |(k, v)| {
        let (signal, ctl) = SignalFuture::new();
        self
          .update_sender
          // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
          .send_async(Update::Insert(k, v.mapv(|e| nan_to_num(e)), ctl))
          .await
          .unwrap();
        signal.await;
      })
      .await;
  }

  pub async fn delete(&self, key: &str) {
    let (signal, ctl) = SignalFuture::new();
    self
      .update_sender
      .send_async(Update::Delete(key.to_string(), ctl))
      .await
      .unwrap();
    signal.await;
  }

  pub fn dim(&self) -> usize {
    self.dim.get()
  }
}
