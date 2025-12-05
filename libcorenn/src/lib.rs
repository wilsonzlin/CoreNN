// Note: avx512_target_feature, path_add_extension, stdarch_x86_avx512 are now stable
#![feature(duration_millis_float)]
#![feature(f16)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_f16))]
#![cfg_attr(
  any(target_arch = "x86", target_arch = "x86_64"),
  feature(stdarch_x86_avx512_f16)
)]

use ahash::HashSet;
use ahash::HashSetExt;
use arbitrary_lock::ArbitraryLock;
use cache::new_cv_cache;
use cache::new_node_cache;
use cache::CVCache;
use cache::NodeCache;
use cfg::Cfg;
use cfg::CompressionMode;
use common::nan_to_num;
use common::Id;
use compaction::compact;
use compressor::pq::ProductQuantizer;
use compressor::scalar::ScalarQuantizer;
use compressor::trunc::TruncCompressor;
use compressor::Compressor;
use compressor::DistanceTable;
use compressor::CV;
use dashmap::DashMap;
use dashmap::DashSet;
use itertools::Itertools;
use metric::Metric;
use metric::StdMetric;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use parking_lot::RwLock;
use std::cmp::max;
use std::collections::VecDeque;
use std::convert::identity;
use std::iter::zip;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::spawn;
use store::in_memory::InMemoryStore;
use store::rocksdb::RocksDBStore;
use store::schema::DbNodeData;
use store::schema::ADD_EDGES;
use store::schema::CFG;
use store::schema::DELETED;
use store::schema::ID_TO_KEY;
use store::schema::KEY_TO_ID;
use store::schema::NODE;
use store::schema::PQ_MODEL;
use store::schema::SQ_MODEL;
use store::Store;
use tracing::debug;
use util::AtomUsz;
use vec::VecData;

pub mod cache;
pub mod cfg;
pub mod common;
pub mod compaction;
pub mod compressor;
pub mod metric;
pub mod store;
pub mod util;
pub mod vec;

enum Mode {
  // Lazy cache of DbNodeData in-memory.
  // We don't prepopulate as that makes start time unnecessarily long.
  Uncompressed(NodeCache),
  // Second element is a cache of compressed vectors.
  // Caching isn't to save computation, it's to avoid DB roundtrip (same as Uncompressed).
  // In compressed mode, graph edges are always fetched from DB.
  Compressed(Arc<dyn Compressor>, CVCache),
}

// It's possible to transition to compressed mode during a query.
// As such, we clone the compressor and make it possible to get distance between compressed and uncompressed vectors.
// This should almost never happen, but implemented to avoid panicking or giving up entirely (e.g. returning NaN).
#[derive(Debug)]
enum PointVec {
  Uncompressed(Arc<VecData>),
  Compressed(Arc<dyn Compressor>, CV),
}

// DbNodeData vector + some DB cfg + ergonomic types and helper methods.
#[derive(Debug)]
struct Point {
  id: Id,
  vec: PointVec,
  metric_type: StdMetric,
  metric: Metric,
  // Optional convenient slot to store distance to something. Not set initially when getting node.
  dist: OrderedFloat<f64>,
}

impl Point {
  pub fn dist(&self, other: &Point) -> f64 {
    match (&self.vec, &other.vec) {
      (PointVec::Uncompressed(a), PointVec::Uncompressed(b)) => (self.metric)(a, b),
      (PointVec::Compressed(c, a), PointVec::Compressed(_c, b)) => c.dist(self.metric_type, a, b),
      (PointVec::Uncompressed(u), PointVec::Compressed(c, b))
      | (PointVec::Compressed(c, b), PointVec::Uncompressed(u)) => {
        c.dist(self.metric_type, &c.compress(u), b)
      }
    }
  }

  #[allow(dead_code)]
  pub fn dist_query(&self, query: &VecData) -> f64 {
    self.dist_query_with_table(query, None)
  }
  
  /// Compute distance to query, using ADC table if available for faster computation.
  pub fn dist_query_with_table(&self, query: &VecData, table: Option<&DistanceTable>) -> f64 {
    match &self.vec {
      PointVec::Uncompressed(v) => (self.metric)(v, query),
      PointVec::Compressed(c, cv) => c.dist_query(query, cv, self.metric_type, table),
    }
  }
}

pub struct State {
  add_edges: DashMap<Id, Vec<Id>>,
  // Cfg is not configurable during runtime (requires a restart).
  // Otherwise, we'd have to handle at any time: changing metric, compression mode, etc.
  cfg: Cfg,
  // True if compaction is currently running.
  compaction_check: Mutex<bool>,
  // True if compression is enabled.
  compression_transition_check: Mutex<bool>,
  count: AtomUsz,
  db: Arc<dyn Store>,
  deleted: DashSet<Id>,
  first_insert_lock: Mutex<bool>,
  key_lock: ArbitraryLock<String, Mutex<()>>,
  // Node reader don't need to check lock, only writers (to the same node) need to synchronize.
  node_write_lock: ArbitraryLock<Id, Mutex<()>>,
  metric: Metric,
  mode: RwLock<Mode>,
  next_id: AtomUsz,
}

#[derive(Clone)]
pub struct CoreNN(Arc<State>);

impl Deref for CoreNN {
  type Target = State;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl CoreNN {
  /// For internal use only. No guarantees about DB schema or state.
  pub fn internal_db(&self) -> &Arc<dyn Store> {
    &self.db
  }

  pub fn cfg(&self) -> &Cfg {
    &self.cfg
  }

  /// Some IDs may not be returned, if they don't exist, or due to subtle eventual consistency issues.
  fn get_nodes(&self, ids: &[Id]) -> Vec<Option<Arc<DbNodeData>>> {
    match &*self.mode.read() {
      Mode::Uncompressed(node_cache) => node_cache.multi_get(ids),
      Mode::Compressed(..) => self
        .db
        .read_ents(&NODE, ids.iter())
        .into_iter()
        .map(|n| n.map(|n| Arc::new(n)))
        .collect_vec(),
    }
  }

  /// Some IDs may not be returned, if they don't exist, or due to subtle eventual consistency issues.
  fn get_points<'a>(
    &'a self,
    ids: &'a [Id],
    query: Option<&'a VecData>,
  ) -> impl Iterator<Item = Option<Point>> + 'a {
    self.get_points_with_table(ids, query, None)
  }
  
  /// Get points with optional ADC distance table for faster compressed distance computation.
  fn get_points_with_table<'a>(
    &'a self,
    ids: &'a [Id],
    query: Option<&'a VecData>,
    dist_table: Option<&'a DistanceTable>,
  ) -> impl Iterator<Item = Option<Point>> + 'a {
    // Hold lock across all reads. Getting some compressed nodes and others uncompressed breaks all code that uses this data.
    let vecs = match &*self.mode.read() {
      Mode::Uncompressed(node_cache) => node_cache
        .multi_get(ids)
        .into_iter()
        .map(|raw| {
          let raw = raw?;
          Some(PointVec::Uncompressed(raw.vector.clone()))
        })
        .collect_vec(),
      Mode::Compressed(compressor, cache) => cache
        .multi_get(ids)
        .into_iter()
        .map(|cv| Some(PointVec::Compressed(compressor.clone(), cv?)))
        .collect_vec(),
    };
    zip(ids, vecs).map(move |(&id, vec)| {
      let vec = vec?;
      let mut node = Point {
        id,
        vec,
        metric: self.metric,
        metric_type: self.cfg.metric,
        dist: OrderedFloat(f64::INFINITY),
      };
      if let Some(q) = query {
        node.dist.0 = node.dist_query_with_table(q, dist_table);
      }
      Some(node)
    })
  }

  #[allow(dead_code)]
  fn get_point(&self, id: Id, query: Option<&VecData>) -> Option<Point> {
    self.get_points(&[id], query).exactly_one().ok().unwrap()
  }

  fn prune_candidates(&self, node: &VecData, candidate_ids: &[Id]) -> Vec<Id> {
    let max_edges = self.cfg.max_edges;
    let dist_thresh = self.cfg.distance_threshold;

    let mut candidates = self
      .get_points(candidate_ids, Some(node))
      .filter_map(|n| n)
      .sorted_unstable_by_key(|s| s.dist)
      .collect::<VecDeque<_>>();

    let mut new_neighbors = Vec::new();
    // Even though the algorithm in the paper doesn't actually pop, the later pruning of the candidates at the end of the loop guarantees it will always be removed because d(p*, p') will always be zero for itself (p* == p').
    while let Some(p_star) = candidates.pop_front() {
      new_neighbors.push(p_star.id);
      if new_neighbors.len() == max_edges {
        break;
      }
      candidates.retain(|s| {
        let cand_dist_to_node = s.dist.0;
        let cand_dist_to_pick = p_star.dist(s);
        cand_dist_to_node <= cand_dist_to_pick * dist_thresh
      });
    }
    new_neighbors
  }

  fn search(&self, query: &VecData, k: usize, search_list_cap: usize) -> (Vec<Point>, DashSet<Id>) {
    // NOTE: This is intentionally simple over optimized.
    // Not the most optimal data structures or avoiding of malloc/memcpy.
    // And that's OK — simple makes this easier to understand and maintain.
    // The performance is still extremely fast — and probably fits in cache better and branches less.

    assert!(
      search_list_cap >= k,
      "search list capacity must be greater than or equal to k"
    );
    
    // Create ADC distance table for fast compressed distance computation.
    // This is created once and reused for all distance computations in this search.
    let dist_table: Option<DistanceTable> = match &*self.mode.read() {
      Mode::Compressed(compressor, _) => compressor.create_distance_table(query, self.cfg.metric),
      Mode::Uncompressed(_) => None,
    };
    let dist_table_ref = dist_table.as_ref();
    
    // Our list of candidate nodes, always sorted by distance.
    // This is our result list, but also the candidate list for expansion.
    let mut search_list = Vec::<Point>::new();
    // Seen != expansion. We just want to prevent duplicate nodes from being added to the search list.
    // Use DashSet as we'll insert from for_each_concurrent.
    let seen = DashSet::new();
    // There's no need to expand the same node more than once.
    let mut expanded = HashSet::new();
    
    // Early termination tracking: if the best k results haven't improved in
    // several iterations, we can stop early.
    let mut stale_iterations = 0;
    let max_stale_iterations = 3; // Stop if no improvement for 3 iterations
    let mut prev_best_dist = f64::INFINITY;

    // Start with the entry node.
    let Some(entry) = self.get_points_with_table(&[0], Some(query), dist_table_ref).next().flatten() else {
      // No entry node, empty DB.
      return Default::default();
    };
    search_list.push(entry);
    seen.insert(0);

    loop {
      // Pop and mark beam_width nodes for expansion.
      // We pop as we'll later re-rank then re-insert with updated dists.
      let to_expand = search_list
        .extract_if(.., |p| expanded.insert(p.id))
        .take(self.cfg.beam_width)
        .collect_vec();
      if to_expand.is_empty() {
        break;
      };

      let fetched = self.get_nodes(&to_expand.iter().map(|p| p.id).collect_vec());

      // Add expanded neighbors to search list.
      let mut to_add = Vec::<Point>::new();
      let mut neighbor_ids = Vec::<Id>::new();
      for (mut point, node) in zip(to_expand, fetched) {
        // Node doesn't exist anymore.
        let Some(node) = node else {
          continue;
        };

        // Collect its neighbors to total set of neighbors.
        for &neighbor in node.neighbors.iter() {
          // We've seen this node in a previous search iteration,
          // or in this iteration — but from another node's expansion.
          if !seen.insert(neighbor) {
            continue;
          }
          neighbor_ids.push(neighbor);
        }
        // There may be additional neighbors.
        if let Some(add) = self.add_edges.get(&point.id) {
          for &neighbor in add.iter() {
            if !seen.insert(neighbor) {
              continue;
            }
            neighbor_ids.push(neighbor);
          }
        };

        // Re-rank using full vector.
        point.dist.0 = (self.metric)(&node.vector, query);
        to_add.push(point);
      }
      // Get all neighbors at once, using ADC for fast distance computation.
      for p in self.get_points_with_table(&neighbor_ids, Some(query), dist_table_ref).flatten() {
        to_add.push(p);
      }

      // WARNING: If you want to optimize by batching inserts, be careful:
      // Two source values to add could be inserted at the same position but between themselves are not sorted.
      // Remember to handle this scenario.
      for point in to_add {
        // Remove soft-deleted if already expanded. We still need to expand soft-deleted to traverse the graph accurately.
        if self.deleted.contains(&point.id) && expanded.contains(&point.id) {
          continue;
        }
        let pos = search_list
          .binary_search_by_key(&point.dist, |s| s.dist)
          .map_or_else(identity, identity);
        search_list.insert(pos, point);
      }

      // Without truncation each iteration, we'll search the entire graph.
      search_list.truncate(search_list_cap);
      
      // Early termination check: has the k-th best distance improved?
      if search_list.len() >= k {
        let current_kth_dist = search_list[k - 1].dist.0;
        // Check if we've made meaningful progress (at least 0.1% improvement)
        if current_kth_dist >= prev_best_dist * 0.999 {
          stale_iterations += 1;
          if stale_iterations >= max_stale_iterations {
            // No improvement - terminate early
            break;
          }
        } else {
          stale_iterations = 0;
          prev_best_dist = current_kth_dist;
        }
      }
    }

    // We use `seen` as candidates for new neighbors, so we should remove soft-deleted here too to avoid new edges to them.
    // It's not necessary for correctness, but good to have. (We'll have to prune these edges later during compaction if we don't.)
    seen.retain(|id| !self.deleted.contains(id));
    search_list.truncate(k);
    (search_list, seen)
  }

  fn new(
    dir: Option<impl AsRef<Path>>,
    init_cfg: Option<Cfg>,
    create_if_missing: bool,
    error_if_exists: bool,
  ) -> CoreNN {
    let db: Arc<dyn Store> = match dir {
      Some(dir) => Arc::new(RocksDBStore::open(dir, create_if_missing, error_if_exists)),
      None => Arc::new(InMemoryStore::new()),
    };
    debug!("opened database");

    let is_empty = NODE.iter(&db).next().is_none();

    let cfg = if let Some(cfg) = init_cfg {
      CFG.put(&db, (), &cfg);
      cfg
    } else {
      CFG.read(&db, ()).unwrap()
    };

    debug!(
      beam_width = cfg.beam_width,
      max_edges = cfg.max_edges,
      max_add_edges = cfg.max_add_edges,
      metric = ?cfg.metric,
      query_search_list_cap = cfg.query_search_list_cap,
      update_search_list_cap = cfg.update_search_list_cap,
      "loaded config"
    );

    let deleted = DashSet::new();
    DELETED.iter(&db).for_each(|(id, _)| {
      deleted.insert(id);
    });
    debug!(count = deleted.len(), "loaded deleted");

    let add_edges = DashMap::new();
    // 0 is reserved for the internal entry node.
    let mut next_id = 1;
    // Including soft-deleted.
    let mut count = 0;
    // All nodes must have an ADD_EDGES entry, even if empty.
    // Since we iterate over all it, we also use it to track next_id and count.
    // Soft-deleted nodes still have ADD_EDGES. Truly deleted nodes aren't mentioned anywhere. Therefore, scanning ADD_EDGES is enough for correct next_id.
    ADD_EDGES.iter(&db).for_each(|(id, add)| {
      add_edges.insert(id, add);
      next_id = next_id.max(id + 1);
      count += 1;
    });

    let metric = cfg.metric.get_fn();
    debug!(next_id, count, "loaded state");

    let mode = if count > cfg.compression_threshold {
      let compressor: Option<Arc<dyn Compressor>> = match cfg.compression_mode {
        CompressionMode::PQ => PQ_MODEL.read(&db, ()).map(|pq| {
          let compressor: Arc<dyn Compressor> = Arc::new(pq);
          compressor
        }),
        CompressionMode::SQ => SQ_MODEL.read(&db, ()).map(|sq| {
          let compressor: Arc<dyn Compressor> = Arc::new(sq);
          compressor
        }),
        CompressionMode::Trunc => Some(Arc::new(TruncCompressor::new(cfg.trunc_dims))),
      };
      match compressor {
        Some(c) => Mode::Compressed(c.clone(), new_cv_cache(db.clone(), c)),
        None => Mode::Uncompressed(new_node_cache(db.clone())),
      }
    } else {
      Mode::Uncompressed(new_node_cache(db.clone()))
    };

    CoreNN(Arc::new(State {
      add_edges,
      cfg,
      compaction_check: Mutex::new(false),
      compression_transition_check: Mutex::new(matches!(mode, Mode::Compressed(..))),
      count: count.into(),
      db,
      deleted,
      first_insert_lock: Mutex::new(is_empty),
      key_lock: ArbitraryLock::new(),
      metric,
      mode: RwLock::new(mode),
      next_id: next_id.into(),
      node_write_lock: ArbitraryLock::new(),
    }))
  }

  pub fn create(dir: impl AsRef<Path>, cfg: Cfg) -> CoreNN {
    Self::new(Some(dir), Some(cfg), true, true)
  }

  pub fn open(dir: impl AsRef<Path>) -> CoreNN {
    Self::new(Some(dir), None, false, false)
  }

  pub fn new_in_memory(cfg: Cfg) -> CoreNN {
    Self::new(None::<PathBuf>, Some(cfg), false, true)
  }

  pub fn query<D>(&self, query: &[D], k: usize) -> Vec<(String, f64)>
  where
    D: num::Float,
    VecData: From<Vec<D>>,
  {
    let query = VecData::from(nan_to_num(query));
    self.query_vec(query, k)
  }

  /// WARNING: `query` must not contain any NaN values.
  /// It's possible to get less than k results due to data changes during the query.
  pub fn query_vec(&self, query: VecData, k: usize) -> Vec<(String, f64)> {
    let res = self
      .search(&query, k, max(k, self.cfg.query_search_list_cap))
      .0;
    let keys = self.db.read_ents(&ID_TO_KEY, res.iter().map(|r| r.id));
    zip(keys, res)
      .filter_map(|(k, r)| k.map(|k| (k, r.dist.0)))
      .collect()
  }

  fn maybe_compact(&self) {
    if self.deleted.len() < self.cfg.compaction_threshold_deletes {
      return;
    };

    let mut is_compacting = self.compaction_check.lock();
    if *is_compacting {
      return;
    };
    *is_compacting = true;
    drop(is_compacting);

    let corenn = self.clone();
    spawn(move || {
      compact(&corenn);
      let mut is_compacting = corenn.compaction_check.lock();
      *is_compacting = false;
    });
  }

  fn maybe_enable_compression(&self) {
    if self.count.get() <= self.cfg.compression_threshold {
      return;
    };

    let mut is_enabled = self.compression_transition_check.lock();
    if *is_enabled {
      // Already enabled or in process of being enabled.
      return;
    };
    *is_enabled = true;
    // Don't block subsequent calls to `maybe_enable_compression`.
    drop(is_enabled);

    let corenn = self.clone();
    spawn(move || {
      tracing::warn!(
        threshold = corenn.cfg.compression_threshold,
        n = corenn.count.get(),
        "enabling compression"
      );
      let compressor: Arc<dyn Compressor> = match corenn.cfg.compression_mode {
        CompressionMode::PQ => {
          let pq = ProductQuantizer::<f32>::train_from_corenn(&corenn);
          PQ_MODEL.put(&corenn.db, (), &pq);
          Arc::new(pq)
        }
        CompressionMode::SQ => {
          let sq = ScalarQuantizer::train_from_corenn(&corenn);
          SQ_MODEL.put(&corenn.db, (), &sq);
          Arc::new(sq)
        }
        CompressionMode::Trunc => Arc::new(TruncCompressor::new(corenn.cfg.trunc_dims)),
      };
      *corenn.mode.write() = Mode::Compressed(
        compressor.clone(),
        new_cv_cache(corenn.db.clone(), compressor),
      );
      tracing::info!("enabled compression");
    });
  }

  pub fn insert<D>(&self, key: &String, vec: &[D])
  where
    D: num::Float,
    VecData: From<Vec<D>>,
  {
    // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
    let vec = VecData::from(nan_to_num(vec));
    self.insert_vec(key, vec)
  }

  /// WARNING: `vec` must not contain any NaN values.
  pub fn insert_vec(&self, key: &String, vec: VecData) {
    let vec = Arc::new(vec);
    let id = self.next_id.inc();
    // Lock during the entire insert. Correctness is much more complex otherwise.
    // Inserting a key multiple times in parallel is rare/a sign of a bug.
    let lock = self.key_lock.get(key.to_string());
    let _g = lock.lock();
    let mut txn = Vec::new();
    if let Some(existing_id) = KEY_TO_ID.read(&self.db, key) {
      ID_TO_KEY.batch_delete(&mut txn, existing_id);
      DELETED.batch_put(&mut txn, existing_id, ());
      self.deleted.insert(existing_id);
    }
    ID_TO_KEY.batch_put(&mut txn, id, key);
    KEY_TO_ID.batch_put(&mut txn, key, id);
    // All nodes must have an ADD_EDGES entry, even if empty.
    // (See `CoreNN::open`.)
    ADD_EDGES.batch_put(&mut txn, id, &vec![]);

    {
      let mut is_first = self.first_insert_lock.lock();
      if *is_first {
        *is_first = false;
        // We need to continue holding the lock throughout the whole insert, as otherwise other inserts will be expecting a 0 node but it won't exist, and will be detached.
        debug!("first graph update");

        // Insert internal copy as permanent entry node.
        NODE.batch_put(&mut txn, 0usize, DbNodeData {
          version: 0,
          neighbors: vec![id],
          vector: vec.clone(),
        });
        ADD_EDGES.batch_put(&mut txn, 0, vec![]);
        // Insert node.
        NODE.batch_put(&mut txn, id, DbNodeData {
          version: 0,
          neighbors: vec![0],
          vector: vec.clone(),
        });
        ADD_EDGES.batch_put(&mut txn, id, vec![]);

        self.db.write(txn);
        return;
      };
    };

    let candidates = self.search(&vec, 1, self.cfg.update_search_list_cap).1;
    let neighbors = self.prune_candidates(&vec, &candidates.into_iter().collect_vec());
    NODE.batch_put(&mut txn, id, DbNodeData {
      version: 0,
      neighbors: neighbors.clone(),
      vector: vec.clone(),
    });

    // Backedges.
    for j in neighbors {
      // We lock one at a time (instead of across all) to avoid deadlocks.
      // If we don't manage to commit the insert, but do commit some backedges, that's safe, since our code is resilient to edges to non-existent nodes.
      let lock = self.node_write_lock.get(j);
      let _g = lock.lock();
      // We intentionally prune before, not after, adding new backneighbors.
      // Otherwise, we'd have to handle these add_edges to new nodes that aren't yet available in the DB.
      // From experience, that makes the system much more complex, subtle, and error-prone, as it breaks the simple source-of-truth and consistency invariants.
      // This keeps the code and design simple and correct, a worthwhile trade-off.
      // It may seem wasteful to not prune with new neighbors together in one go, but the next time this node's touched will just add more add_edges again anyway.
      let mut add_edges = self
        .add_edges
        .get(&j)
        .map(|e| e.clone())
        .unwrap_or_default();
      if add_edges.len() + 1 >= self.cfg.max_add_edges {
        // We need to prune this neighbor.
        // TODO For now, we always read from DB to avoid any subtle cache stale race conditions, since we hold a write lock so we are the only writer + we are always reading from DB so we'll always get the latest version. Is it necessary?
        let Some(DbNodeData {
          version,
          mut neighbors,
          vector,
        }) = NODE.read(&self.db, j)
        else {
          // Eventual consistency: node is gone.
          continue;
        };
        neighbors.extend_from_slice(&add_edges);
        neighbors = self.prune_candidates(&vector, &neighbors);
        let new_node = DbNodeData {
          version: version + 1,
          neighbors,
          vector,
        };
        NODE.batch_put(&mut txn, j, &new_node);
        if let Mode::Uncompressed(cache) = &*self.mode.read() {
          cache.insert(j, Arc::new(new_node));
        };
        add_edges.clear();
      }
      add_edges.push(id);
      ADD_EDGES.batch_put(&mut txn, j, &add_edges);
      self.add_edges.insert(j, add_edges);
    }

    self.db.write(txn);

    self.maybe_enable_compression();
    self.maybe_compact();
  }

  pub fn delete(&self, key: &String) {
    let lock = self.key_lock.get(key.to_string());
    let _g = lock.lock();
    let mut txn = Vec::new();
    let Some(existing_id) = KEY_TO_ID.read(&self.db, key) else {
      return;
    };
    ID_TO_KEY.batch_delete(&mut txn, existing_id);
    KEY_TO_ID.batch_delete(&mut txn, key);
    DELETED.batch_put(&mut txn, existing_id, ());
    self.deleted.insert(existing_id);
    self.db.write(txn);

    self.maybe_compact();
  }
}
