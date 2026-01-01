use crate::error::Error;
use crate::error::Result;
use crate::id::NodeId;
use crate::metric::Metric;
use crate::vectors::VectorStore;
use crate::vectors::VectorStoreMut;
use crate::vectors::VectorRef;
use crate::visited::VisitedListPool;
use ahash::HashSet;
use ahash::HashSetExt;
use ahash::RandomState;
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::f64;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;
use tracing::warn;

const DEFAULT_LABEL_OPERATION_LOCKS: usize = 65_536;
const NODE_ID_NONE: u32 = u32::MAX;
const HNSW_DATA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HnswConfig {
  pub dim: usize,
  pub max_nodes: usize,
  pub m: usize,
  pub ef_construction: usize,
  pub ef_search: usize,
  pub seed: u64,
  pub label_operation_locks: usize,
}

impl HnswConfig {
  pub fn new(dim: usize, max_nodes: usize) -> Self {
    Self {
      dim,
      max_nodes,
      m: 16,
      ef_construction: 200,
      ef_search: 50,
      seed: 100,
      label_operation_locks: DEFAULT_LABEL_OPERATION_LOCKS,
    }
  }

  pub fn m(mut self, m: usize) -> Self {
    self.m = m;
    self
  }

  pub fn ef_construction(mut self, ef: usize) -> Self {
    self.ef_construction = ef;
    self
  }

  pub fn ef_search(mut self, ef: usize) -> Self {
    self.ef_search = ef;
    self
  }

  pub fn seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  pub fn label_operation_locks(mut self, locks: usize) -> Self {
    self.label_operation_locks = locks;
    self
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOutcome {
  Inserted,
  Updated,
  Resurrected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit<K> {
  pub key: K,
  pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HnswData<K> {
  pub version: u32,
  pub cfg: HnswConfig,
  pub entry_point: Option<u32>,
  pub max_level: i32,
  /// Node keys in `NodeId` order.
  pub keys: Vec<K>,
  /// Per-node tombstone flag (0/1), in `NodeId` order.
  pub deleted: Vec<u8>,
  /// Per-node max level, in `NodeId` order.
  pub levels: Vec<u32>,
  /// For each node, for each level `0..=levels[node]` (in that order), the neighbor list length.
  pub neighbor_counts: Vec<u16>,
  /// Concatenated neighbor ids for all lists described by `neighbor_counts`.
  pub neighbors: Vec<u32>,
}

fn neighbors_count(header: u32) -> usize {
  (header & 0xffff) as usize
}

fn pack_neighbors_count(cnt: usize) -> Result<u32> {
  if cnt > u16::MAX as usize {
    return Err(Error::InvalidIndexFormat("neighbor list too large".to_string()));
  }
  Ok(cnt as u32)
}

#[derive(Clone, Copy)]
struct NeighborList<'a> {
  data: &'a [AtomicU32],
  len: usize,
}

impl<'a> IntoIterator for NeighborList<'a> {
  type Item = NodeId;
  type IntoIter = NeighborIter<'a>;

  fn into_iter(self) -> Self::IntoIter {
    NeighborIter {
      data: self.data,
      idx: 0,
      end: self.len,
    }
  }
}

struct NeighborIter<'a> {
  data: &'a [AtomicU32],
  idx: usize,
  end: usize,
}

impl Iterator for NeighborIter<'_> {
  type Item = NodeId;

  fn next(&mut self) -> Option<Self::Item> {
    if self.idx >= self.end {
      return None;
    }
    let raw = self.data[self.idx].load(Ordering::Relaxed);
    self.idx += 1;
    Some(NodeId::new(raw))
  }
}

pub struct Hnsw<K, M>
where
  K: Eq + Hash + Clone + Send + Sync + 'static,
  M: Metric,
{
  metric: M,
  cfg: HnswConfig,

  max_m: usize,
  max_m0: usize,

  mult: f64,

  /// Prevents resizing/serialization from racing concurrent mutations/search.
  mutation_lock: RwLock<()>,

  /// Serializes operations on the same key (hashed).
  key_locks: Vec<Mutex<()>>,
  key_lock_hasher: RandomState,

  /// Protects `entry_point` and `max_level` updates.
  global: Mutex<()>,

  /// Protects link-list updates per node.
  link_locks: Vec<Mutex<()>>,

  /// External key -> internal node mapping.
  key_to_node: DashMap<K, NodeId, RandomState>,

  /// Internal node -> external key (immutable).
  node_keys: Vec<OnceLock<K>>,

  /// Number of allocated nodes.
  cur_node_count: AtomicUsize,

  /// Number of deleted nodes.
  deleted_count: AtomicUsize,

  /// Per node.
  node_deleted: Vec<AtomicU8>,
  node_level: Vec<AtomicI32>,

  /// Base layer: per node, [header, neighbors...max_m0]
  level0_links: Vec<AtomicU32>,
  /// Upper layers: per node, if level>0, [ [header, neighbors...max_m] * level ]
  upper_links: Vec<OnceLock<Box<[AtomicU32]>>>,

  /// Search parameter (should be > k).
  ef_search: AtomicUsize,

  /// Search visited list pool (per thread handle).
  visited_pool: VisitedListPool,

  level_rng: Mutex<StdRng>,
  update_rng: Mutex<StdRng>,

  ef_construction: usize,
  max_level: AtomicI32,
  entry_point: AtomicU32,
}

impl<K, M> Hnsw<K, M>
where
  K: Eq + Hash + Clone + Send + Sync + 'static,
  M: Metric,
{
  pub fn new(metric: M, mut cfg: HnswConfig) -> Self {
    assert!(cfg.max_nodes <= u32::MAX as usize);
    assert!(cfg.dim > 0, "dim must be > 0");
    assert!(cfg.m >= 2, "m must be >= 2");
    if !cfg.label_operation_locks.is_power_of_two() {
      let next_pow2 = cfg.label_operation_locks.next_power_of_two();
      warn!(
        "label_operation_locks={} is not power-of-two; rounding up to {}",
        cfg.label_operation_locks, next_pow2
      );
      cfg.label_operation_locks = next_pow2;
    }

    let m = if cfg.m <= 10_000 {
      cfg.m
    } else {
      warn!("m={} exceeds 10000; capping to 10000", cfg.m);
      10_000
    };
    cfg.m = m;

    let max_m = m;
    let max_m0 = m * 2;
    let ef_construction = cfg.ef_construction.max(m);
    cfg.ef_construction = ef_construction;

    let mult = 1.0 / (m as f64).ln();

    let level0_words_per_node = 1 + max_m0;
    let level0_total_words = cfg.max_nodes * level0_words_per_node;
    let max_nodes = cfg.max_nodes;

    let mut key_locks = Vec::with_capacity(cfg.label_operation_locks);
    for _ in 0..cfg.label_operation_locks {
      key_locks.push(Mutex::new(()));
    }

    let mut link_locks = Vec::with_capacity(cfg.max_nodes);
    for _ in 0..cfg.max_nodes {
      link_locks.push(Mutex::new(()));
    }

    let mut node_keys = Vec::with_capacity(cfg.max_nodes);
    node_keys.resize_with(cfg.max_nodes, OnceLock::new);

    let mut node_deleted = Vec::with_capacity(cfg.max_nodes);
    node_deleted.resize_with(cfg.max_nodes, || AtomicU8::new(0));

    let mut node_level = Vec::with_capacity(cfg.max_nodes);
    node_level.resize_with(cfg.max_nodes, || AtomicI32::new(0));

    let mut level0_links = Vec::with_capacity(level0_total_words);
    level0_links.resize_with(level0_total_words, || AtomicU32::new(0));

    let mut upper_links = Vec::with_capacity(cfg.max_nodes);
    upper_links.resize_with(cfg.max_nodes, OnceLock::new);

    let seed = cfg.seed;
    let update_seed = seed.wrapping_add(1);
    let ef_search = cfg.ef_search;

    Self {
      metric,
      cfg,
      max_m,
      max_m0,
      mult,
      mutation_lock: RwLock::new(()),
      key_locks,
      key_lock_hasher: RandomState::new(),
      global: Mutex::new(()),
      link_locks,
      key_to_node: DashMap::with_hasher(RandomState::new()),
      node_keys,
      cur_node_count: AtomicUsize::new(0),
      deleted_count: AtomicUsize::new(0),
      node_deleted,
      node_level,
      level0_links,
      upper_links,
      ef_search: AtomicUsize::new(ef_search),
      visited_pool: VisitedListPool::new(16, max_nodes),
      level_rng: Mutex::new(StdRng::seed_from_u64(seed)),
      update_rng: Mutex::new(StdRng::seed_from_u64(update_seed)),
      ef_construction,
      max_level: AtomicI32::new(-1),
      entry_point: AtomicU32::new(NODE_ID_NONE),
    }
  }

  pub fn dim(&self) -> usize {
    self.cfg.dim
  }

  pub fn m(&self) -> usize {
    self.cfg.m
  }

  pub fn ef_construction(&self) -> usize {
    self.ef_construction
  }

  pub fn max_nodes(&self) -> usize {
    self.cfg.max_nodes
  }

  pub fn len(&self) -> usize {
    self.cur_node_count.load(Ordering::Acquire)
  }

  pub fn deleted_len(&self) -> usize {
    self.deleted_count.load(Ordering::Acquire)
  }

  pub fn live_len(&self) -> usize {
    self.len().saturating_sub(self.deleted_len())
  }

  pub fn ef_search(&self) -> usize {
    self.ef_search.load(Ordering::Acquire)
  }

  pub fn set_ef_search(&self, ef: usize) {
    self.ef_search.store(ef, Ordering::Release);
  }

  fn key_lock<'a>(&'a self, key: &K) -> &'a Mutex<()> {
    let mut hasher = self.key_lock_hasher.build_hasher();
    key.hash(&mut hasher);
    let idx = (hasher.finish() as usize) & (self.key_locks.len() - 1);
    &self.key_locks[idx]
  }

  fn entry_point_node(&self) -> Option<NodeId> {
    let raw = self.entry_point.load(Ordering::Acquire);
    (raw != NODE_ID_NONE).then_some(NodeId::new(raw))
  }

  fn is_deleted(&self, node: NodeId) -> bool {
    self.node_deleted[node.as_usize()].load(Ordering::Acquire) != 0
  }

  fn mark_deleted(&self, node: NodeId) -> Result<()> {
    let old = self.node_deleted[node.as_usize()].swap(1, Ordering::AcqRel);
    if old == 0 {
      self.deleted_count.fetch_add(1, Ordering::AcqRel);
    }
    Ok(())
  }

  fn unmark_deleted(&self, node: NodeId) -> Result<bool> {
    let old = self.node_deleted[node.as_usize()].swap(0, Ordering::AcqRel);
    if old != 0 {
      self.deleted_count.fetch_sub(1, Ordering::AcqRel);
      return Ok(true);
    }
    Ok(false)
  }

  fn alloc_node(&self) -> Result<NodeId> {
    let mut cur = self.cur_node_count.load(Ordering::Acquire);
    loop {
      if cur >= self.cfg.max_nodes {
        return Err(Error::IndexFull {
          max_nodes: self.cfg.max_nodes,
        });
      }
      match self.cur_node_count.compare_exchange_weak(
        cur,
        cur + 1,
        Ordering::AcqRel,
        Ordering::Acquire,
      ) {
        Ok(_) => return Ok(NodeId::new(cur as u32)),
        Err(actual) => cur = actual,
      }
    }
  }

  fn get_random_level(&self) -> i32 {
    let mut u: f64 = self.level_rng.lock().gen();
    if u == 0.0 {
      u = f64::MIN_POSITIVE;
    }
    let r = -u.ln() * self.mult;
    r as i32
  }

  fn level0_block_range(&self, node: NodeId) -> std::ops::Range<usize> {
    let words = 1 + self.max_m0;
    let start = node.as_usize() * words;
    start..start + words
  }

  fn level0_block(&self, node: NodeId) -> &[AtomicU32] {
    let r = self.level0_block_range(node);
    &self.level0_links[r]
  }

  fn upper_block(&self, node: NodeId, level: usize) -> Result<&[AtomicU32]> {
    debug_assert!(level > 0);
    let Some(raw) = self
      .upper_links
      .get(node.as_usize())
      .and_then(|c| c.get())
    else {
      return Err(Error::InvalidIndexFormat("missing linklist".to_string()));
    };
    let words_per_level = 1 + self.max_m;
    let start = (level - 1) * words_per_level;
    let end = start + words_per_level;
    if end > raw.len() {
      return Err(Error::InvalidIndexFormat(
        "linklist level out of bounds".to_string(),
      ));
    }
    Ok(&raw[start..end])
  }

  fn neighbors_at_level(&self, node: NodeId, level: usize) -> Result<NeighborList<'_>> {
    let block = if level == 0 {
      self.level0_block(node)
    } else {
      self.upper_block(node, level)?
    };
    let header = block[0].load(Ordering::Acquire);
    let cnt = neighbors_count(header);
    let cap = if level == 0 { self.max_m0 } else { self.max_m };
    if cnt > cap {
      return Err(Error::InvalidIndexFormat("neighbor list too large".to_string()));
    }
    Ok(NeighborList {
      data: &block[1..],
      len: cnt,
    })
  }

  fn set_neighbors_count(header: &AtomicU32, count: usize) -> Result<()> {
    header.store(pack_neighbors_count(count)?, Ordering::Release);
    Ok(())
  }

  fn node_key(&self, node: NodeId) -> Result<&K> {
    self
      .node_keys
      .get(node.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
      .get()
      .ok_or_else(|| Error::InvalidIndexFormat("missing node key".to_string()))
  }

  fn distance_between_nodes<V: VectorStore>(&self, vectors: &V, a: NodeId, b: NodeId) -> Result<f32> {
    let va = vectors.vector(a).ok_or(Error::MissingVector)?;
    let vb = vectors.vector(b).ok_or(Error::MissingVector)?;
    let va = va.as_f32_slice();
    let vb = vb.as_f32_slice();
    if va.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: va.len(),
      });
    }
    if vb.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: vb.len(),
      });
    }
    Ok(self.metric.distance(va, vb))
  }

  fn distance_query_to_node<V: VectorStore>(&self, vectors: &V, query: &[f32], node: NodeId) -> Result<f32> {
    if query.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: query.len(),
      });
    }
    let v = vectors.vector(node).ok_or(Error::MissingVector)?;
    let v = v.as_f32_slice();
    if v.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: v.len(),
      });
    }
    Ok(self.metric.distance(query, v))
  }

  fn get_neighbors_by_heuristic2<V: VectorStore>(
    &self,
    vectors: &V,
    top_candidates: &mut BinaryHeap<(OrderedFloat<f32>, NodeId)>,
    m: usize,
  ) -> Result<()> {
    if top_candidates.len() < m {
      return Ok(());
    }

    let mut queue_closest: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();
    while let Some((dist, id)) = top_candidates.pop() {
      queue_closest.push((OrderedFloat(-dist.0), id));
    }

    let mut return_list: Vec<(OrderedFloat<f32>, NodeId)> = Vec::with_capacity(m);
    while let Some((neg_dist_to_query, cur_id)) = queue_closest.pop() {
      if return_list.len() >= m {
        break;
      }
      let dist_to_query = -neg_dist_to_query.0;

      let mut good = true;
      for &(_, selected_id) in &return_list {
        let cur_dist = self.distance_between_nodes(vectors, selected_id, cur_id)?;
        if cur_dist < dist_to_query {
          good = false;
          break;
        }
      }

      if good {
        return_list.push((neg_dist_to_query, cur_id));
      }
    }

    for (neg_dist, id) in return_list {
      top_candidates.push((OrderedFloat(-neg_dist.0), id));
    }
    Ok(())
  }

  fn search_base_layer<V: VectorStore>(
    &self,
    vectors: &V,
    ep: NodeId,
    query: &[f32],
    layer: usize,
    ef: usize,
    filter: Option<&dyn Fn(&K) -> bool>,
  ) -> Result<BinaryHeap<(OrderedFloat<f32>, NodeId)>> {
    let mut visited = self.visited_pool.get();
    let visited_tag = visited.tag;
    let visited_mass = visited.mass_mut();

    let mut top_candidates: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();
    let mut candidate_set: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();

    let mut lower_bound;
    let ep_key = self.node_key(ep)?;
    let ep_allowed = filter.map(|f| f(ep_key)).unwrap_or(true);
    if !self.is_deleted(ep) && ep_allowed {
      let dist = self.distance_query_to_node(vectors, query, ep)?;
      lower_bound = dist;
      top_candidates.push((OrderedFloat(dist), ep));
      candidate_set.push((OrderedFloat(-dist), ep));
    } else {
      lower_bound = f32::INFINITY;
      candidate_set.push((OrderedFloat(-lower_bound), ep));
    }

    visited_mass[ep.as_usize()] = visited_tag;

    while let Some((neg_dist, cur)) = candidate_set.pop() {
      let cur_dist = -neg_dist.0;
      if cur_dist > lower_bound && top_candidates.len() == ef {
        break;
      }

      for cand in self.neighbors_at_level(cur, layer)? {
        if visited_mass[cand.as_usize()] == visited_tag {
          continue;
        }
        visited_mass[cand.as_usize()] = visited_tag;

        let dist = self.distance_query_to_node(vectors, query, cand)?;
        let should_consider = top_candidates.len() < ef || lower_bound > dist;
        if !should_consider {
          continue;
        }

        candidate_set.push((OrderedFloat(-dist), cand));

        let cand_key = self.node_key(cand)?;
        let cand_allowed = filter.map(|f| f(cand_key)).unwrap_or(true);
        if !self.is_deleted(cand) && cand_allowed {
          top_candidates.push((OrderedFloat(dist), cand));
        }

        while top_candidates.len() > ef {
          top_candidates.pop();
        }
        if let Some((worst, _)) = top_candidates.peek() {
          lower_bound = worst.0;
        }
      }
    }

    Ok(top_candidates)
  }

  fn mutually_connect_new_element<V: VectorStore>(
    &self,
    vectors: &V,
    node: NodeId,
    mut top_candidates: BinaryHeap<(OrderedFloat<f32>, NodeId)>,
    level: usize,
    is_update: bool,
  ) -> Result<NodeId> {
    let cap = if level == 0 { self.max_m0 } else { self.max_m };
    self.get_neighbors_by_heuristic2(vectors, &mut top_candidates, self.cfg.m)?;
    if top_candidates.len() > self.cfg.m {
      return Err(Error::InvalidIndexFormat(
        "heuristic returned more than m candidates".to_string(),
      ));
    }

    let mut selected: Vec<NodeId> = Vec::with_capacity(self.cfg.m);
    while let Some((_dist, id)) = top_candidates.pop() {
      selected.push(id);
    }

    let next_entry = *selected
      .last()
      .ok_or_else(|| Error::InvalidIndexFormat("empty selected neighbor list".to_string()))?;

    for &neighbor in &selected {
      let neighbor_level = self.node_level[neighbor.as_usize()].load(Ordering::Acquire);
      if level > neighbor_level.max(0) as usize {
        return Err(Error::InvalidIndexFormat(
          "trying to link on a non-existent level".to_string(),
        ));
      }
    }

    {
      let _lock = self.link_locks[node.as_usize()].lock();
      let block = if level == 0 {
        self.level0_block(node)
      } else {
        self.upper_block(node, level)?
      };
      let header = block[0].load(Ordering::Acquire);
      if neighbors_count(header) != 0 && !is_update {
        return Err(Error::InvalidIndexFormat(
          "newly inserted node should have blank neighbor list".to_string(),
        ));
      }
      for (idx, &neighbor) in selected.iter().enumerate() {
        block[1 + idx].store(neighbor.as_u32(), Ordering::Relaxed);
      }
      Self::set_neighbors_count(&block[0], selected.len())?;
    }

    self.connect_backlinks(vectors, node, &selected, level, is_update)?;

    if selected.len() > cap {
      return Err(Error::InvalidIndexFormat("too many selected neighbors".to_string()));
    }

    Ok(next_entry)
  }

  fn connect_backlinks<V: VectorStore>(
    &self,
    vectors: &V,
    node: NodeId,
    selected: &[NodeId],
    level: usize,
    is_update: bool,
  ) -> Result<()> {
    let mcurmax = if level > 0 { self.max_m } else { self.max_m0 };

    for &neighbor in selected {
      if neighbor == node {
        return Err(Error::InvalidIndexFormat(
          "trying to connect a node to itself".to_string(),
        ));
      }
      let neighbor_level = self.node_level[neighbor.as_usize()].load(Ordering::Acquire);
      if level > neighbor_level.max(0) as usize {
        return Err(Error::InvalidIndexFormat(
          "trying to link on a non-existent level".to_string(),
        ));
      }

      let _lock = self.link_locks[neighbor.as_usize()].lock();
      let existing = self.neighbors_at_level(neighbor, level)?;
      let sz_other = existing.len;
      let is_present = is_update && existing.into_iter().any(|id| id == node);
      if sz_other > mcurmax {
        return Err(Error::InvalidIndexFormat(
          "bad neighbor list size".to_string(),
        ));
      }
      if is_present {
        continue;
      }

      let block = if level == 0 {
        self.level0_block(neighbor)
      } else {
        self.upper_block(neighbor, level)?
      };

      if sz_other < mcurmax {
        block[1 + sz_other].store(node.as_u32(), Ordering::Relaxed);
        Self::set_neighbors_count(&block[0], sz_other + 1)?;
        continue;
      }

      let mut candidates: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();
      let d_max = self.distance_between_nodes(vectors, node, neighbor)?;
      candidates.push((OrderedFloat(d_max), node));

      for existing in existing.into_iter() {
        let dist = self.distance_between_nodes(vectors, existing, neighbor)?;
        candidates.push((OrderedFloat(dist), existing));
      }

      self.get_neighbors_by_heuristic2(vectors, &mut candidates, mcurmax)?;

      let mut new_neighbors: Vec<NodeId> = Vec::with_capacity(candidates.len());
      while let Some((_dist, id)) = candidates.pop() {
        new_neighbors.push(id);
      }

      for (idx, &id) in new_neighbors.iter().enumerate() {
        block[1 + idx].store(id.as_u32(), Ordering::Relaxed);
      }
      Self::set_neighbors_count(&block[0], new_neighbors.len())?;
    }

    Ok(())
  }

  fn update_point<V: VectorStore>(&self, vectors: &V, node: NodeId, update_neighbor_probability: f32) -> Result<()> {
    let max_level_copy = self.max_level.load(Ordering::Acquire);
    let entry = self.entry_point_node().ok_or(Error::EmptyIndex)?;

    if entry == node && self.len() == 1 {
      return Ok(());
    }

    let elem_level = self.node_level[node.as_usize()].load(Ordering::Acquire);
    if elem_level < 0 {
      return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
    }

    for layer in 0..=elem_level as usize {
      let mut s_cand = HashSet::<NodeId>::new();
      let mut s_neigh = HashSet::<NodeId>::new();

      let list_one_hop = self.get_connections_with_lock(node, layer)?;
      if list_one_hop.is_empty() {
        continue;
      }

      s_cand.insert(node);

      let update_decisions: Vec<f32> = {
        let mut rng = self.update_rng.lock();
        (0..list_one_hop.len()).map(|_| rng.gen::<f32>()).collect()
      };

      for (el_one_hop, decision) in list_one_hop.into_iter().zip(update_decisions) {
        s_cand.insert(el_one_hop);

        if decision > update_neighbor_probability {
          continue;
        }

        s_neigh.insert(el_one_hop);
        let list_two_hop = self.get_connections_with_lock(el_one_hop, layer)?;
        for el_two_hop in list_two_hop {
          s_cand.insert(el_two_hop);
        }
      }

      for neigh in s_neigh {
        let size = if s_cand.contains(&neigh) {
          s_cand.len().saturating_sub(1)
        } else {
          s_cand.len()
        };
        if size == 0 {
          continue;
        }

        let elements_to_keep = self.ef_construction.min(size);
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();

        for cand in s_cand.iter().copied() {
          if cand == neigh {
            continue;
          }
          let dist = self.distance_between_nodes(vectors, neigh, cand)?;
          if candidates.len() < elements_to_keep {
            candidates.push((OrderedFloat(dist), cand));
          } else if dist < candidates.peek().unwrap().0 .0 {
            candidates.pop();
            candidates.push((OrderedFloat(dist), cand));
          }
        }

        let cap = if layer == 0 { self.max_m0 } else { self.max_m };
        self.get_neighbors_by_heuristic2(vectors, &mut candidates, cap)?;

        let _lock = self.link_locks[neigh.as_usize()].lock();
        let block = if layer == 0 {
          self.level0_block(neigh)
        } else {
          self.upper_block(neigh, layer)?
        };

        let cand_size = candidates.len();
        for idx in 0..cand_size {
          block[1 + idx].store(candidates.pop().unwrap().1.as_u32(), Ordering::Relaxed);
        }
        Self::set_neighbors_count(&block[0], cand_size)?;
      }
    }

    self.repair_connections_for_update(vectors, entry, node, elem_level as usize, max_level_copy.max(0) as usize)?;

    Ok(())
  }

  fn repair_connections_for_update<V: VectorStore>(
    &self,
    vectors: &V,
    entry: NodeId,
    node: NodeId,
    node_level: usize,
    max_level: usize,
  ) -> Result<()> {
    let mut curr = entry;
    if node_level < max_level {
      let node_vec = vectors.vector(node).ok_or(Error::MissingVector)?;
      let node_vec = node_vec.as_f32_slice();
      let mut curdist = self.distance_query_to_node(vectors, node_vec, curr)?;
      for level in (node_level + 1..=max_level).rev() {
        let mut changed = true;
        while changed {
          changed = false;
          for cand in self.neighbors_at_level(curr, level)? {
            let d = self.distance_between_nodes(vectors, node, cand)?;
            if d < curdist {
              curdist = d;
              curr = cand;
              changed = true;
            }
          }
        }
      }
    }

    if node_level > max_level {
      return Err(Error::InvalidIndexFormat(
        "node level cannot exceed max level".to_string(),
      ));
    }

    let node_vec = vectors.vector(node).ok_or(Error::MissingVector)?;
    let node_vec = node_vec.as_f32_slice();

    for level in (0..=node_level).rev() {
      let mut top_candidates = self.search_base_layer(vectors, curr, node_vec, level, self.ef_construction, None)?;

      let mut filtered: BinaryHeap<(OrderedFloat<f32>, NodeId)> = BinaryHeap::new();
      while let Some(cand) = top_candidates.pop() {
        if cand.1 != node {
          filtered.push(cand);
        }
      }

      if !filtered.is_empty() {
        let entry_deleted = self.is_deleted(entry);
        if entry_deleted {
          let dist = self.distance_between_nodes(vectors, node, entry)?;
          filtered.push((OrderedFloat(dist), entry));
          if filtered.len() > self.ef_construction {
            filtered.pop();
          }
        }
        curr = self.mutually_connect_new_element(vectors, node, filtered, level, true)?;
      }
    }

    Ok(())
  }

  fn get_connections_with_lock(&self, node: NodeId, level: usize) -> Result<Vec<NodeId>> {
    let _lock = self.link_locks[node.as_usize()].lock();
    Ok(self.neighbors_at_level(node, level)?.into_iter().collect())
  }

  fn add_point_at_level<V: VectorStore>(
    &self,
    vectors: &V,
    node: NodeId,
    forced_level: Option<i32>,
  ) -> Result<()> {
    let cur_level = forced_level.unwrap_or_else(|| self.get_random_level());
    if cur_level < 0 {
      return Err(Error::InvalidIndexFormat("level must be >= 0".to_string()));
    }
    self.node_level[node.as_usize()].store(cur_level, Ordering::Release);

    // Clear base layer neighbor list.
    for word in self.level0_block(node) {
      word.store(0, Ordering::Relaxed);
    }

    // Initialize upper layers if any.
    if cur_level > 0 {
      let words = (cur_level as usize) * (1 + self.max_m);
      let mut raw = Vec::with_capacity(words);
      raw.resize_with(words, || AtomicU32::new(0));
      self
        .upper_links
        .get(node.as_usize())
        .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
        .set(raw.into_boxed_slice())
        .map_err(|_| Error::InvalidIndexFormat("upper linklist already initialized".to_string()))?;
    }

    let mut templock = Some(self.global.lock());
    let max_level_copy = self.max_level.load(Ordering::Acquire);
    if cur_level <= max_level_copy {
      drop(templock.take());
    }

    let mut curr_obj = self.entry_point_node();
    let Some(entry) = curr_obj else {
      self.entry_point.store(node.as_u32(), Ordering::Release);
      self.max_level.store(cur_level, Ordering::Release);
      return Ok(());
    };

    let node_vec = vectors.vector(node).ok_or(Error::MissingVector)?;
    let node_vec = node_vec.as_f32_slice();

    if max_level_copy >= 0 && cur_level < max_level_copy {
      let mut curdist = self.distance_query_to_node(vectors, node_vec, entry)?;
      let mut curr = entry;
      for level in ((cur_level + 1) as usize..=max_level_copy as usize).rev() {
        let mut changed = true;
        while changed {
          changed = false;
          for cand in self.neighbors_at_level(curr, level)? {
            let d = self.distance_query_to_node(vectors, node_vec, cand)?;
            if d < curdist {
              curdist = d;
              curr = cand;
              changed = true;
            }
          }
        }
      }
      curr_obj = Some(curr);
    }

    let mut curr_obj = curr_obj.expect("entry checked above");

    let entry_deleted = self.is_deleted(entry);
    let max_conn_level = usize::min(cur_level.max(0) as usize, max_level_copy.max(0) as usize);

    let mut selected_per_level: Vec<Vec<NodeId>> = vec![Vec::new(); max_conn_level + 1];

    // Phase 1: fill `node`'s own neighbor lists, but do NOT publish backlinks yet.
    for level in (0..=max_conn_level).rev() {
      let mut top_candidates = self.search_base_layer(vectors, curr_obj, node_vec, level, self.ef_construction, None)?;
      if entry_deleted {
        let dist = self.distance_query_to_node(vectors, node_vec, entry)?;
        top_candidates.push((OrderedFloat(dist), entry));
        if top_candidates.len() > self.ef_construction {
          top_candidates.pop();
        }
      }

      self.get_neighbors_by_heuristic2(vectors, &mut top_candidates, self.cfg.m)?;
      if top_candidates.len() > self.cfg.m {
        return Err(Error::InvalidIndexFormat(
          "heuristic returned more than m candidates".to_string(),
        ));
      }

      let mut selected: Vec<NodeId> = Vec::with_capacity(self.cfg.m);
      while let Some((_dist, id)) = top_candidates.pop() {
        selected.push(id);
      }

      let next_entry = *selected
        .last()
        .ok_or_else(|| Error::InvalidIndexFormat("empty selected neighbor list".to_string()))?;

      for &neighbor in &selected {
        let neighbor_level = self.node_level[neighbor.as_usize()].load(Ordering::Acquire);
        if level > neighbor_level.max(0) as usize {
          return Err(Error::InvalidIndexFormat(
            "trying to link on a non-existent level".to_string(),
          ));
        }
      }

      {
        let _lock = self.link_locks[node.as_usize()].lock();
        let block = if level == 0 {
          self.level0_block(node)
        } else {
          self.upper_block(node, level)?
        };
        let header = block[0].load(Ordering::Acquire);
        if neighbors_count(header) != 0 {
          return Err(Error::InvalidIndexFormat(
            "new node should have blank neighbor list".to_string(),
          ));
        }
        for (idx, &neighbor) in selected.iter().enumerate() {
          block[1 + idx].store(neighbor.as_u32(), Ordering::Relaxed);
        }
        Self::set_neighbors_count(&block[0], selected.len())?;
      }

      selected_per_level[level] = selected;
      curr_obj = next_entry;
    }

    // Phase 2: publish backlinks.
    for level in (0..=max_conn_level).rev() {
      self.connect_backlinks(vectors, node, &selected_per_level[level], level, false)?;
    }

    if cur_level > max_level_copy {
      debug_assert!(templock.is_some());
      self.entry_point.store(node.as_u32(), Ordering::Release);
      self.max_level.store(cur_level, Ordering::Release);
    }

    Ok(())
  }

  pub(crate) fn legacy_start_loading(&self, cur_element_count: usize) -> Result<()> {
    if cur_element_count > self.cfg.max_nodes {
      return Err(Error::InvalidIndexFormat(
        "cur_element_count > max_nodes".to_string(),
      ));
    }
    self
      .cur_node_count
      .store(cur_element_count, Ordering::Release);
    Ok(())
  }

  pub(crate) fn legacy_set_node_key(&self, internal_id: u32, key: K) -> Result<()> {
    let node = NodeId::new(internal_id);
    self
      .node_keys
      .get(node.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
      .set(key.clone())
      .map_err(|_| Error::InvalidIndexFormat("node key already set".to_string()))?;
    if self.key_to_node.insert(key, node).is_some() {
      return Err(Error::InvalidIndexFormat("duplicate node key".to_string()));
    }
    Ok(())
  }

  pub(crate) fn legacy_set_node_level(&self, internal_id: u32, level: i32) -> Result<()> {
    if level < 0 {
      return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
    }
    let node = NodeId::new(internal_id);
    self.node_level[node.as_usize()].store(level, Ordering::Release);

    if level == 0 {
      return Ok(());
    }

    let words = (level as usize) * (1 + self.max_m);
    let mut raw = Vec::with_capacity(words);
    raw.resize_with(words, || AtomicU32::new(0));
    self
      .upper_links
      .get(node.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
      .set(raw.into_boxed_slice())
      .map_err(|_| Error::InvalidIndexFormat("upper linklist already initialized".to_string()))?;
    Ok(())
  }

  pub(crate) fn legacy_set_node_deleted(&self, internal_id: u32, deleted: bool) -> Result<()> {
    let node = NodeId::new(internal_id);
    let v = if deleted { 1 } else { 0 };
    self.node_deleted[node.as_usize()].store(v, Ordering::Release);
    Ok(())
  }

  pub(crate) fn legacy_set_neighbors(
    &self,
    internal_id: u32,
    level: usize,
    neighbors: &[u32],
  ) -> Result<()> {
    let node = NodeId::new(internal_id);
    let cap = if level == 0 { self.max_m0 } else { self.max_m };
    if neighbors.len() > cap {
      return Err(Error::InvalidIndexFormat(
        "neighbor list too large".to_string(),
      ));
    }
    for &neighbor in neighbors {
      if neighbor == internal_id {
        return Err(Error::InvalidIndexFormat(
          "self edge in neighbor list".to_string(),
        ));
      }
      if neighbor as usize >= self.len() {
        return Err(Error::InvalidIndexFormat(
          "neighbor id out of bounds".to_string(),
        ));
      }
    }

    let block = if level == 0 {
      self.level0_block(node)
    } else {
      self.upper_block(node, level)?
    };
    for (idx, &neighbor) in neighbors.iter().enumerate() {
      block[1 + idx].store(neighbor, Ordering::Relaxed);
    }
    Self::set_neighbors_count(&block[0], neighbors.len())?;
    Ok(())
  }

  pub(crate) fn legacy_finish_loading(
    &self,
    max_level: i32,
    entry_point_internal_id: Option<u32>,
    deleted_count: usize,
  ) -> Result<()> {
    let entry = entry_point_internal_id.unwrap_or(NODE_ID_NONE);
    if entry != NODE_ID_NONE && entry as usize >= self.len() {
      return Err(Error::InvalidIndexFormat(
        "entry point out of bounds".to_string(),
      ));
    }
    self.deleted_count.store(deleted_count, Ordering::Release);
    self.max_level.store(max_level, Ordering::Release);
    self.entry_point.store(entry, Ordering::Release);
    Ok(())
  }

  pub fn to_data(&self) -> Result<HnswData<K>> {
    let _mutation_guard = self.mutation_lock.write();

    let len = self.len();
    let max_level = self.max_level.load(Ordering::Acquire);
    let entry_point = self.entry_point_node().map(|n| n.as_u32());

    if len == 0 {
      if entry_point.is_some() || max_level != -1 {
        return Err(Error::InvalidIndexFormat(
          "empty index has non-empty entry point/maxlevel".to_string(),
        ));
      }
    } else {
      let Some(entry) = entry_point else {
        return Err(Error::InvalidIndexFormat(
          "non-empty index missing entry point".to_string(),
        ));
      };
      if entry as usize >= len {
        return Err(Error::InvalidIndexFormat(
          "entry point out of bounds".to_string(),
        ));
      }
      if max_level < 0 {
        return Err(Error::InvalidIndexFormat(
          "non-empty index has maxlevel < 0".to_string(),
        ));
      }
      let entry_level = self.node_level[entry as usize].load(Ordering::Acquire);
      if entry_level != max_level {
        return Err(Error::InvalidIndexFormat(
          "entry point level != max_level".to_string(),
        ));
      }
    }

    let mut cfg = self.cfg.clone();
    cfg.ef_search = self.ef_search.load(Ordering::Acquire);
    cfg.m = self.max_m;
    cfg.ef_construction = self.ef_construction;

    #[derive(Debug)]
    struct Chunk<K> {
      keys: Vec<K>,
      deleted: Vec<u8>,
      levels: Vec<u32>,
      neighbor_counts: Vec<u16>,
      neighbors: Vec<u32>,
      max_level: i32,
    }

    let threads = rayon::current_num_threads();
    let desired_chunks = threads.saturating_mul(8).max(1);
    let min_chunk = 16_384usize;
    let chunk_size = (len / desired_chunks).max(min_chunk).max(1);
    let chunk_count = len.div_ceil(chunk_size);

    let chunks: Result<Vec<Chunk<K>>> = (0..chunk_count)
      .into_par_iter()
      .map(|chunk_idx| -> Result<Chunk<K>> {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(len);
        let range_len = end - start;

        let mut keys = Vec::with_capacity(range_len);
        let mut deleted = Vec::with_capacity(range_len);
        let mut levels = Vec::with_capacity(range_len);

        let mut neighbor_counts = Vec::new();
        let mut neighbors = Vec::new();

        let mut max_level = -1i32;

        for internal_id in start..end {
          let node = NodeId::new(internal_id as u32);
          keys.push(self.node_key(node)?.clone());

          let is_deleted = self.is_deleted(node);
          deleted.push(if is_deleted { 1 } else { 0 });

          let level_i32 = self.node_level[internal_id].load(Ordering::Acquire);
          if level_i32 < 0 {
            return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
          }
          max_level = max(max_level, level_i32);
          let level_u32 = level_i32 as u32;
          levels.push(level_u32);

          for l in 0..=level_u32 as usize {
            let list = self.neighbors_at_level(node, l)?;
            let cnt: u16 = list
              .len
              .try_into()
              .map_err(|_| Error::InvalidIndexFormat("neighbor list too large".to_string()))?;
            neighbor_counts.push(cnt);
            for neighbor in list {
              neighbors.push(neighbor.as_u32());
            }
          }
        }

        Ok(Chunk {
          keys,
          deleted,
          levels,
          neighbor_counts,
          neighbors,
          max_level,
        })
      })
      .collect();
    let chunks = chunks?;

    let total_neighbor_counts: usize = chunks.iter().map(|c| c.neighbor_counts.len()).sum();
    let total_neighbors: usize = chunks.iter().map(|c| c.neighbors.len()).sum();

    let mut keys = Vec::with_capacity(len);
    let mut deleted = Vec::with_capacity(len);
    let mut levels = Vec::with_capacity(len);
    let mut neighbor_counts = Vec::with_capacity(total_neighbor_counts);
    let mut neighbors = Vec::with_capacity(total_neighbors);

    let mut computed_max_level = -1i32;
    for chunk in chunks {
      computed_max_level = max(computed_max_level, chunk.max_level);
      keys.extend(chunk.keys);
      deleted.extend(chunk.deleted);
      levels.extend(chunk.levels);
      neighbor_counts.extend(chunk.neighbor_counts);
      neighbors.extend(chunk.neighbors);
    }

    if computed_max_level != max_level {
      return Err(Error::InvalidIndexFormat(
        "max_level does not match node levels".to_string(),
      ));
    }

    if keys.len() != len || deleted.len() != len || levels.len() != len {
      return Err(Error::InvalidIndexFormat(
        "serialized node arrays have mismatched lengths".to_string(),
      ));
    }

    let expected_neighbor_count_entries: usize = levels
      .iter()
      .try_fold(0usize, |acc, &level| acc.checked_add(level as usize + 1))
      .ok_or_else(|| Error::InvalidIndexFormat("neighbor counts overflow".to_string()))?;
    if neighbor_counts.len() != expected_neighbor_count_entries {
      return Err(Error::InvalidIndexFormat(
        "neighbor list counts length mismatch".to_string(),
      ));
    }

    let expected_neighbors_len: usize = neighbor_counts
      .iter()
      .try_fold(0usize, |acc, &cnt| acc.checked_add(cnt as usize))
      .ok_or_else(|| Error::InvalidIndexFormat("neighbors overflow".to_string()))?;
    if neighbors.len() != expected_neighbors_len {
      return Err(Error::InvalidIndexFormat(
        "neighbor ids length mismatch".to_string(),
      ));
    }

    Ok(HnswData {
      version: HNSW_DATA_VERSION,
      cfg,
      entry_point,
      max_level,
      keys,
      deleted,
      levels,
      neighbor_counts,
      neighbors,
    })
  }

  pub fn from_data(metric: M, data: HnswData<K>) -> Result<Self> {
    if data.version != HNSW_DATA_VERSION {
      return Err(Error::InvalidIndexFormat(format!(
        "unsupported hnsw data version {}",
        data.version
      )));
    }
    if data.cfg.dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if data.cfg.m < 2 {
      return Err(Error::InvalidIndexFormat("m must be >= 2".to_string()));
    }
    if data.keys.len() > data.cfg.max_nodes {
      return Err(Error::InvalidIndexFormat(
        "node count > max_nodes".to_string(),
      ));
    }
    if data.cfg.max_nodes > u32::MAX as usize {
      return Err(Error::InvalidIndexFormat(
        "max_nodes exceeds u32::MAX".to_string(),
      ));
    }

    let node_count = data.keys.len();
    if data.deleted.len() != node_count || data.levels.len() != node_count {
      return Err(Error::InvalidIndexFormat(
        "serialized node arrays have mismatched lengths".to_string(),
      ));
    }
    for &d in &data.deleted {
      if d > 1 {
        return Err(Error::InvalidIndexFormat(
          "deleted flag is not 0/1".to_string(),
        ));
      }
    }

    if node_count == 0 {
      if data.entry_point.is_some() || data.max_level != -1 {
        return Err(Error::InvalidIndexFormat(
          "empty index has non-empty entry point/maxlevel".to_string(),
        ));
      }
      if !data.neighbor_counts.is_empty() || !data.neighbors.is_empty() {
        return Err(Error::InvalidIndexFormat(
          "empty index has non-empty neighbor data".to_string(),
        ));
      }
    } else {
      let Some(ep) = data.entry_point else {
        return Err(Error::InvalidIndexFormat(
          "non-empty index missing entry point".to_string(),
        ));
      };
      if ep as usize >= node_count {
        return Err(Error::InvalidIndexFormat(
          "entry point out of bounds".to_string(),
        ));
      }
      if data.max_level < 0 {
        return Err(Error::InvalidIndexFormat(
          "non-empty index has maxlevel < 0".to_string(),
        ));
      }
    }

    let mut max_node_level: i32 = -1;
    for &level in &data.levels {
      if level > i32::MAX as u32 {
        return Err(Error::InvalidIndexFormat(
          "node level too large".to_string(),
        ));
      }
      max_node_level = max(max_node_level, level as i32);
    }

    if node_count > 0 && data.max_level != max_node_level {
      return Err(Error::InvalidIndexFormat(
        "max_level does not match node levels".to_string(),
      ));
    }
    if let Some(ep) = data.entry_point {
      if data.levels[ep as usize] as i32 != data.max_level {
        return Err(Error::InvalidIndexFormat(
          "entry point level != max_level".to_string(),
        ));
      }
    }

    let m_capped = data.cfg.m.min(10_000);
    let max_m = m_capped;
    let max_m0 = m_capped
      .checked_mul(2)
      .ok_or_else(|| Error::InvalidIndexFormat("max_m0 overflow".to_string()))?;

    let expected_neighbor_count_entries: usize = data
      .levels
      .iter()
      .try_fold(0usize, |acc, &level| acc.checked_add(level as usize + 1))
      .ok_or_else(|| Error::InvalidIndexFormat("neighbor counts overflow".to_string()))?;
    if data.neighbor_counts.len() != expected_neighbor_count_entries {
      return Err(Error::InvalidIndexFormat(
        "neighbor list counts length mismatch".to_string(),
      ));
    }

    let expected_neighbors_len: usize = data
      .neighbor_counts
      .iter()
      .try_fold(0usize, |acc, &cnt| acc.checked_add(cnt as usize))
      .ok_or_else(|| Error::InvalidIndexFormat("neighbors overflow".to_string()))?;
    if data.neighbors.len() != expected_neighbors_len {
      return Err(Error::InvalidIndexFormat(
        "neighbor ids length mismatch".to_string(),
      ));
    }

    let mut list_idx = 0usize;
    let mut neighbor_idx = 0usize;
    let mut seen = HashSet::<u32>::new();

    for (internal_id, &level_u32) in data.levels.iter().enumerate() {
      for level in 0..=level_u32 as usize {
        let cnt = data
          .neighbor_counts
          .get(list_idx)
          .copied()
          .ok_or_else(|| Error::InvalidIndexFormat("neighbor counts out of bounds".to_string()))?
          as usize;
        list_idx += 1;

        let cap = if level == 0 { max_m0 } else { max_m };
        if cnt > cap {
          return Err(Error::InvalidIndexFormat(
            "neighbor list too large".to_string(),
          ));
        }

        let slice = data
          .neighbors
          .get(neighbor_idx..neighbor_idx + cnt)
          .ok_or_else(|| Error::InvalidIndexFormat("neighbor ids out of bounds".to_string()))?;
        neighbor_idx += cnt;

        seen.clear();
        for &neighbor in slice {
          if neighbor == internal_id as u32 {
            return Err(Error::InvalidIndexFormat(
              "self edge in neighbor list".to_string(),
            ));
          }
          if neighbor as usize >= node_count {
            return Err(Error::InvalidIndexFormat(
              "neighbor id out of bounds".to_string(),
            ));
          }
          if data.levels[neighbor as usize] < level as u32 {
            return Err(Error::InvalidIndexFormat(
              "neighbor does not exist at level".to_string(),
            ));
          }
          if !seen.insert(neighbor) {
            return Err(Error::InvalidIndexFormat(
              "duplicate neighbor in list".to_string(),
            ));
          }
        }
      }
    }
    debug_assert_eq!(list_idx, data.neighbor_counts.len());
    debug_assert_eq!(neighbor_idx, data.neighbors.len());

    let h = Self::new(metric, data.cfg);
    h.legacy_start_loading(node_count)?;

    let mut deleted_count = 0usize;
    for internal_id in 0..node_count {
      let deleted = data.deleted[internal_id] != 0;
      if deleted {
        deleted_count += 1;
      }
      h.legacy_set_node_key(internal_id as u32, data.keys[internal_id].clone())?;
      h.legacy_set_node_level(internal_id as u32, data.levels[internal_id] as i32)?;
      h.legacy_set_node_deleted(internal_id as u32, deleted)?;
    }

    let mut list_idx = 0usize;
    let mut neighbor_idx = 0usize;
    for internal_id in 0..node_count {
      for level in 0..=data.levels[internal_id] as usize {
        let cnt = data.neighbor_counts[list_idx] as usize;
        list_idx += 1;
        let slice = &data.neighbors[neighbor_idx..neighbor_idx + cnt];
        neighbor_idx += cnt;
        h.legacy_set_neighbors(internal_id as u32, level, slice)?;
      }
    }
    debug_assert_eq!(list_idx, data.neighbor_counts.len());
    debug_assert_eq!(neighbor_idx, data.neighbors.len());

    h.legacy_finish_loading(data.max_level, data.entry_point, deleted_count)?;
    Ok(h)
  }

  pub fn insert<V: VectorStoreMut>(&self, vectors: &V, key: K, vector: &[f32]) -> Result<()> {
    if vector.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: vector.len(),
      });
    }

    let _mutation_guard = self.mutation_lock.read();
    let _key_guard = self.key_lock(&key).lock();

    if self.key_to_node.contains_key(&key) {
      return Err(Error::KeyAlreadyExists);
    }

    let node = self.alloc_node()?;
    // Store the vector before publishing the node into the graph.
    vectors.set(node, vector)?;
    self
      .node_keys
      .get(node.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
      .set(key.clone())
      .map_err(|_| Error::InvalidIndexFormat("node key already set".to_string()))?;
    self.node_deleted[node.as_usize()].store(0, Ordering::Release);
    self.key_to_node.insert(key, node);

    self.add_point_at_level(vectors, node, None)?;
    Ok(())
  }

  pub fn set<V: VectorStoreMut>(&self, vectors: &V, key: K, vector: &[f32]) -> Result<SetOutcome> {
    if vector.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: vector.len(),
      });
    }

    let _mutation_guard = self.mutation_lock.read();
    let _key_guard = self.key_lock(&key).lock();

    if let Some(existing) = self.key_to_node.get(&key).map(|e| *e.value()) {
      vectors.set(existing, vector)?;
      let resurrected = self.unmark_deleted(existing)?;
      self.update_point(vectors, existing, 1.0)?;
      return Ok(if resurrected {
        SetOutcome::Resurrected
      } else {
        SetOutcome::Updated
      });
    }

    let node = self.alloc_node()?;
    vectors.set(node, vector)?;
    self
      .node_keys
      .get(node.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node out of bounds".to_string()))?
      .set(key.clone())
      .map_err(|_| Error::InvalidIndexFormat("node key already set".to_string()))?;
    self.node_deleted[node.as_usize()].store(0, Ordering::Release);
    self.key_to_node.insert(key, node);
    self.add_point_at_level(vectors, node, None)?;

    Ok(SetOutcome::Inserted)
  }

  pub fn delete(&self, key: &K) -> Result<bool> {
    let _mutation_guard = self.mutation_lock.read();
    let _key_guard = self.key_lock(key).lock();
    let Some(node) = self.key_to_node.get(key).map(|e| *e.value()) else {
      return Ok(false);
    };
    if self.is_deleted(node) {
      return Ok(false);
    }
    self.mark_deleted(node)?;
    Ok(true)
  }

  pub fn search<V: VectorStore>(
    &self,
    vectors: &V,
    query: &[f32],
    k: usize,
    filter: Option<&dyn Fn(&K) -> bool>,
  ) -> Result<Vec<SearchHit<K>>> {
    if query.len() != self.cfg.dim {
      return Err(Error::DimensionMismatch {
        expected: self.cfg.dim,
        actual: query.len(),
      });
    }

    let _mutation_guard = self.mutation_lock.read();

    let entry = self.entry_point_node().ok_or(Error::EmptyIndex)?;
    let max_level = self.max_level.load(Ordering::Acquire);

    let mut curr = entry;
    let mut curdist = self.distance_query_to_node(vectors, query, curr)?;

    for level in (1..=max_level.max(0) as usize).rev() {
      let mut changed = true;
      while changed {
        changed = false;
        for cand in self.neighbors_at_level(curr, level)? {
          let d = self.distance_query_to_node(vectors, query, cand)?;
          if d < curdist {
            curdist = d;
            curr = cand;
            changed = true;
          }
        }
      }
    }

    let ef = max(k + 1, self.ef_search.load(Ordering::Acquire));
    let top_candidates = self.search_base_layer(vectors, curr, query, 0, ef, filter)?;
    let mut out = Vec::with_capacity(k);
    for (dist, node) in top_candidates.into_sorted_vec() {
      if out.len() >= k {
        break;
      }
      if self.is_deleted(node) {
        continue;
      }
      let key = self.node_key(node)?;
      if filter.map(|f| f(key)).unwrap_or(true) {
        out.push(SearchHit {
          key: key.clone(),
          distance: dist.0,
        });
      }
    }
    Ok(out)
  }

  pub fn entry_key(&self) -> Result<K> {
    let entry = self.entry_point_node().ok_or(Error::EmptyIndex)?;
    Ok(self.node_key(entry)?.clone())
  }

  pub fn keys(&self) -> Vec<K> {
    let len = self.len();
    let mut out = Vec::with_capacity(len);
    for internal_id in 0..len {
      if let Some(key) = self.node_keys[internal_id].get() {
        out.push(key.clone());
      }
    }
    out
  }

  pub fn node_id(&self, key: &K) -> Result<NodeId> {
    self
      .key_to_node
      .get(key)
      .map(|e| *e.value())
      .ok_or(Error::KeyNotFound)
  }

  pub fn is_deleted_key(&self, key: &K) -> Result<bool> {
    let node = self
      .key_to_node
      .get(key)
      .map(|e| *e.value())
      .ok_or(Error::KeyNotFound)?;
    Ok(self.is_deleted(node))
  }

  pub fn level(&self, key: &K) -> Result<usize> {
    let node = self
      .key_to_node
      .get(key)
      .map(|e| *e.value())
      .ok_or(Error::KeyNotFound)?;
    let level = self.node_level[node.as_usize()].load(Ordering::Acquire);
    if level < 0 {
      return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
    }
    Ok(level as usize)
  }

  pub fn neighbors(&self, key: &K, level: usize) -> Result<Vec<K>> {
    let node = self
      .key_to_node
      .get(key)
      .map(|e| *e.value())
      .ok_or(Error::KeyNotFound)?;
    let max_level = self.node_level[node.as_usize()].load(Ordering::Acquire);
    if max_level < 0 {
      return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
    }
    if level > max_level as usize {
      return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for neighbor in self.neighbors_at_level(node, level)? {
      out.push(self.node_key(neighbor)?.clone());
    }
    Ok(out)
  }

  pub fn merged_neighbors(&self, key: &K, min_level: usize) -> Result<Vec<K>> {
    let node = self
      .key_to_node
      .get(key)
      .map(|e| *e.value())
      .ok_or(Error::KeyNotFound)?;
    let max_level = self.node_level[node.as_usize()].load(Ordering::Acquire);
    if max_level < 0 {
      return Err(Error::InvalidIndexFormat("node level < 0".to_string()));
    }
    let max_level = max_level as usize;
    if min_level > max_level {
      return Ok(Vec::new());
    }

    let mut seen = HashSet::<u32>::new();
    for level in min_level..=max_level {
      for neighbor in self.neighbors_at_level(node, level)? {
        seen.insert(neighbor.as_u32());
      }
    }

    let mut ids: Vec<u32> = seen.into_iter().collect();
    ids.sort_unstable();

    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
      out.push(self.node_key(NodeId::new(id))?.clone());
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::metric::L2;
  use crate::vectors::InMemoryVectorStore;
  use crate::vectors::VectorStore;
  use rand::rngs::StdRng;
  use rand::Rng;
  use rand::SeedableRng;
  use proptest::prelude::*;
  use std::sync::Arc;
  use std::thread;

  fn assert_integrity<K: Eq + Hash + Clone + Send + Sync + 'static, M: Metric>(h: &Hnsw<K, M>) {
    let len = h.len();
    assert!(len <= h.max_nodes());

    let mut deleted = 0usize;
    for internal_id in 0..len {
      let node = NodeId::new(internal_id as u32);
      let key = h
        .node_keys
        .get(internal_id)
        .and_then(|k| k.get())
        .expect("missing node key");
      let mapped = h
        .key_to_node
        .get(key)
        .map(|e| *e.value())
        .expect("missing key_to_node entry");
      assert_eq!(mapped, node);

      if h.node_deleted[internal_id].load(Ordering::Acquire) != 0 {
        deleted += 1;
      }

      let level = h.node_level[internal_id].load(Ordering::Acquire);
      assert!(level >= 0);

      if level == 0 {
        assert!(h.upper_links[internal_id].get().is_none());
      } else {
        assert!(h.upper_links[internal_id].get().is_some());
      }

      for l in 0..=level as usize {
        let cap = if l == 0 { h.max_m0 } else { h.max_m };
        let list = h.neighbors_at_level(node, l).expect("neighbors_at_level");
        assert!(list.len <= cap);
        for neighbor in list {
          assert_ne!(neighbor, node);
          assert!(neighbor.as_usize() < len);
          let neighbor_level = h.node_level[neighbor.as_usize()].load(Ordering::Acquire);
          assert!(neighbor_level >= l as i32);
        }
      }
    }
    assert_eq!(h.deleted_len(), deleted);
  }

  fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
  }

  #[test]
  fn serde_bincode_roundtrip_preserves_graph() {
    let dim = 8;
    let cfg = HnswConfig::new(dim, 1024)
      .m(16)
      .ef_construction(128)
      .ef_search(64)
      .seed(123);
    let store = InMemoryVectorStore::new(dim, cfg.max_nodes);
    let h = Hnsw::new(L2::new(), cfg);

    let mut rng = StdRng::seed_from_u64(1);
    for key in 0u64..200 {
      let v = random_vec(&mut rng, dim);
      h.insert(&store, key, &v).unwrap();
    }

    for key in 0u64..50 {
      assert!(h.delete(&key).unwrap());
    }

    for key in 25u64..75 {
      let v = random_vec(&mut rng, dim);
      let _ = h.set(&store, key, &v).unwrap();
    }

    h.set_ef_search(123);

    assert_integrity(&h);

    let data = h.to_data().unwrap();
    let bytes = bincode::serialize(&data).unwrap();
    let decoded: HnswData<u64> = bincode::deserialize(&bytes).unwrap();
    let h2 = Hnsw::from_data(L2::new(), decoded).unwrap();

    assert_integrity(&h2);
    assert_eq!(data, h2.to_data().unwrap());

    let q = random_vec(&mut rng, dim);
    let hits1 = h.search(&store, &q, 10, None).unwrap();
    let hits2 = h2.search(&store, &q, 10, None).unwrap();
    assert_eq!(hits1, hits2);
  }

  #[test]
  fn from_data_rejects_unknown_version() {
    let cfg = HnswConfig::new(4, 16).m(8);
    let h: Hnsw<u64, _> = Hnsw::new(L2::new(), cfg);
    let mut data = h.to_data().unwrap();
    data.version += 1;
    let err = match Hnsw::from_data(L2::new(), data) {
      Ok(_) => panic!("expected version mismatch error"),
      Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidIndexFormat(_)));
  }

  #[test]
  fn insert_delete_resurrect() {
    let dim = 8;
    let cfg = HnswConfig::new(dim, 128)
      .m(8)
      .ef_construction(64)
      .ef_search(64)
      .seed(123);
    let store = InMemoryVectorStore::new(dim, cfg.max_nodes);
    let h = Hnsw::new(L2::new(), cfg);

    let mut rng = StdRng::seed_from_u64(1);
    let a = random_vec(&mut rng, dim);
    let b = random_vec(&mut rng, dim);
    let c = random_vec(&mut rng, dim);

    h.insert(&store, 1, &a).unwrap();
    h.insert(&store, 2, &b).unwrap();
    h.insert(&store, 3, &c).unwrap();
    assert_integrity(&h);

    assert!(h.delete(&2).unwrap());
    assert!(h.is_deleted_key(&2).unwrap());
    assert_integrity(&h);

    let hits = h.search(&store, &b, 10, None).unwrap();
    assert!(hits.iter().all(|hit| hit.key != 2));

    let b2 = random_vec(&mut rng, dim);
    assert_eq!(h.set(&store, 2, &b2).unwrap(), SetOutcome::Resurrected);
    assert!(!h.is_deleted_key(&2).unwrap());
    assert_integrity(&h);

    let hits = h.search(&store, &b2, 10, None).unwrap();
    assert!(hits.iter().any(|hit| hit.key == 2));
  }

  #[test]
  fn set_updates_existing_key() {
    let dim = 4;
    let cfg = HnswConfig::new(dim, 64)
      .m(8)
      .ef_construction(64)
      .ef_search(64)
      .seed(42);
    let store = InMemoryVectorStore::new(dim, cfg.max_nodes);
    let h = Hnsw::new(L2::new(), cfg);

    let v1 = vec![0.0, 0.0, 0.0, 0.0];
    let v2 = vec![1.0, 1.0, 1.0, 1.0];

    h.insert(&store, 7, &v1).unwrap();
    assert_eq!(h.set(&store, 7, &v2).unwrap(), SetOutcome::Updated);
    assert!(!h.is_deleted_key(&7).unwrap());

    let id = h.node_id(&7).unwrap();
    let stored = store.vector(id).unwrap();
    assert_eq!(stored.as_f32_slice(), v2.as_slice());
    assert_integrity(&h);
  }

  #[test]
  fn search_respects_filter() {
    let dim = 6;
    let cfg = HnswConfig::new(dim, 256)
      .m(16)
      .ef_construction(128)
      .ef_search(128)
      .seed(999);
    let store = InMemoryVectorStore::new(dim, cfg.max_nodes);
    let h = Hnsw::new(L2::new(), cfg);

    let mut rng = StdRng::seed_from_u64(99);
    for key in 0u64..50 {
      let v = random_vec(&mut rng, dim);
      h.insert(&store, key, &v).unwrap();
    }
    assert_integrity(&h);

    let q = random_vec(&mut rng, dim);
    let hits = h
      .search(&store, &q, 20, Some(&|k: &u64| k % 2 == 0))
      .unwrap();
    assert!(hits.iter().all(|hit| hit.key % 2 == 0));
  }

  #[test]
  fn parallel_set_delete_search_smoke() {
    let dim = 8;
    let cfg = HnswConfig::new(dim, 10_000)
      .m(16)
      .ef_construction(128)
      .ef_search(128)
      .seed(7);
    let store = Arc::new(InMemoryVectorStore::new(dim, cfg.max_nodes));
    let h = Arc::new(Hnsw::new(L2::new(), cfg));

    for key in 0u64..1000 {
      let v = vec![key as f32; dim];
      h.insert(&*store, key, &v).unwrap();
    }

    let mut threads = Vec::new();
    for t in 0..8u64 {
      let h = h.clone();
      let store = store.clone();
      threads.push(thread::spawn(move || {
        let mut rng = StdRng::seed_from_u64(1000 + t);
        for _ in 0..2000 {
          let key = rng.gen_range(0u64..2000);
          match rng.gen_range(0u8..3) {
            0 => {
              let v = random_vec(&mut rng, dim);
              let _ = h.set(&*store, key, &v);
            }
            1 => {
              let _ = h.delete(&key);
            }
            _ => {
              let q = random_vec(&mut rng, dim);
              let _ = h.search(&*store, &q, 10, None);
            }
          }
        }
      }));
    }
    for t in threads {
      t.join().unwrap();
    }

    assert_integrity(&h);
  }

  #[derive(Clone, Debug)]
  enum Op {
    Insert { key: u64, vector: Vec<f32> },
    Set { key: u64, vector: Vec<f32> },
    Delete { key: u64 },
    Search { query: Vec<f32>, k: usize },
  }

  fn op_strategy(dim: usize) -> impl Strategy<Value = Op> {
    let key = 0u64..64;
    let vector = prop::collection::vec(-1000i16..1000, dim)
      .prop_map(|v| v.into_iter().map(|x| x as f32 / 100.0).collect::<Vec<_>>());
    prop_oneof![
      (key.clone(), vector.clone())
        .prop_map(|(key, vector)| Op::Insert { key, vector }),
      (key.clone(), vector.clone()).prop_map(|(key, vector)| Op::Set { key, vector }),
      key.clone().prop_map(|key| Op::Delete { key }),
      (vector, 0usize..10).prop_map(|(query, k)| Op::Search { query, k }),
    ]
  }

  proptest! {
    #[test]
    fn proptest_random_ops(ops in prop::collection::vec(op_strategy(4), 1..100)) {
      let dim = 4;
      let cfg = HnswConfig::new(dim, 10_000)
        .m(16)
        .ef_construction(128)
        .ef_search(128)
        .seed(123);
      let store = InMemoryVectorStore::new(dim, cfg.max_nodes);
      let h = Hnsw::new(L2::new(), cfg);

      let mut model = std::collections::HashMap::<u64, bool>::new(); // key -> deleted

      for op in ops {
        match op {
          Op::Insert { key, vector } => {
            let res = h.insert(&store, key, &vector);
            if model.contains_key(&key) {
              prop_assert!(matches!(res, Err(Error::KeyAlreadyExists)));
            } else {
              prop_assert!(res.is_ok());
              model.insert(key, false);
            }
          }
          Op::Set { key, vector } => {
            let existed = model.get(&key).copied();
            let out = h.set(&store, key, &vector).unwrap();
            match existed {
              None => prop_assert_eq!(out, SetOutcome::Inserted),
              Some(true) => prop_assert_eq!(out, SetOutcome::Resurrected),
              Some(false) => prop_assert_eq!(out, SetOutcome::Updated),
            }
            model.insert(key, false);
          }
          Op::Delete { key } => {
            let deleted = h.delete(&key).unwrap();
            match model.get(&key).copied() {
              None => prop_assert!(!deleted),
              Some(true) => prop_assert!(!deleted),
              Some(false) => {
                prop_assert!(deleted);
                model.insert(key, true);
              }
            }
          }
          Op::Search { query, k } => {
            match h.search(&store, &query, k, None) {
              Ok(hits) => {
                prop_assert!(hits.len() <= k);
                for w in hits.windows(2) {
                  prop_assert!(w[0].distance <= w[1].distance + 1e-6);
                  prop_assert_ne!(w[0].key, w[1].key);
                }
                for hit in hits {
                  let Some(&is_deleted) = model.get(&hit.key) else {
                    prop_assert!(false, "search returned unknown key");
                    continue;
                  };
                  prop_assert!(!is_deleted);
                }
              }
              Err(Error::EmptyIndex) => {
                prop_assert!(model.is_empty());
              }
              Err(err) => {
                prop_assert!(false, "unexpected search error: {err:?}");
              }
            }
          }
        }

        assert_integrity(&h);
      }
    }
  }
}
