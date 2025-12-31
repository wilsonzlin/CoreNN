use crate::error::Error;
use crate::error::Result;
use crate::space::label_allowed;
use crate::visited::VisitedListPool;
use crate::LabelType;
use crate::SearchStopCondition;
use crate::Space;
use crate::TableInt;
use arc_swap::ArcSwapOption;
use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::f64;
use std::io::Read;
use std::io::Write;
use std::mem::size_of;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::warn;

const DELETE_MARK: u32 = 0x01 << 16;
const MAX_LABEL_OPERATION_LOCKS: usize = 65_536;

fn linklist_count(header: u32) -> usize {
  (header & 0xffff) as usize
}

fn set_linklist_count(header: &mut u32, count: usize) {
  let count: u32 = count.try_into().expect("count overflow");
  *header = (*header & 0xffff_0000) | (count & 0xffff);
}

#[derive(Clone, Copy)]
struct LinkList<'a> {
  data: &'a [AtomicU32],
  len: usize,
}

impl LinkList<'_> {
  fn len(&self) -> usize {
    self.len
  }
}

struct LinkListIter<'a> {
  data: &'a [AtomicU32],
  idx: usize,
  end: usize,
}

impl Iterator for LinkListIter<'_> {
  type Item = TableInt;

  fn next(&mut self) -> Option<Self::Item> {
    if self.idx >= self.end {
      return None;
    }
    let id = self.data[self.idx].load(Ordering::Relaxed);
    self.idx += 1;
    Some(id)
  }
}

impl<'a> IntoIterator for LinkList<'a> {
  type Item = TableInt;
  type IntoIter = LinkListIter<'a>;

  fn into_iter(self) -> Self::IntoIter {
    LinkListIter {
      data: self.data,
      idx: 0,
      end: self.len,
    }
  }
}

#[derive(Debug)]
pub struct HnswIndex<S: Space> {
  space: S,

  max_elements: usize,

  m: usize,
  max_m: usize,
  max_m0: usize,
  ef_construction: usize,

  mult: f64,
  rev_size: f64,

  allow_replace_deleted: bool,

  visited_list_pool: VisitedListPool,

  /// Prevents `save_to_*` / `resize_index` from racing concurrent mutations.
  mutation_lock: RwLock<()>,

  /// Locks operations with element by label value (hashed).
  label_op_locks: Vec<Mutex<()>>,

  /// Protects `enter_point_node` and `max_level` updates.
  global: Mutex<()>,

  /// Protects link list updates per internal id.
  link_list_locks: Vec<Mutex<()>>,

  label_lookup: Mutex<HashMap<LabelType, TableInt>>,

  deleted_elements: Mutex<HashSet<TableInt>>,

  cur_element_count: AtomicUsize,
  num_deleted: AtomicUsize,

  ef: AtomicUsize,
  max_level: AtomicI32,
  /// `TableInt::MAX` means empty.
  enter_point_node: AtomicU32,

  labels: Vec<AtomicUsize>,
  vectors: Vec<ArcSwapOption<Vec<f32>>>,

  // Base layer: per element, [header, neighbors...max_m0]
  level0_links: Vec<AtomicU32>,
  // Upper layers: per element, if level>0, [ [header, neighbors...max_m] * level ]
  link_lists: Vec<OnceLock<Box<[AtomicU32]>>>,
  element_levels: Vec<AtomicI32>,

  level_rng: Mutex<StdRng>,
  update_probability_rng: Mutex<StdRng>,
}

impl<S: Space> HnswIndex<S> {
  pub fn new(
    space: S,
    max_elements: usize,
    m: usize,
    ef_construction: usize,
    random_seed: u64,
    allow_replace_deleted: bool,
  ) -> Self {
    assert!(max_elements <= TableInt::MAX as usize);
    assert!(space.dim() > 0, "dim must be > 0");
    assert!(m >= 2, "M must be >= 2");

    let m = if m <= 10000 {
      m
    } else {
      warn!("M parameter exceeds 10000; capping to 10000");
      10000
    };

    let max_m = m;
    let max_m0 = m * 2;
    let ef_construction = ef_construction.max(m);

    let level0_words_per_element = 1 + max_m0;
    let level0_total_words = max_elements * level0_words_per_element;

    let mult = 1.0 / (m as f64).ln();
    let rev_size = 1.0 / mult;

    let mut label_op_locks = Vec::with_capacity(MAX_LABEL_OPERATION_LOCKS);
    for _ in 0..MAX_LABEL_OPERATION_LOCKS {
      label_op_locks.push(Mutex::new(()));
    }

    let mut link_list_locks = Vec::with_capacity(max_elements);
    for _ in 0..max_elements {
      link_list_locks.push(Mutex::new(()));
    }

    let mut labels = Vec::with_capacity(max_elements);
    labels.resize_with(max_elements, || AtomicUsize::new(0));

    let mut vectors = Vec::with_capacity(max_elements);
    vectors.resize_with(max_elements, ArcSwapOption::empty);

    let mut level0_links = Vec::with_capacity(level0_total_words);
    level0_links.resize_with(level0_total_words, || AtomicU32::new(0));

    let mut link_lists = Vec::with_capacity(max_elements);
    link_lists.resize_with(max_elements, OnceLock::new);

    let mut element_levels = Vec::with_capacity(max_elements);
    element_levels.resize_with(max_elements, || AtomicI32::new(0));

    Self {
      space,
      max_elements,
      m,
      max_m,
      max_m0,
      ef_construction,
      mult,
      rev_size,
      allow_replace_deleted,
      visited_list_pool: VisitedListPool::new(1, max_elements),
      mutation_lock: RwLock::new(()),
      label_op_locks,
      global: Mutex::new(()),
      link_list_locks,
      label_lookup: Mutex::new(HashMap::new()),
      deleted_elements: Mutex::new(HashSet::new()),
      cur_element_count: AtomicUsize::new(0),
      num_deleted: AtomicUsize::new(0),
      ef: AtomicUsize::new(10),
      max_level: AtomicI32::new(-1),
      enter_point_node: AtomicU32::new(TableInt::MAX),
      labels,
      vectors,
      level0_links,
      link_lists,
      element_levels,
      level_rng: Mutex::new(StdRng::seed_from_u64(random_seed)),
      update_probability_rng: Mutex::new(StdRng::seed_from_u64(random_seed.wrapping_add(1))),
    }
  }

  pub fn space(&self) -> &S {
    &self.space
  }

  pub fn dim(&self) -> usize {
    self.space.dim()
  }

  pub fn set_ef(&self, ef: usize) {
    self.ef.store(ef, Ordering::Release);
  }

  pub fn get_max_elements(&self) -> usize {
    self.max_elements
  }

  pub fn get_current_element_count(&self) -> usize {
    self.cur_element_count.load(Ordering::Acquire)
  }

  pub fn get_deleted_count(&self) -> usize {
    self.num_deleted.load(Ordering::Acquire)
  }

  fn enter_point_node(&self) -> Option<TableInt> {
    let raw = self.enter_point_node.load(Ordering::Acquire);
    if raw == TableInt::MAX {
      None
    } else {
      Some(raw)
    }
  }

  fn label_op_lock(&self, label: LabelType) -> &Mutex<()> {
    let lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    &self.label_op_locks[lock_id]
  }

  fn level0_block_range(&self, internal_id: TableInt) -> Result<std::ops::Range<usize>> {
    let words = 1 + self.max_m0;
    let start = internal_id as usize * words;
    let end = start + words;
    if end > self.level0_links.len() {
      return Err(Error::InvalidIndexFormat(
        "internalId out of bounds".to_string(),
      ));
    }
    Ok(start..end)
  }

  fn level0_block(&self, internal_id: TableInt) -> Result<&[AtomicU32]> {
    let r = self.level0_block_range(internal_id)?;
    Ok(&self.level0_links[r])
  }

  fn upper_block(&self, internal_id: TableInt, level: usize) -> Result<&[AtomicU32]> {
    debug_assert!(level > 0);
    let Some(raw) = self
      .link_lists
      .get(internal_id as usize)
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

  fn linklist_at_level(&self, internal_id: TableInt, level: usize) -> Result<LinkList<'_>> {
    let block = if level == 0 {
      self.level0_block(internal_id)?
    } else {
      self.upper_block(internal_id, level)?
    };
    let header = block[0].load(Ordering::Acquire);
    let cnt = linklist_count(header);
    let cap = if level == 0 { self.max_m0 } else { self.max_m };
    if cnt > cap {
      return Err(Error::InvalidIndexFormat("linklist too large".to_string()));
    }
    Ok(LinkList {
      data: &block[1..],
      len: cnt,
    })
  }

  fn set_linklist_count_atomic(&self, header: &AtomicU32, count: usize) -> Result<()> {
    if count > u16::MAX as usize {
      return Err(Error::InvalidIndexFormat("linklist too large".to_string()));
    }
    let mut old = header.load(Ordering::Relaxed);
    loop {
      let mut new = old;
      set_linklist_count(&mut new, count);
      match header.compare_exchange_weak(old, new, Ordering::Release, Ordering::Relaxed) {
        Ok(_) => return Ok(()),
        Err(actual) => old = actual,
      }
    }
  }

  fn is_marked_deleted_internal(&self, internal_id: TableInt) -> bool {
    let Ok(block) = self.level0_block(internal_id) else {
      return false;
    };
    let header = block[0].load(Ordering::Acquire);
    (header & DELETE_MARK) != 0
  }

  fn mark_deleted_internal(&self, internal_id: TableInt) -> Result<()> {
    if internal_id as usize >= self.get_current_element_count() {
      return Err(Error::InvalidIndexFormat(
        "internalId out of bounds".to_string(),
      ));
    }
    let header = self.level0_block(internal_id)?[0].fetch_or(DELETE_MARK, Ordering::AcqRel);
    if (header & DELETE_MARK) != 0 {
      return Err(Error::InvalidIndexFormat(
        "The requested to delete element is already deleted".to_string(),
      ));
    }
    self.num_deleted.fetch_add(1, Ordering::AcqRel);
    if self.allow_replace_deleted {
      self.deleted_elements.lock().insert(internal_id);
    }
    Ok(())
  }

  fn unmark_deleted_internal(&self, internal_id: TableInt) -> Result<()> {
    if internal_id as usize >= self.get_current_element_count() {
      return Err(Error::InvalidIndexFormat(
        "internalId out of bounds".to_string(),
      ));
    }
    let header = self.level0_block(internal_id)?[0].fetch_and(!DELETE_MARK, Ordering::AcqRel);
    if (header & DELETE_MARK) == 0 {
      return Err(Error::InvalidIndexFormat(
        "The requested to undelete element is not deleted".to_string(),
      ));
    }
    self.num_deleted.fetch_sub(1, Ordering::AcqRel);
    if self.allow_replace_deleted {
      self.deleted_elements.lock().remove(&internal_id);
    }
    Ok(())
  }

  pub fn mark_delete(&self, label: LabelType) -> Result<()> {
    let _mutation_guard = self.mutation_lock.read();
    let _label_lock = self.label_op_lock(label).lock();
    let internal_id = self
      .label_lookup
      .lock()
      .get(&label)
      .copied()
      .ok_or(Error::LabelNotFound(label))?;
    self.mark_deleted_internal(internal_id)
  }

  pub fn unmark_delete(&self, label: LabelType) -> Result<()> {
    let _mutation_guard = self.mutation_lock.read();
    let _label_lock = self.label_op_lock(label).lock();
    let internal_id = self
      .label_lookup
      .lock()
      .get(&label)
      .copied()
      .ok_or(Error::LabelNotFound(label))?;
    self.unmark_deleted_internal(internal_id)
  }

  pub fn get_external_label(&self, internal_id: TableInt) -> Result<LabelType> {
    if internal_id as usize >= self.get_current_element_count() {
      return Err(Error::InvalidIndexFormat(
        "internalId out of bounds".to_string(),
      ));
    }
    Ok(self.labels[internal_id as usize].load(Ordering::Acquire))
  }

  pub fn get_data_by_label(&self, label: LabelType) -> Result<Arc<Vec<f32>>> {
    let internal_id = self
      .label_lookup
      .lock()
      .get(&label)
      .copied()
      .ok_or(Error::LabelNotFound(label))?;
    if self.is_marked_deleted_internal(internal_id) {
      return Err(Error::LabelNotFound(label));
    }
    self
      .vectors
      .get(internal_id as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
      .load_full()
      .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))
  }

  fn get_random_level(&self) -> i32 {
    let mut u: f64 = self.level_rng.lock().gen();
    if u == 0.0 {
      u = f64::MIN_POSITIVE;
    }
    let r = -u.ln() * self.mult;
    r as i32
  }

  fn distance_between_internal(&self, a: TableInt, b: TableInt) -> Result<f32> {
    let va = self
      .vectors
      .get(a as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
      .load();
    let va = va
      .as_ref()
      .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
    let vb = self
      .vectors
      .get(b as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
      .load();
    let vb = vb
      .as_ref()
      .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
    Ok(self.space.distance(va.as_slice(), vb.as_slice()))
  }

  fn vector_guard(&self, internal_id: TableInt) -> Result<arc_swap::Guard<Option<Arc<Vec<f32>>>>> {
    self
      .vectors
      .get(internal_id as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))
      .map(|v| v.load())
  }

  fn distance_query_to_internal(&self, query: &[f32], internal_id: TableInt) -> Result<f32> {
    let v = self.vector_guard(internal_id)?;
    let v = v
      .as_ref()
      .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
    Ok(self.space.distance(query, v.as_slice()))
  }

  fn get_neighbors_by_heuristic2(
    &self,
    top_candidates: &mut BinaryHeap<(OrderedFloat<f32>, TableInt)>,
    m: usize,
  ) -> Result<()> {
    if top_candidates.len() < m {
      return Ok(());
    }

    let mut queue_closest: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
    while let Some((dist, id)) = top_candidates.pop() {
      queue_closest.push((OrderedFloat(-dist.0), id));
    }

    let mut return_list: Vec<(OrderedFloat<f32>, TableInt)> = Vec::with_capacity(m);
    while let Some((neg_dist_to_query, cur_id)) = queue_closest.pop() {
      if return_list.len() >= m {
        break;
      }
      let dist_to_query = -neg_dist_to_query.0;

      let mut good = true;
      for &(_, selected_id) in &return_list {
        let cur_dist = self.distance_between_internal(selected_id, cur_id)?;
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

  fn search_base_layer(
    &self,
    ep_id: TableInt,
    data_point: &[f32],
    layer: usize,
  ) -> Result<BinaryHeap<(OrderedFloat<f32>, TableInt)>> {
    let mut visited = self.visited_list_pool.get();
    let visited_tag = visited.tag;
    let visited_mass = visited.mass_mut();

    let mut top_candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
    let mut candidate_set: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();

    let mut lower_bound;
    if !self.is_marked_deleted_internal(ep_id) {
      let dist = self.distance_query_to_internal(data_point, ep_id)?;
      top_candidates.push((OrderedFloat(dist), ep_id));
      lower_bound = dist;
      candidate_set.push((OrderedFloat(-dist), ep_id));
    } else {
      lower_bound = f32::INFINITY;
      candidate_set.push((OrderedFloat(-lower_bound), ep_id));
    }
    visited_mass[ep_id as usize] = visited_tag;

    while let Some((neg_dist, cur_node)) = candidate_set.pop() {
      let cur_dist = -neg_dist.0;
      if cur_dist > lower_bound && top_candidates.len() == self.ef_construction {
        break;
      }

      for candidate_id in self.linklist_at_level(cur_node, layer)? {
        if visited_mass[candidate_id as usize] == visited_tag {
          continue;
        }
        visited_mass[candidate_id as usize] = visited_tag;

        let dist1 = self.distance_query_to_internal(data_point, candidate_id)?;
        if top_candidates.len() < self.ef_construction || lower_bound > dist1 {
          candidate_set.push((OrderedFloat(-dist1), candidate_id));
          if !self.is_marked_deleted_internal(candidate_id) {
            top_candidates.push((OrderedFloat(dist1), candidate_id));
          }
          if top_candidates.len() > self.ef_construction {
            top_candidates.pop();
          }
          if let Some((worst, _)) = top_candidates.peek() {
            lower_bound = worst.0;
          }
        }
      }
    }

    Ok(top_candidates)
  }

  fn mutually_connect_new_element(
    &self,
    cur_c: TableInt,
    top_candidates: &mut BinaryHeap<(OrderedFloat<f32>, TableInt)>,
    level: usize,
    is_update: bool,
  ) -> Result<TableInt> {
    self.get_neighbors_by_heuristic2(top_candidates, self.m)?;
    if top_candidates.len() > self.m {
      return Err(Error::InvalidIndexFormat(
        "heuristic returned more than M candidates".to_string(),
      ));
    }

    let mut selected_neighbors: Vec<TableInt> = Vec::with_capacity(self.m);
    while let Some((_dist, id)) = top_candidates.pop() {
      selected_neighbors.push(id);
    }

    let next_closest_entry_point = *selected_neighbors
      .last()
      .ok_or_else(|| Error::InvalidIndexFormat("empty selected neighbor list".to_string()))?;

    for &neighbor in &selected_neighbors {
      if level > self.element_levels[neighbor as usize].load(Ordering::Acquire) as usize {
        return Err(Error::InvalidIndexFormat(
          "Trying to make a link on a non-existent level".to_string(),
        ));
      }
    }

    {
      let _cur_lock = self.link_list_locks[cur_c as usize].lock();
      let block = if level == 0 {
        self.level0_block(cur_c)?
      } else {
        self.upper_block(cur_c, level)?
      };
      let header = block[0].load(Ordering::Acquire);
      if linklist_count(header) != 0 && !is_update {
        return Err(Error::InvalidIndexFormat(
          "The newly inserted element should have blank link list".to_string(),
        ));
      }
      for (idx, &neighbor) in selected_neighbors.iter().enumerate() {
        block[1 + idx].store(neighbor, Ordering::Relaxed);
      }
      self.set_linklist_count_atomic(&block[0], selected_neighbors.len())?;
    }

    self.connect_backlinks(cur_c, &selected_neighbors, level, is_update)?;

    Ok(next_closest_entry_point)
  }

  fn connect_backlinks(
    &self,
    cur_c: TableInt,
    selected_neighbors: &[TableInt],
    level: usize,
    is_update: bool,
  ) -> Result<()> {
    let mcurmax = if level > 0 { self.max_m } else { self.max_m0 };

    for &neighbor in selected_neighbors {
      if neighbor == cur_c {
        return Err(Error::InvalidIndexFormat(
          "Trying to connect an element to itself".to_string(),
        ));
      }
      if level > self.element_levels[neighbor as usize].load(Ordering::Acquire) as usize {
        return Err(Error::InvalidIndexFormat(
          "Trying to make a link on a non-existent level".to_string(),
        ));
      }

      let _lock = self.link_list_locks[neighbor as usize].lock();
      let existing = self.linklist_at_level(neighbor, level)?;
      let sz_link_list_other = existing.len();
      let is_cur_c_present = is_update && existing.into_iter().any(|id| id == cur_c);
      if sz_link_list_other > mcurmax {
        return Err(Error::InvalidIndexFormat(
          "Bad value of sz_link_list_other".to_string(),
        ));
      }

      if is_cur_c_present {
        continue;
      }

      if sz_link_list_other < mcurmax {
        let block = if level == 0 {
          self.level0_block(neighbor)?
        } else {
          self.upper_block(neighbor, level)?
        };
        block[1 + sz_link_list_other].store(cur_c, Ordering::Relaxed);
        self.set_linklist_count_atomic(&block[0], sz_link_list_other + 1)?;
        continue;
      }

      let existing = existing.into_iter().collect::<Vec<_>>();
      let d_max = self.distance_between_internal(cur_c, neighbor)?;
      let mut candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
      candidates.push((OrderedFloat(d_max), cur_c));

      for existing in existing {
        let dist = self.distance_between_internal(existing, neighbor)?;
        candidates.push((OrderedFloat(dist), existing));
      }

      self.get_neighbors_by_heuristic2(&mut candidates, mcurmax)?;

      let mut new_neighbors: Vec<TableInt> = Vec::with_capacity(candidates.len());
      while let Some((_dist, id)) = candidates.pop() {
        new_neighbors.push(id);
      }

      let block = if level == 0 {
        self.level0_block(neighbor)?
      } else {
        self.upper_block(neighbor, level)?
      };
      for (indx, &id) in new_neighbors.iter().enumerate() {
        block[1 + indx].store(id, Ordering::Relaxed);
      }
      self.set_linklist_count_atomic(&block[0], new_neighbors.len())?;
    }

    Ok(())
  }

  fn get_connections_with_lock(
    &self,
    internal_id: TableInt,
    level: usize,
  ) -> Result<Vec<TableInt>> {
    let _lock = self.link_list_locks[internal_id as usize].lock();
    Ok(self
      .linklist_at_level(internal_id, level)?
      .into_iter()
      .collect())
  }

  fn repair_connections_for_update(
    &self,
    data_point: &[f32],
    entry_point_internal_id: TableInt,
    data_point_internal_id: TableInt,
    data_point_level: usize,
    max_level: usize,
  ) -> Result<()> {
    let mut curr_obj = entry_point_internal_id;
    if data_point_level < max_level {
      let mut curdist = self.distance_query_to_internal(data_point, curr_obj)?;
      for level in (data_point_level + 1..=max_level).rev() {
        let mut changed = true;
        while changed {
          changed = false;
          for cand in self.linklist_at_level(curr_obj, level)? {
            let d = self.distance_query_to_internal(data_point, cand)?;
            if d < curdist {
              curdist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
    }

    if data_point_level > max_level {
      return Err(Error::InvalidIndexFormat(
        "Level of item to be updated cannot be bigger than max level".to_string(),
      ));
    }

    for level in (0..=data_point_level).rev() {
      let mut top_candidates = self.search_base_layer(curr_obj, data_point, level)?;
      let mut filtered: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
      while let Some(cand) = top_candidates.pop() {
        if cand.1 != data_point_internal_id {
          filtered.push(cand);
        }
      }

      if filtered.is_empty() {
        continue;
      }

      let ep_deleted = self.is_marked_deleted_internal(entry_point_internal_id);
      if ep_deleted {
        let dist = self.distance_query_to_internal(data_point, entry_point_internal_id)?;
        filtered.push((OrderedFloat(dist), entry_point_internal_id));
        if filtered.len() > self.ef_construction {
          filtered.pop();
        }
      }

      curr_obj =
        self.mutually_connect_new_element(data_point_internal_id, &mut filtered, level, true)?;
    }

    Ok(())
  }

  fn update_point(
    &self,
    data_point: &[f32],
    internal_id: TableInt,
    update_neighbor_probability: f32,
  ) -> Result<()> {
    self
      .vectors
      .get(internal_id as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
      .store(Some(Arc::new(data_point.to_vec())));

    let max_level_copy = self.max_level.load(Ordering::Acquire);
    let entry_point_copy = self.enter_point_node();
    if entry_point_copy == Some(internal_id) && self.get_current_element_count() == 1 {
      return Ok(());
    }

    let entry_point_copy = entry_point_copy.ok_or(Error::EmptyIndex)?;
    let elem_level = self.element_levels[internal_id as usize].load(Ordering::Acquire);
    if elem_level < 0 {
      return Err(Error::InvalidIndexFormat(
        "element level is negative".to_string(),
      ));
    }
    let elem_level = elem_level as usize;

    for layer in 0..=elem_level {
      let mut s_cand: HashSet<TableInt> = HashSet::new();
      let mut s_neigh: HashSet<TableInt> = HashSet::new();

      let list_one_hop = self.get_connections_with_lock(internal_id, layer)?;
      if list_one_hop.is_empty() {
        continue;
      }

      s_cand.insert(internal_id);

      let update_decisions: Vec<f32> = {
        let mut rng = self.update_probability_rng.lock();
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
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();

        for cand in s_cand.iter().copied() {
          if cand == neigh {
            continue;
          }
          let dist = self.distance_between_internal(neigh, cand)?;
          if candidates.len() < elements_to_keep {
            candidates.push((OrderedFloat(dist), cand));
          } else if dist < candidates.peek().unwrap().0 .0 {
            candidates.pop();
            candidates.push((OrderedFloat(dist), cand));
          }
        }

        let cap = if layer == 0 { self.max_m0 } else { self.max_m };
        self.get_neighbors_by_heuristic2(&mut candidates, cap)?;

        let _lock = self.link_list_locks[neigh as usize].lock();
        let block = if layer == 0 {
          self.level0_block(neigh)?
        } else {
          self.upper_block(neigh, layer)?
        };

        let cand_size = candidates.len();
        for idx in 0..cand_size {
          block[1 + idx].store(candidates.pop().unwrap().1, Ordering::Relaxed);
        }
        self.set_linklist_count_atomic(&block[0], cand_size)?;
      }
    }

    self.repair_connections_for_update(
      data_point,
      entry_point_copy,
      internal_id,
      elem_level,
      max_level_copy.max(0) as usize,
    )?;

    Ok(())
  }

  fn add_point_at_level(
    &self,
    data_point: &[f32],
    label: LabelType,
    forced_level: Option<i32>,
  ) -> Result<TableInt> {
    let _mutation_guard = self.mutation_lock.read();

    let cur_c: TableInt;
    {
      let mut label_lookup = self.label_lookup.lock();
      if let Some(&existing) = label_lookup.get(&label) {
        if self.allow_replace_deleted && self.is_marked_deleted_internal(existing) {
          return Err(Error::InvalidIndexFormat(
            "Can't use addPoint to update deleted elements if replacement of deleted elements is enabled."
              .to_string(),
          ));
        }
        drop(label_lookup);
        if self.is_marked_deleted_internal(existing) {
          self.unmark_deleted_internal(existing)?;
        }
        self.update_point(data_point, existing, 1.0)?;
        return Ok(existing);
      }

      let cur_count = self.cur_element_count.load(Ordering::Acquire);
      if cur_count >= self.max_elements {
        return Err(Error::IndexFull {
          max_elements: self.max_elements,
        });
      }
      cur_c = cur_count as TableInt;
      self.cur_element_count.store(cur_count + 1, Ordering::Release);
      label_lookup.insert(label, cur_c);
    }

    let curlevel = forced_level.unwrap_or_else(|| self.get_random_level());
    if curlevel < 0 {
      return Err(Error::InvalidIndexFormat(
        "level must be >= 0".to_string(),
      ));
    }
    self.element_levels[cur_c as usize].store(curlevel, Ordering::Release);

    // "memset" base layer for the new element.
    for word in self.level0_block(cur_c)? {
      word.store(0, Ordering::Relaxed);
    }

    // Initialization of the data and label.
    self.labels[cur_c as usize].store(label, Ordering::Release);
    self
      .vectors
      .get(cur_c as usize)
      .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
      .store(Some(Arc::new(data_point.to_vec())));

    if curlevel > 0 {
      let words = (curlevel as usize) * (1 + self.max_m);
      let mut raw = Vec::with_capacity(words);
      raw.resize_with(words, || AtomicU32::new(0));
      self
        .link_lists
        .get(cur_c as usize)
        .ok_or_else(|| Error::InvalidIndexFormat("internalId out of bounds".to_string()))?
        .set(raw.into_boxed_slice())
        .map_err(|_| Error::InvalidIndexFormat("linklist already initialized".to_string()))?;
    }

    let mut templock = Some(self.global.lock());
    let maxlevelcopy = self.max_level.load(Ordering::Acquire);
    if curlevel <= maxlevelcopy {
      drop(templock.take());
    }

    let mut curr_obj = self.enter_point_node();
    let Some(enterpoint_copy) = curr_obj else {
      // First element.
      self.enter_point_node.store(cur_c, Ordering::Release);
      self.max_level.store(curlevel, Ordering::Release);
      return Ok(cur_c);
    };

    if maxlevelcopy >= 0 && curlevel < maxlevelcopy {
      let mut curdist = self.distance_query_to_internal(data_point, enterpoint_copy)?;
      let mut curr = enterpoint_copy;
      for level in ((curlevel + 1) as usize..=maxlevelcopy as usize).rev() {
        let mut changed = true;
        while changed {
          changed = false;
          let _lock = self.link_list_locks[curr as usize].lock();
          for cand in self.linklist_at_level(curr, level)? {
            let d = self.distance_query_to_internal(data_point, cand)?;
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

    let mut curr_obj = curr_obj.expect("enterpoint checked above");

    let ep_deleted = self.is_marked_deleted_internal(enterpoint_copy);
    let max_conn_level =
      usize::min(curlevel.max(0) as usize, maxlevelcopy.max(0) as usize);

    let mut selected_neighbors_per_level: Vec<Vec<TableInt>> = vec![Vec::new(); max_conn_level + 1];

    // Phase 1: fill `cur_c`'s own link lists, but do NOT publish backlinks yet.
    for level in (0..=max_conn_level).rev() {
      let mut top_candidates = self.search_base_layer(curr_obj, data_point, level)?;
      if ep_deleted {
        let dist = self.distance_query_to_internal(data_point, enterpoint_copy)?;
        top_candidates.push((OrderedFloat(dist), enterpoint_copy));
        if top_candidates.len() > self.ef_construction {
          top_candidates.pop();
        }
      }

      self.get_neighbors_by_heuristic2(&mut top_candidates, self.m)?;
      if top_candidates.len() > self.m {
        return Err(Error::InvalidIndexFormat(
          "heuristic returned more than M candidates".to_string(),
        ));
      }

      let mut selected_neighbors: Vec<TableInt> = Vec::with_capacity(self.m);
      while let Some((_dist, id)) = top_candidates.pop() {
        selected_neighbors.push(id);
      }

      let next_closest_entry_point = *selected_neighbors
        .last()
        .ok_or_else(|| Error::InvalidIndexFormat("empty selected neighbor list".to_string()))?;

      for &neighbor in &selected_neighbors {
        if level > self.element_levels[neighbor as usize].load(Ordering::Acquire) as usize {
          return Err(Error::InvalidIndexFormat(
            "Trying to make a link on a non-existent level".to_string(),
          ));
        }
      }

      {
        let _cur_lock = self.link_list_locks[cur_c as usize].lock();
        let block = if level == 0 {
          self.level0_block(cur_c)?
        } else {
          self.upper_block(cur_c, level)?
        };
        let header = block[0].load(Ordering::Acquire);
        if linklist_count(header) != 0 {
          return Err(Error::InvalidIndexFormat(
            "The newly inserted element should have blank link list".to_string(),
          ));
        }
        for (idx, &neighbor) in selected_neighbors.iter().enumerate() {
          block[1 + idx].store(neighbor, Ordering::Relaxed);
        }
        self.set_linklist_count_atomic(&block[0], selected_neighbors.len())?;
      }

      selected_neighbors_per_level[level] = selected_neighbors;
      curr_obj = next_closest_entry_point;
    }

    // Phase 2: publish backlinks.
    for level in (0..=max_conn_level).rev() {
      self.connect_backlinks(cur_c, &selected_neighbors_per_level[level], level, false)?;
    }

    if curlevel > maxlevelcopy {
      debug_assert!(templock.is_some());
      self.enter_point_node.store(cur_c, Ordering::Release);
      self.max_level.store(curlevel, Ordering::Release);
    }

    Ok(cur_c)
  }

  pub fn add_point(&self, data_point: &[f32], label: LabelType) -> Result<TableInt> {
    if data_point.len() != self.space.dim() {
      return Err(Error::DimensionMismatch {
        expected: self.space.dim(),
        actual: data_point.len(),
      });
    }
    let _label_lock = self.label_op_lock(label).lock();
    self.add_point_at_level(data_point, label, None)
  }

  pub fn add_point_replace_deleted(&self, data_point: &[f32], label: LabelType) -> Result<TableInt> {
    if !self.allow_replace_deleted {
      return Err(Error::InvalidIndexFormat(
        "Replacement of deleted elements is disabled in constructor".to_string(),
      ));
    }
    if data_point.len() != self.space.dim() {
      return Err(Error::DimensionMismatch {
        expected: self.space.dim(),
        actual: data_point.len(),
      });
    }

    let _mutation_guard = self.mutation_lock.read();
    let _label_lock = self.label_op_lock(label).lock();

    {
      let label_lookup = self.label_lookup.lock();
      if label_lookup.contains_key(&label) {
        drop(label_lookup);
        return self.add_point_at_level(data_point, label, None);
      }
    }

    let internal_id_replaced = {
      let mut deleted = self.deleted_elements.lock();
      deleted.iter().next().copied().map(|id| {
        deleted.remove(&id);
        id
      })
    };

    let Some(internal_id_replaced) = internal_id_replaced else {
      return self.add_point_at_level(data_point, label, None);
    };

    let label_replaced = self.get_external_label(internal_id_replaced)?;
    self.labels[internal_id_replaced as usize].store(label, Ordering::Release);

    {
      let mut label_lookup = self.label_lookup.lock();
      label_lookup.remove(&label_replaced);
      label_lookup.insert(label, internal_id_replaced);
    }

    self.unmark_deleted_internal(internal_id_replaced)?;
    self.update_point(data_point, internal_id_replaced, 1.0)?;

    Ok(internal_id_replaced)
  }

  fn search_base_layer_st<const BARE_BONE: bool>(
    &self,
    ep_id: TableInt,
    query: &[f32],
    ef: usize,
    filter: Option<&dyn Fn(LabelType) -> bool>,
  ) -> Result<BinaryHeap<(OrderedFloat<f32>, TableInt)>> {
    let mut visited = self.visited_list_pool.get();
    let visited_tag = visited.tag;
    let visited_mass = visited.mass_mut();

    let mut top_candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
    let mut candidate_set: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();

    let mut lower_bound;
    let ep_label = self.get_external_label(ep_id)?;
    if BARE_BONE || (!self.is_marked_deleted_internal(ep_id) && label_allowed(filter, ep_label)) {
      let dist = self.distance_query_to_internal(query, ep_id)?;
      lower_bound = dist;
      top_candidates.push((OrderedFloat(dist), ep_id));
      candidate_set.push((OrderedFloat(-dist), ep_id));
    } else {
      lower_bound = f32::INFINITY;
      candidate_set.push((OrderedFloat(-lower_bound), ep_id));
    }

    visited_mass[ep_id as usize] = visited_tag;

    while let Some((neg_dist, current_node_id)) = candidate_set.pop() {
      let candidate_dist = -neg_dist.0;

      let flag_stop_search = if BARE_BONE {
        candidate_dist > lower_bound
      } else {
        candidate_dist > lower_bound && top_candidates.len() == ef
      };
      if flag_stop_search {
        break;
      }

      for candidate_id in self.linklist_at_level(current_node_id, 0)? {
        if visited_mass[candidate_id as usize] == visited_tag {
          continue;
        }
        visited_mass[candidate_id as usize] = visited_tag;

        let dist = self.distance_query_to_internal(query, candidate_id)?;
        let flag_consider_candidate = top_candidates.len() < ef || lower_bound > dist;
        if !flag_consider_candidate {
          continue;
        }

        candidate_set.push((OrderedFloat(-dist), candidate_id));

        if BARE_BONE {
          top_candidates.push((OrderedFloat(dist), candidate_id));
        } else {
          let cand_label = self.get_external_label(candidate_id)?;
          if !self.is_marked_deleted_internal(candidate_id) && label_allowed(filter, cand_label) {
            top_candidates.push((OrderedFloat(dist), candidate_id));
          }
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

  fn search_base_layer_st_stop_condition(
    &self,
    ep_id: TableInt,
    query: &[f32],
    filter: Option<&dyn Fn(LabelType) -> bool>,
    stop_condition: &mut dyn SearchStopCondition,
  ) -> Result<BinaryHeap<(OrderedFloat<f32>, TableInt)>> {
    let mut visited = self.visited_list_pool.get();
    let visited_tag = visited.tag;
    let visited_mass = visited.mass_mut();

    let mut top_candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
    let mut candidate_set: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();

    let mut lower_bound;
    let ep_label = self.get_external_label(ep_id)?;
    if !self.is_marked_deleted_internal(ep_id) && label_allowed(filter, ep_label) {
      let ep_data = self.vector_guard(ep_id)?;
      let ep_data = ep_data
        .as_ref()
        .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
      let dist = self.space.distance(query, ep_data.as_slice());
      lower_bound = dist;
      top_candidates.push((OrderedFloat(dist), ep_id));
      stop_condition.add_point_to_result(ep_label, ep_data.as_slice(), dist);
      candidate_set.push((OrderedFloat(-dist), ep_id));
    } else {
      lower_bound = f32::INFINITY;
      candidate_set.push((OrderedFloat(-lower_bound), ep_id));
    }

    visited_mass[ep_id as usize] = visited_tag;

    while let Some((neg_dist, current_node_id)) = candidate_set.pop() {
      let candidate_dist = -neg_dist.0;
      if stop_condition.should_stop_search(candidate_dist, lower_bound) {
        break;
      }

      for candidate_id in self.linklist_at_level(current_node_id, 0)? {
        if visited_mass[candidate_id as usize] == visited_tag {
          continue;
        }
        visited_mass[candidate_id as usize] = visited_tag;

        let cand_data = self.vector_guard(candidate_id)?;
        let cand_data = cand_data
          .as_ref()
          .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
        let dist = self.space.distance(query, cand_data.as_slice());

        if !stop_condition.should_consider_candidate(dist, lower_bound) {
          continue;
        }

        candidate_set.push((OrderedFloat(-dist), candidate_id));

        let cand_label = self.get_external_label(candidate_id)?;
        if !self.is_marked_deleted_internal(candidate_id) && label_allowed(filter, cand_label) {
          top_candidates.push((OrderedFloat(dist), candidate_id));
          stop_condition.add_point_to_result(cand_label, cand_data.as_slice(), dist);
        }

        while stop_condition.should_remove_extra() {
          let Some((dist, id)) = top_candidates.pop() else {
            break;
          };
          let label = self.get_external_label(id)?;
          let data = self.vector_guard(id)?;
          let data = data
            .as_ref()
            .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
          stop_condition.remove_point_from_result(label, data.as_slice(), dist.0);
        }

        if let Some((worst, _)) = top_candidates.peek() {
          lower_bound = worst.0;
        }
      }
    }

    Ok(top_candidates)
  }

  pub fn search_stop_condition_closest(
    &self,
    query: &[f32],
    stop_condition: &mut dyn SearchStopCondition,
    filter: Option<&dyn Fn(LabelType) -> bool>,
  ) -> Result<Vec<(LabelType, f32)>> {
    if query.len() != self.space.dim() {
      return Err(Error::DimensionMismatch {
        expected: self.space.dim(),
        actual: query.len(),
      });
    }
    if self.get_current_element_count() == 0 {
      return Ok(Vec::new());
    }

    let mut curr_obj = self.enter_point_node().ok_or(Error::EmptyIndex)?;
    let mut cur_dist = self.distance_query_to_internal(query, curr_obj)?;

    let max_level = self.max_level.load(Ordering::Acquire);
    for level in (1..=max_level.max(0) as usize).rev() {
      let mut changed = true;
      while changed {
        changed = false;
        for cand in self.linklist_at_level(curr_obj, level)? {
          let d = self.distance_query_to_internal(query, cand)?;
          if d < cur_dist {
            cur_dist = d;
            curr_obj = cand;
            changed = true;
          }
        }
      }
    }

    let mut top_candidates =
      self.search_base_layer_st_stop_condition(curr_obj, query, filter, stop_condition)?;

    let mut result: Vec<(LabelType, f32)> = Vec::with_capacity(top_candidates.len());
    while let Some((dist, id)) = top_candidates.pop() {
      result.push((self.get_external_label(id)?, dist.0));
    }
    result.reverse();
    stop_condition.filter_results(&mut result);
    Ok(result)
  }

  pub fn search_knn(
    &self,
    query: &[f32],
    k: usize,
    filter: Option<&dyn Fn(LabelType) -> bool>,
  ) -> Result<Vec<(LabelType, f32)>> {
    if query.len() != self.space.dim() {
      return Err(Error::DimensionMismatch {
        expected: self.space.dim(),
        actual: query.len(),
      });
    }
    if self.get_current_element_count() == 0 {
      return Ok(Vec::new());
    }

    let mut curr_obj = self.enter_point_node().ok_or(Error::EmptyIndex)?;
    let mut cur_dist = self.distance_query_to_internal(query, curr_obj)?;

    let max_level = self.max_level.load(Ordering::Acquire);
    for level in (1..=max_level.max(0) as usize).rev() {
      let mut changed = true;
      while changed {
        changed = false;
        for cand in self.linklist_at_level(curr_obj, level)? {
          let d = self.distance_query_to_internal(query, cand)?;
          if d < cur_dist {
            cur_dist = d;
            curr_obj = cand;
            changed = true;
          }
        }
      }
    }

    let ef = max(self.ef.load(Ordering::Acquire), k);
    let bare_bone_search = self.get_deleted_count() == 0 && filter.is_none();
    let mut top_candidates = if bare_bone_search {
      self.search_base_layer_st::<true>(curr_obj, query, ef, filter)?
    } else {
      self.search_base_layer_st::<false>(curr_obj, query, ef, filter)?
    };

    while top_candidates.len() > k {
      top_candidates.pop();
    }

    let mut res = Vec::with_capacity(top_candidates.len());
    while let Some((dist, id)) = top_candidates.pop() {
      res.push((self.get_external_label(id)?, dist.0));
    }
    res.reverse();
    Ok(res)
  }

  pub fn resize_index(&mut self, new_max_elements: usize) -> Result<()> {
    let _mutation_guard = self.mutation_lock.write();

    let cur_count = self.cur_element_count.load(Ordering::Acquire);
    if new_max_elements < cur_count {
      return Err(Error::InvalidIndexFormat(
        "Cannot resize, max element is less than the current number of elements".to_string(),
      ));
    }
    if new_max_elements > TableInt::MAX as usize {
      return Err(Error::InvalidIndexFormat(
        "new_max_elements exceeds internal id range".to_string(),
      ));
    }

    self.max_elements = new_max_elements;
    self.visited_list_pool.resize(1, new_max_elements);

    self
      .element_levels
      .resize_with(new_max_elements, || AtomicI32::new(0));
    self.labels.resize_with(new_max_elements, || AtomicUsize::new(0));
    self.vectors.resize_with(new_max_elements, ArcSwapOption::empty);
    self
      .link_list_locks
      .resize_with(new_max_elements, || Mutex::new(()));

    let words_per_element = 1 + self.max_m0;
    self.level0_links.resize_with(
      new_max_elements * words_per_element,
      || AtomicU32::new(0),
    );

    self.link_lists.resize_with(new_max_elements, OnceLock::new);
    Ok(())
  }

  pub fn index_file_size(&self) -> usize {
    let _mutation_guard = self.mutation_lock.write();
    let cur_element_count = self.cur_element_count.load(Ordering::Acquire);

    let size_links_level0 = (1 + self.max_m0) * size_of::<u32>();
    let size_data = self.space.dim() * size_of::<f32>();
    let size_data_per_element = size_links_level0 + size_data + size_of::<LabelType>();
    let size_links_per_element = (1 + self.max_m) * size_of::<u32>();

    let mut size = 0usize;
    size += size_of::<usize>(); // offsetLevel0
    size += size_of::<usize>(); // max_elements
    size += size_of::<usize>(); // cur_element_count
    size += size_of::<usize>(); // size_data_per_element
    size += size_of::<usize>(); // label_offset
    size += size_of::<usize>(); // offsetData
    size += size_of::<i32>(); // maxlevel
    size += size_of::<u32>(); // enterpoint_node
    size += size_of::<usize>(); // maxM
    size += size_of::<usize>(); // maxM0
    size += size_of::<usize>(); // M
    size += size_of::<f64>(); // mult
    size += size_of::<usize>(); // ef_construction

    size += cur_element_count * size_data_per_element;

    for i in 0..cur_element_count {
      let level = self.element_levels[i].load(Ordering::Acquire);
      let link_list_size = if level > 0 {
        size_links_per_element * (level as usize)
      } else {
        0
      };
      size += size_of::<u32>();
      size += link_list_size;
    }
    size
  }

  pub fn save_to_writer(&self, mut w: impl Write) -> Result<()> {
    let _mutation_guard = self.mutation_lock.write();

    let cur_element_count = self.cur_element_count.load(Ordering::Acquire);
    let max_level = self.max_level.load(Ordering::Acquire);
    let enter_point_raw = self.enter_point_node.load(Ordering::Acquire);

    let size_links_level0 = (1 + self.max_m0) * size_of::<u32>();
    let data_size = self.space.dim() * size_of::<f32>();
    let size_data_per_element = size_links_level0 + data_size + size_of::<LabelType>();
    let label_offset = size_links_level0 + data_size;
    let offset_data = size_links_level0;

    w.write_all(&0usize.to_le_bytes())?;
    w.write_all(&self.max_elements.to_le_bytes())?;
    w.write_all(&cur_element_count.to_le_bytes())?;
    w.write_all(&size_data_per_element.to_le_bytes())?;
    w.write_all(&label_offset.to_le_bytes())?;
    w.write_all(&offset_data.to_le_bytes())?;
    w.write_all(&max_level.to_le_bytes())?;
    w.write_all(&enter_point_raw.to_le_bytes())?;
    w.write_all(&self.max_m.to_le_bytes())?;
    w.write_all(&self.max_m0.to_le_bytes())?;
    w.write_all(&self.m.to_le_bytes())?;
    w.write_all(&self.mult.to_le_bytes())?;
    w.write_all(&self.ef_construction.to_le_bytes())?;

    let words_per_element = 1 + self.max_m0;
    let dim = self.space.dim();
    let mut level0_buf: Vec<u32> = vec![0u32; words_per_element];
    for i in 0..cur_element_count {
      let start = i * words_per_element;
      let end = start + words_per_element;
      for (dst, src) in level0_buf
        .iter_mut()
        .zip(self.level0_links[start..end].iter())
      {
        *dst = src.load(Ordering::Acquire);
      }
      w.write_all(bytemuck::cast_slice(&level0_buf))?;

      let v = self.vectors[i].load();
      let v = v
        .as_ref()
        .ok_or_else(|| Error::InvalidIndexFormat("missing vector".to_string()))?;
      if v.len() != dim {
        return Err(Error::InvalidIndexFormat(
          "vector dimension mismatch".to_string(),
        ));
      }
      w.write_all(bytemuck::cast_slice(v.as_slice()))?;

      let label = self.labels[i].load(Ordering::Acquire);
      w.write_all(&label.to_le_bytes())?;
    }

    let words_per_level = 1 + self.max_m;
    let mut upper_buf: Vec<u32> = Vec::new();
    for i in 0..cur_element_count {
      let level = self.element_levels[i].load(Ordering::Acquire);
      let link_list_size = if level > 0 {
        (words_per_level * level as usize * size_of::<u32>()) as u32
      } else {
        0u32
      };
      w.write_all(&link_list_size.to_le_bytes())?;
      if link_list_size != 0 {
        let Some(raw) = self.link_lists[i].get() else {
          return Err(Error::InvalidIndexFormat("missing linklist".to_string()));
        };
        if raw.len() != words_per_level * (level as usize) {
          return Err(Error::InvalidIndexFormat(
            "linklist size mismatch".to_string(),
          ));
        }
        upper_buf.resize(raw.len(), 0u32);
        for (dst, src) in upper_buf.iter_mut().zip(raw.iter()) {
          *dst = src.load(Ordering::Acquire);
        }
        w.write_all(bytemuck::cast_slice(&upper_buf))?;
      }
    }

    Ok(())
  }

  pub fn save_to_vec(&self) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(self.index_file_size());
    self.save_to_writer(&mut out)?;
    Ok(out)
  }

  pub fn load_from_reader(space: S, mut r: impl Read, max_elements: Option<usize>) -> Result<Self> {
    Self::load_from_reader_with_options(space, &mut r, max_elements, 100, false)
  }

  pub fn load_from_reader_with_options(
    space: S,
    mut r: impl Read,
    max_elements: Option<usize>,
    random_seed: u64,
    allow_replace_deleted: bool,
  ) -> Result<Self> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    Self::load_from_bytes_with_options(
      space,
      &buf,
      max_elements,
      random_seed,
      allow_replace_deleted,
    )
  }

  pub fn load_from_bytes(space: S, data: &[u8], max_elements: Option<usize>) -> Result<Self> {
    Self::load_from_bytes_with_options(space, data, max_elements, 100, false)
  }

  pub fn load_from_bytes_with_options(
    space: S,
    data: &[u8],
    max_elements: Option<usize>,
    random_seed: u64,
    allow_replace_deleted: bool,
  ) -> Result<Self> {
    let mut rd = &*data;

    let read_usize = |rd: &mut &[u8]| -> Result<usize> {
      if rd.len() < size_of::<usize>() {
        return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
      }
      let (bytes, rest) = rd.split_at(size_of::<usize>());
      *rd = rest;
      Ok(usize::from_le_bytes(bytes.try_into().unwrap()))
    };

    let read_i32 = |rd: &mut &[u8]| -> Result<i32> {
      if rd.len() < 4 {
        return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
      }
      let (bytes, rest) = rd.split_at(4);
      *rd = rest;
      Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    };

    let read_u32 = |rd: &mut &[u8]| -> Result<u32> {
      if rd.len() < 4 {
        return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
      }
      let (bytes, rest) = rd.split_at(4);
      *rd = rest;
      Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    };

    let read_f64 = |rd: &mut &[u8]| -> Result<f64> {
      if rd.len() < 8 {
        return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
      }
      let (bytes, rest) = rd.split_at(8);
      *rd = rest;
      Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    };

    let offset_level0 = read_usize(&mut rd)?;
    let file_max_elements = read_usize(&mut rd)?;
    let cur_element_count = read_usize(&mut rd)?;

    let mut max_elements = max_elements.unwrap_or(0);
    if max_elements < cur_element_count {
      max_elements = file_max_elements;
    }

    let size_data_per_element = read_usize(&mut rd)?;
    let label_offset = read_usize(&mut rd)?;
    let offset_data = read_usize(&mut rd)?;
    let max_level = read_i32(&mut rd)?;
    let enter_point_raw = read_u32(&mut rd)?;
    let max_m = read_usize(&mut rd)?;
    let max_m0 = read_usize(&mut rd)?;
    let m = read_usize(&mut rd)?;
    let mult = read_f64(&mut rd)?;
    let ef_construction = read_usize(&mut rd)?;

    if m < 2 {
      return Err(Error::InvalidIndexFormat("invalid M".to_string()));
    }

    if offset_level0 != 0 {
      return Err(Error::InvalidIndexFormat(format!(
        "unsupported offset_level0={offset_level0}"
      )));
    }
    if cur_element_count > max_elements {
      return Err(Error::InvalidIndexFormat(
        "cur_element_count > max_elements".to_string(),
      ));
    }

    let dim = space.dim();
    let expected_data_size = dim * size_of::<f32>();
    if offset_data + expected_data_size + size_of::<LabelType>() != size_data_per_element {
      return Err(Error::InvalidIndexFormat(
        "incompatible dimension for index".to_string(),
      ));
    }
    if label_offset != offset_data + expected_data_size {
      return Err(Error::InvalidIndexFormat(
        "unexpected label_offset".to_string(),
      ));
    }

    let max_m_expected = m;
    if max_m != max_m_expected {
      // hnswlib stores both `maxM_` and `M_` in the header, and they are typically equal.
      // Be strict for now to avoid subtle incompatibilities.
      return Err(Error::InvalidIndexFormat(
        "unsupported: maxM != M".to_string(),
      ));
    }

    if max_m0 != m.saturating_mul(2) {
      return Err(Error::InvalidIndexFormat(
        "unsupported: maxM0 != 2*M".to_string(),
      ));
    }

    let mut idx = Self::new(
      space,
      max_elements,
      m,
      ef_construction,
      random_seed,
      allow_replace_deleted,
    );
    idx.mult = mult;
    idx.rev_size = 1.0 / mult;
    idx.max_level.store(max_level, Ordering::Release);
    idx.enter_point_node.store(enter_point_raw, Ordering::Release);
    idx.ef.store(10, Ordering::Release);
    idx.cur_element_count
      .store(cur_element_count, Ordering::Release);

    let words_per_element = 1 + idx.max_m0;
    let bytes_per_element_links = words_per_element * size_of::<u32>();

    {
      let mut label_lookup = idx.label_lookup.lock();
      for i in 0..cur_element_count {
        if rd.len() < bytes_per_element_links + expected_data_size + size_of::<LabelType>() {
          return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
        }

        let l0_words = &idx.level0_links[i * words_per_element..(i + 1) * words_per_element];
        let (l0_bytes, rest) = rd.split_at(bytes_per_element_links);
        rd = rest;
        if let Ok(src) = bytemuck::try_cast_slice::<u8, u32>(l0_bytes) {
          for (dst, &val) in l0_words.iter().zip(src.iter()) {
            dst.store(val, Ordering::Relaxed);
          }
        } else {
          for (dst, chunk) in l0_words.iter().zip(l0_bytes.chunks_exact(4)) {
            dst.store(
              u32::from_le_bytes(chunk.try_into().unwrap()),
              Ordering::Relaxed,
            );
          }
        }

        let (v_bytes, rest) = rd.split_at(expected_data_size);
        rd = rest;
        let mut vec = vec![0.0f32; dim];
        if let Ok(src) = bytemuck::try_cast_slice::<u8, f32>(v_bytes) {
          vec.copy_from_slice(src);
        } else {
          for (dst, chunk) in vec.iter_mut().zip(v_bytes.chunks_exact(4)) {
            *dst = f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap()));
          }
        }
        idx.vectors[i].store(Some(Arc::new(vec)));

        let (label_bytes, rest) = rd.split_at(size_of::<LabelType>());
        rd = rest;
        let label = LabelType::from_le_bytes(label_bytes.try_into().unwrap());
        idx.labels[i].store(label, Ordering::Relaxed);
        if label_lookup.insert(label, i as TableInt).is_some() {
          return Err(Error::InvalidIndexFormat(
            "duplicate external label".to_string(),
          ));
        }
      }
    }

    let words_per_level = 1 + idx.max_m;
    let size_links_per_element = words_per_level * size_of::<u32>();

    for i in 0..cur_element_count {
      let link_list_size = read_u32(&mut rd)? as usize;
      if link_list_size == 0 {
        idx.element_levels[i].store(0, Ordering::Relaxed);
        continue;
      }
      if link_list_size % size_links_per_element != 0 {
        return Err(Error::InvalidIndexFormat(
          "invalid linkListSize".to_string(),
        ));
      }
      let levels = link_list_size / size_links_per_element;
      let words = link_list_size / size_of::<u32>();
      idx.element_levels[i].store(levels as i32, Ordering::Relaxed);
      if rd.len() < link_list_size {
        return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
      }
      let (bytes, rest) = rd.split_at(link_list_size);
      rd = rest;
      let mut raw = vec![0u32; words];
      if let Ok(src) = bytemuck::try_cast_slice::<u8, u32>(bytes) {
        raw.copy_from_slice(src);
      } else {
        for (dst, chunk) in raw.iter_mut().zip(bytes.chunks_exact(4)) {
          *dst = u32::from_le_bytes(chunk.try_into().unwrap());
        }
      }
      let mut atoms = Vec::with_capacity(raw.len());
      for v in raw {
        atoms.push(AtomicU32::new(v));
      }
      idx.link_lists[i]
        .set(atoms.into_boxed_slice())
        .map_err(|_| Error::InvalidIndexFormat("linklist already initialized".to_string()))?;
    }

    if !rd.is_empty() {
      return Err(Error::InvalidIndexFormat(
        "Index seems to be corrupted or unsupported".to_string(),
      ));
    }

    let mut num_deleted = 0usize;
    {
      let mut deleted_elements = idx.deleted_elements.lock();
      for i in 0..cur_element_count {
        if idx.is_marked_deleted_internal(i as TableInt) {
          num_deleted += 1;
          if idx.allow_replace_deleted {
            deleted_elements.insert(i as TableInt);
          }
        }
      }
    }
    idx.num_deleted.store(num_deleted, Ordering::Release);

    Ok(idx)
  }

  pub fn check_integrity(&self) -> Result<()> {
    let _mutation_guard = self.mutation_lock.write();

    let cur_element_count = self.cur_element_count.load(Ordering::Acquire);
    let mut inbound: Vec<usize> = vec![0; cur_element_count];
    for i in 0..cur_element_count {
      let max_level = self.element_levels[i].load(Ordering::Acquire);
      if max_level < 0 {
        return Err(Error::InvalidIndexFormat(
          "negative element level".to_string(),
        ));
      }
      let max_level = max_level as usize;
      for level in 0..=max_level {
        let ll = self.linklist_at_level(i as TableInt, level)?;
        let mut s = HashSet::new();
        for to in ll {
          if to as usize >= cur_element_count {
            return Err(Error::InvalidIndexFormat("bad neighbor id".to_string()));
          }
          if to as usize == i {
            return Err(Error::InvalidIndexFormat("self loop".to_string()));
          }
          inbound[to as usize] += 1;
          s.insert(to);
        }
        if s.len() != ll.len() {
          return Err(Error::InvalidIndexFormat("duplicate edge".to_string()));
        }
      }
    }

    if cur_element_count > 1 {
      for (i, &n) in inbound.iter().enumerate() {
        if n == 0 {
          return Err(Error::InvalidIndexFormat(format!(
            "node {i} has zero inbound connections"
          )));
        }
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::space::InnerProductSpace;
  use crate::space::L2Space;
  use crate::stop_condition::EpsilonSearchStopCondition;
  use crate::stop_condition::MultiVectorSearchStopCondition;
  use crate::view::HnswIndexView;
  use approx::assert_relative_eq;
  use proptest::prelude::*;
  use rand::rngs::StdRng;
  use rand::Rng;
  use rand::SeedableRng;

  fn brute_force_knn<S: Space>(
    space: &S,
    points: &[(LabelType, Vec<f32>)],
    query: &[f32],
    k: usize,
  ) -> Vec<(LabelType, f32)> {
    let mut all: Vec<(LabelType, f32)> = points
      .iter()
      .map(|(l, v)| (*l, space.distance(query, v)))
      .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    all.truncate(k);
    all
  }

  #[test]
  fn delete_mark_is_preserved_when_setting_count() {
    let mut header = DELETE_MARK | 7;
    set_linklist_count(&mut header, 123);
    assert_eq!(header & DELETE_MARK, DELETE_MARK);
    assert_eq!(linklist_count(header), 123);
  }

  #[test]
  fn mark_delete_and_unmark_delete_affects_search_results() {
    let space = L2Space::new(2);
    let idx = HnswIndex::new(space, 10, 8, 64, 42, false);
    idx.add_point(&[0.0, 0.0], 1).unwrap();
    idx.add_point(&[10.0, 10.0], 2).unwrap();
    idx.set_ef(10);

    let res = idx.search_knn(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(res[0].0, 1);

    idx.mark_delete(1).unwrap();
    let res = idx.search_knn(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(res[0].0, 2);

    idx.unmark_delete(1).unwrap();
    let res = idx.search_knn(&[0.0, 0.0], 2, None).unwrap();
    assert_eq!(res[0].0, 1);
  }

  #[test]
  fn replace_deleted_reuses_internal_slot() {
    let space = L2Space::new(2);
    let idx = HnswIndex::new(space, 10, 8, 64, 42, true);
    let id1 = idx.add_point(&[0.0, 0.0], 1).unwrap();
    let _id2 = idx.add_point(&[10.0, 10.0], 2).unwrap();
    idx.mark_delete(1).unwrap();
    assert_eq!(idx.get_deleted_count(), 1);

    let id3 = idx.add_point_replace_deleted(&[1.0, 1.0], 3).unwrap();
    assert_eq!(id3, id1);
    assert_eq!(idx.get_deleted_count(), 0);
    assert!(idx.get_data_by_label(3).is_ok());
    assert!(matches!(idx.get_data_by_label(1), Err(Error::LabelNotFound(1))));
  }

  #[test]
  fn update_existing_label_updates_vector() {
    let space = L2Space::new(2);
    let idx = HnswIndex::new(space, 10, 8, 64, 42, false);
    idx.add_point(&[0.0, 0.0], 1).unwrap();
    idx.add_point(&[100.0, 100.0], 1).unwrap();
    let v = idx.get_data_by_label(1).unwrap();
    assert_relative_eq!(v[0], 100.0);
    assert_relative_eq!(v[1], 100.0);
  }

  #[test]
  fn save_load_roundtrip_is_byte_identical() {
    let space = L2Space::new(4);
    let idx = HnswIndex::new(space, 100, 16, 200, 123, true);

    for i in 0..50 {
      let v = [i as f32, 1.0, 2.0, 3.0];
      idx.add_point_at_level(&v, i as LabelType, Some(0)).unwrap();
    }
    idx.mark_delete(10).unwrap();
    idx.mark_delete(20).unwrap();

    let bytes1 = idx.save_to_vec().unwrap();
    let idx2 =
      HnswIndex::load_from_bytes_with_options(L2Space::new(4), &bytes1, None, 123, true).unwrap();
    let bytes2 = idx2.save_to_vec().unwrap();
    assert_eq!(bytes1, bytes2);
  }

  #[test]
  fn view_can_parse_rust_saved_index() {
    let space = InnerProductSpace::new(3);
    let idx = HnswIndex::new(space, 10, 8, 64, 42, false);
    idx.add_point(&[1.0, 0.0, 0.0], 1).unwrap();
    idx.add_point(&[0.0, 1.0, 0.0], 2).unwrap();
    idx.add_point(&[0.0, 0.0, 1.0], 3).unwrap();
    let bytes = idx.save_to_vec().unwrap();

    let view = HnswIndexView::load(3, &bytes).unwrap();
    assert_eq!(view.cur_element_count, 3);
    assert_eq!(view.m, 8);
    assert!(view.has_label(1));
    assert!(view.has_label(2));
    assert!(view.has_label(3));
  }

  #[test]
  fn exact_knn_for_small_sets_with_high_params_level0_only() {
    let dim = 8;
    let n = 64;
    let k = 5;
    let mut rng = StdRng::seed_from_u64(7);
    let space = L2Space::new(dim);

    let idx = HnswIndex::new(space.clone(), n, n, n, 7, false);
    idx.set_ef(n);

    let mut points: Vec<(LabelType, Vec<f32>)> = Vec::new();
    for label in 0..n {
      let mut v = vec![0.0_f32; dim];
      for x in &mut v {
        *x = rng.gen_range(-1.0..1.0);
      }
      idx
        .add_point_at_level(&v, label as LabelType, Some(0))
        .unwrap();
      points.push((label as LabelType, v));
    }

    for _ in 0..20 {
      let mut q = vec![0.0_f32; dim];
      for x in &mut q {
        *x = rng.gen_range(-1.0..1.0);
      }
      let brute = brute_force_knn(&space, &points, &q, k);
      let got = idx.search_knn(&q, k, None).unwrap();
      let mut got_sorted = got;
      got_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
      assert_eq!(got_sorted, brute);
    }
  }

  #[test]
  fn epsilon_stop_condition_filters_results() {
    let space = L2Space::new(1);
    let idx = HnswIndex::new(space, 10, 8, 64, 42, false);
    for (label, x) in [(0, 0.0), (1, 0.5), (2, 2.0), (3, 10.0)] {
      idx.add_point(&[x], label).unwrap();
    }

    let mut stop = EpsilonSearchStopCondition::new(1.0, 1, 10);
    let res = idx
      .search_stop_condition_closest(&[0.0], &mut stop, None)
      .unwrap();
    assert!(!res.is_empty());
    assert!(res.iter().all(|(_, d)| *d <= 1.0));
  }

  #[test]
  fn multivector_stop_condition_limits_distinct_doc_ids() {
    let space = L2Space::new(1);
    let idx = HnswIndex::new(space, 32, 16, 64, 42, false);
    // Three docs, 3 vectors each, increasing distance from query.
    for label in 0..9usize {
      idx
        .add_point_at_level(&[label as f32], label as LabelType, Some(0))
        .unwrap();
    }

    let mut stop =
      MultiVectorSearchStopCondition::new(|label: LabelType, _dp: &[f32]| label / 3, 2, 3);
    let res = idx
      .search_stop_condition_closest(&[0.0], &mut stop, None)
      .unwrap();

    let distinct_docs = res.iter().map(|(l, _)| l / 3).collect::<HashSet<_>>();
    assert!(distinct_docs.len() <= 2);
  }

  #[test]
  fn parallel_add_point_is_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let dim = 4;
    let n = 256;
    let threads = 8;
    let space = L2Space::new(dim);
    let idx = Arc::new(HnswIndex::new(space, n, 16, 200, 42, false));

    let mut handles = Vec::new();
    for t in 0..threads {
      let idx = idx.clone();
      handles.push(thread::spawn(move || {
        for label in (t..n).step_by(threads) {
          let v = [
            label as f32,
            (label as f32) * 0.25,
            (label as f32) * -0.5,
            1.0,
          ];
          idx.add_point(&v, label).unwrap();
        }
      }));
    }
    for h in handles {
      h.join().unwrap();
    }

    assert_eq!(idx.get_current_element_count(), n);
    idx.set_ef(n);

    for label in [0usize, 1, 2, 17, 63, 128, 255] {
      let v = [
        label as f32,
        (label as f32) * 0.25,
        (label as f32) * -0.5,
        1.0,
      ];
      let got = idx.get_data_by_label(label).unwrap();
      assert_eq!(got.as_slice(), &v);

      let knn = idx.search_knn(&v, 1, None).unwrap();
      assert_eq!(knn[0].0, label);
      assert_relative_eq!(knn[0].1, 0.0);
    }

    idx.check_integrity().unwrap();
  }

  #[test]
  fn parallel_mark_delete_is_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let dim = 2;
    let n = 128;
    let threads = 8;
    let space = L2Space::new(dim);
    let idx = Arc::new(HnswIndex::new(space, n, 16, 200, 42, false));

    for label in 0..n {
      let v = [label as f32, 0.0];
      idx.add_point(&v, label).unwrap();
    }
    idx.set_ef(n);

    let mut handles = Vec::new();
    for t in 0..threads {
      let idx = idx.clone();
      handles.push(thread::spawn(move || {
        for label in (t..n).step_by(threads) {
          if label % 2 == 0 {
            idx.mark_delete(label).unwrap();
          }
        }
      }));
    }
    for h in handles {
      h.join().unwrap();
    }

    assert_eq!(idx.get_deleted_count(), n / 2);

    for label in 0..n {
      let v = [label as f32, 0.0];
      if label % 2 == 0 {
        assert!(matches!(
          idx.get_data_by_label(label),
          Err(Error::LabelNotFound(l)) if l == label
        ));
        let got = idx.search_knn(&v, 1, None).unwrap();
        assert_ne!(got[0].0, label);
      } else {
        assert!(idx.get_data_by_label(label).is_ok());
        let got = idx.search_knn(&v, 1, None).unwrap();
        assert_eq!(got[0].0, label);
      }
    }

    idx.check_integrity().unwrap();
  }

  proptest! {
    #[test]
    fn prop_exact_knn_with_level0_only(
      dim in 2usize..12,
      n in 2usize..64,
      k in 1usize..8,
      seed in any::<u64>(),
    ) {
      let k = k.min(n);
      let mut rng = StdRng::seed_from_u64(seed);
      let space = L2Space::new(dim);

      let idx = HnswIndex::new(space.clone(), n, n, n, seed, false);
      idx.set_ef(n);

      let mut points: Vec<(LabelType, Vec<f32>)> = Vec::with_capacity(n);
      for label in 0..n {
        let mut v = vec![0.0_f32; dim];
        for x in &mut v {
          *x = rng.gen_range(-1.0..1.0);
        }
        idx.add_point_at_level(&v, label as LabelType, Some(0)).unwrap();
        points.push((label as LabelType, v));
      }

      let mut query = vec![0.0_f32; dim];
      for x in &mut query {
        *x = rng.gen_range(-1.0..1.0);
      }

      let brute = brute_force_knn(&space, &points, &query, k);
      let mut got = idx.search_knn(&query, k, None).unwrap();
      got.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0))
      });
      prop_assert_eq!(got, brute);
    }
  }
}
