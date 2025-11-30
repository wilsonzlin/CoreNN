use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use bytemuck::cast_slice;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use ordered_float::OrderedFloat;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::io;
use std::io::Read;
use std::mem::size_of;
use std::ops::Range;

// These names are copied verbatem (except for casing) from the original source.
pub type TableInt = u32; // This is the internal ID.
pub type LinkListSizeInt = u32;
pub type LabelType = usize; // This is the external ID.

pub type Metric = fn(&[f32], &[f32]) -> f64;

// Corresponds to `std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>` in hnswalg.h.
#[derive(Default)]
struct MaxHeap(BinaryHeap<(OrderedFloat<f64>, TableInt)>);

impl MaxHeap {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn len(&self) -> usize {
    self.0.len()
  }

  pub fn push(&mut self, dist: f64, id: TableInt) {
    self.0.push((OrderedFloat(dist), id));
  }

  pub fn pop(&mut self) -> Option<(f64, TableInt)> {
    self.0.pop().map(|e| (e.0 .0, e.1))
  }

  pub fn peek(&self) -> Option<(f64, TableInt)> {
    self.0.peek().map(|e| (e.0 .0, e.1))
  }
}

trait ReadExt {
  fn read_usize(&mut self) -> io::Result<usize>;
}

impl<T: Read> ReadExt for T {
  fn read_usize(&mut self) -> io::Result<usize> {
    let mut buf = vec![0u8; size_of::<usize>()];
    self.read_exact(&mut buf)?;
    Ok(usize::from_le_bytes(buf.try_into().unwrap()))
  }
}

fn get_external_label(
  data_level_0_memory: &[u8],
  internal_id: TableInt,
  size_data_per_element: usize,
  label_offset: usize,
) -> LabelType {
  let mut ptr = &data_level_0_memory[internal_id as usize * size_data_per_element + label_offset..];
  let return_label: LabelType = ptr.read_usize().unwrap();
  return_label
}

#[allow(unused)]
#[derive(Debug)]
pub struct HnswIndex {
  pub dim: usize,

  pub max_elements: usize,
  pub cur_element_count: usize,
  pub size_data_per_element: usize,
  pub size_links_per_element: usize,
  pub m: usize,
  pub max_m: usize,
  pub max_m_0: usize,
  pub ef_construction: usize,
  pub ef: usize,

  pub mult: f64,
  pub rev_size: f64,
  pub max_level: usize, // This is stored as i32 in the original hnswlib, but we store as usize for convenience.

  pub enter_point_node: TableInt,

  pub size_links_level_0: usize,
  pub offset_data: usize,
  pub offset_level_0: usize,
  pub label_offset: usize,

  pub data_level_0_memory: Vec<u8>,
  pub link_lists: Vec<Option<Vec<u8>>>,
  pub element_levels: Vec<usize>, // These are stored as i32 in the original hnswlib, but we store as usize for convenience.

  pub label_lookup: HashMap<LabelType, TableInt>,
}

impl HnswIndex {
  pub fn load(dim: usize, mut rd: impl Read) -> Self {
    let offset_level_0 = rd.read_usize().unwrap();
    let max_elements = rd.read_usize().unwrap();
    let cur_element_count = rd.read_usize().unwrap();
    let size_data_per_element = rd.read_usize().unwrap();
    let label_offset = rd.read_usize().unwrap();
    let offset_data = rd.read_usize().unwrap();
    let max_level = rd.read_i32::<LittleEndian>().unwrap().try_into().unwrap();
    let enter_point_node: TableInt = rd.read_u32::<LittleEndian>().unwrap();
    let max_m = rd.read_usize().unwrap();
    let max_m_0 = rd.read_usize().unwrap();
    let m = rd.read_usize().unwrap();
    let mult = rd.read_f64::<LittleEndian>().unwrap();
    let ef_construction = rd.read_usize().unwrap();

    let mut data_level_0_memory = vec![0u8; max_elements * size_data_per_element];
    rd.read_exact(&mut data_level_0_memory[..cur_element_count * size_data_per_element]).unwrap();

    let size_links_per_element = max_m * size_of::<TableInt>() + size_of::<LinkListSizeInt>();
    let size_links_level_0 = max_m_0 * size_of::<TableInt>() + size_of::<LinkListSizeInt>();

    let mut label_lookup = HashMap::<LabelType, TableInt>::new();
    let mut element_levels = vec![0; max_elements];
    let mut link_lists = vec![None; max_elements];
    let rev_size = 1.0 / mult;
    let ef = 10;
    for i in 0..cur_element_count {
      let external_label = get_external_label(
        &data_level_0_memory,
        i as TableInt,
        size_data_per_element,
        label_offset,
      );
      assert!(label_lookup
        .insert(external_label, i.try_into().unwrap())
        .is_none());
      let link_list_size: usize = rd.read_u32::<LittleEndian>().unwrap().try_into().unwrap();
      if link_list_size > 0 {
        element_levels[i] = link_list_size / size_links_per_element;
        let mut link_list = vec![0u8; link_list_size];
        rd.read_exact(&mut link_list).unwrap();
        link_lists[i] = Some(link_list);
      };
    }

    let mut eof = Vec::new();
    rd.read_to_end(&mut eof).unwrap();
    assert_eq!(eof, vec![]);

    Self {
      dim,
      max_elements,
      cur_element_count,
      size_data_per_element,
      size_links_per_element,
      m,
      max_m,
      max_m_0,
      ef_construction,
      ef,
      mult,
      rev_size,
      max_level,
      enter_point_node,
      size_links_level_0,
      offset_data,
      offset_level_0,
      label_offset,
      data_level_0_memory,
      link_lists,
      element_levels,
      label_lookup,
    }
  }

  pub fn entry_label(&self) -> LabelType {
    self.get_external_label(self.enter_point_node)
  }

  fn get_internal_id(&self, label: LabelType) -> TableInt {
    *self.label_lookup.get(&label).unwrap()
  }

  fn get_external_label(&self, internal_id: TableInt) -> LabelType {
    get_external_label(
      &self.data_level_0_memory,
      internal_id,
      self.size_data_per_element,
      self.label_offset,
    )
  }

  pub fn has_label(&self, label: LabelType) -> bool {
    self.label_lookup.contains_key(&label)
  }

  pub fn internal_ids(&self) -> Range<TableInt> {
    0..TableInt::try_from(self.cur_element_count).unwrap()
  }

  pub fn labels(&self) -> impl Iterator<Item = LabelType> + '_ {
    self
      .internal_ids()
      .map(|internal_id| self.get_external_label(internal_id))
  }

  fn get_raw_data_by_internal_id(&self, internal_id: TableInt) -> &[u8] {
    let internal_id: usize = internal_id.try_into().unwrap();
    let pos = internal_id * self.size_data_per_element + self.offset_data;
    &self.data_level_0_memory[pos..]
  }

  pub fn get_raw_data_by_label(&self, label: LabelType) -> &[u8] {
    self.get_raw_data_by_internal_id(self.get_internal_id(label))
  }

  fn get_data_by_internal_id(&self, internal_id: TableInt) -> &[f32] {
    let ptr = self.get_raw_data_by_internal_id(internal_id);
    cast_slice(&ptr[..4 * self.dim])
  }

  pub fn get_data_by_label(&self, label: LabelType) -> &[f32] {
    self.get_data_by_internal_id(self.get_internal_id(label))
  }

  fn get_link_list(&self, internal_id: TableInt, level: usize) -> &[TableInt] {
    let ptr = if level == 0 {
      // Source: get_linklist0.
      &self.data_level_0_memory
        [internal_id as usize * self.size_data_per_element + self.offset_level_0..]
    } else {
      // Source: get_linklist.
      let raw = self.link_lists[internal_id as usize].as_ref().unwrap();
      &raw[(level - 1) * self.size_links_per_element..]
    };
    // Source: getListCount.
    let cnt = u16::from_le_bytes(ptr[..2].try_into().unwrap()) as usize;
    cast_slice(&ptr[4..4 + cnt * size_of::<TableInt>()])
  }

  pub fn get_level_neighbors(
    &self,
    label: LabelType,
    level: usize,
  ) -> impl Iterator<Item = LabelType> + '_ {
    let internal_id = self.get_internal_id(label);
    let neighbors = self.get_link_list(internal_id, level);
    neighbors
      .into_iter()
      .map(|&internal_id| self.get_external_label(internal_id))
  }

  // Provide `min_level` > 0 to filter out all nodes at lower levels and edges to them.
  pub fn get_merged_neighbors(&self, label: LabelType, min_level: usize) -> HashSet<LabelType> {
    let internal_id = self.get_internal_id(label);
    let mut out = HashSet::new();
    // Source: searchKnn and searchBaseLayerST (called by searchKnn).
    for level in min_level..=self.get_node_level(label) {
      let list = self.get_link_list(internal_id, level);
      for &internal_id in list {
        let id = self.get_external_label(internal_id);
        // There should be no edges to lower level nodes as that should only be possible by looking at the out-neighbor list at a lower level.
        debug_assert!(self.get_node_level(id) >= min_level);
        out.insert(id);
      }
    }
    out
  }

  pub fn get_node_level(&self, label: LabelType) -> usize {
    let internal_id = self.get_internal_id(label);
    self.element_levels[internal_id as usize]
  }

  fn search_base_layer_st_bare_bone(
    &self,
    ep_id: TableInt,
    query: &[f32],
    metric: Metric,
    ef: usize,
  ) -> MaxHeap {
    let mut visited = HashSet::<TableInt>::new();

    let mut top_candidates = MaxHeap::new();
    let mut candidate_set = MaxHeap::new();

    let ep_data = self.get_data_by_internal_id(ep_id);
    let dist = metric(query, &ep_data);
    let mut lower_bound = dist;
    top_candidates.push(dist, ep_id);
    candidate_set.push(-dist, ep_id);

    visited.insert(ep_id);

    while let Some(p) = candidate_set.pop() {
      let candidate_dist = -p.0;
      if candidate_dist > lower_bound {
        break;
      }

      let current_node_id = p.1;
      let data = self.get_link_list(current_node_id, 0);
      for &candidate_id in data {
        if visited.insert(candidate_id) {
          let curr_obj_1 = self.get_data_by_internal_id(candidate_id);
          let dist = metric(query, &curr_obj_1);
          let flag_consider_candidate = top_candidates.len() < ef || lower_bound > dist;
          if flag_consider_candidate {
            candidate_set.push(-dist, candidate_id);
            top_candidates.push(dist, candidate_id);
            let mut flag_remove_extra = top_candidates.len() > ef;
            while flag_remove_extra {
              top_candidates.pop().unwrap();
              flag_remove_extra = top_candidates.len() > ef;
            }
            if let Some(p) = top_candidates.peek() {
              lower_bound = p.0;
            }
          }
        }
      }
    }

    top_candidates
  }

  pub fn search_knn(
    &self,
    query: &[f32],
    k: usize,
    metric: Metric,
  ) -> Vec<(LabelType, f64)> {
    let mut curr_obj = self.enter_point_node;
    let mut cur_dist = metric(query, &self.get_data_by_internal_id(curr_obj));

    for level in (1..=self.max_level).rev() {
      let mut changed = true;
      while changed {
        changed = false;

        let data = self.get_link_list(curr_obj, level);
        for &cand in data {
          let d = metric(query, self.get_data_by_internal_id(cand));

          if d < cur_dist {
            cur_dist = d;
            curr_obj = cand;
            changed = true;
          }
        }
      }
    }

    let mut top_candidates =
      self.search_base_layer_st_bare_bone(curr_obj, query, metric, max(self.ef, k));

    while top_candidates.len() > k {
      top_candidates.pop();
    }

    let mut res = Vec::new();
    while let Some((dist, id)) = top_candidates.pop() {
      let label = self.get_external_label(id);
      res.push((label, dist));
    }
    res.reverse();
    res
  }
}
