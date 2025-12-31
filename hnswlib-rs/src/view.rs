use crate::error::Error;
use crate::error::Result;
use crate::space::label_allowed;
use crate::LabelType;
use crate::LinkListSizeInt;
use crate::Space;
use crate::TableInt;
use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use ordered_float::OrderedFloat;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::ops::Range;

fn consume<const N: usize>(rd: &mut &[u8]) -> Result<[u8; N]> {
  if rd.len() < N {
    return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
  }
  let (bytes, rest) = rd.split_at(N);
  *rd = rest;
  Ok(bytes.try_into().unwrap())
}

fn consume_usize(rd: &mut &[u8]) -> Result<usize> {
  let bytes = consume::<{ size_of::<usize>() }>(rd)?;
  Ok(usize::from_le_bytes(bytes))
}

fn consume_u32(rd: &mut &[u8]) -> Result<u32> {
  Ok(u32::from_le_bytes(consume::<4>(rd)?))
}

fn consume_i32(rd: &mut &[u8]) -> Result<i32> {
  Ok(i32::from_le_bytes(consume::<4>(rd)?))
}

fn consume_f64(rd: &mut &[u8]) -> Result<f64> {
  Ok(f64::from_le_bytes(consume::<8>(rd)?))
}

fn consume_bytes<'a>(rd: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
  if rd.len() < n {
    return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
  }
  let (bytes, rest) = rd.split_at(n);
  *rd = rest;
  Ok(bytes)
}

fn get_external_label(
  data_level_0_memory: &[u8],
  internal_id: TableInt,
  size_data_per_element: usize,
  label_offset: usize,
) -> Result<LabelType> {
  let start = internal_id as usize * size_data_per_element + label_offset;
  let end = start + size_of::<LabelType>();
  if end > data_level_0_memory.len() {
    return Err(Error::InvalidIndexFormat(
      "label_offset out of bounds".to_string(),
    ));
  }
  Ok(LabelType::from_le_bytes(
    data_level_0_memory[start..end].try_into().unwrap(),
  ))
}

#[derive(Debug)]
pub struct HnswIndexView<'a> {
  pub dim: usize,

  pub max_elements: usize,
  pub cur_element_count: usize,
  pub size_data_per_element: usize,
  pub size_links_per_element: usize,
  pub num_deleted: usize,
  pub m: usize,
  pub max_m: usize,
  pub max_m_0: usize,
  pub ef_construction: usize,
  pub ef: usize,

  pub mult: f64,
  pub rev_size: f64,
  pub max_level: i32,

  pub enter_point_node: Option<TableInt>,

  pub size_links_level_0: usize,
  pub offset_data: usize,
  pub offset_level_0: usize,
  pub label_offset: usize,

  pub data_level_0_memory: &'a [u8],
  pub link_lists: Vec<Option<&'a [u8]>>,
  pub element_levels: Vec<usize>,

  pub label_lookup: HashMap<LabelType, TableInt>,
}

impl<'a> HnswIndexView<'a> {
  pub fn load(dim: usize, data: &'a [u8]) -> Result<Self> {
    let rd = &mut &*data;

    let offset_level_0 = consume_usize(rd)?;
    let max_elements = consume_usize(rd)?;
    let cur_element_count = consume_usize(rd)?;
    let size_data_per_element = consume_usize(rd)?;
    let label_offset = consume_usize(rd)?;
    let offset_data = consume_usize(rd)?;
    let max_level = consume_i32(rd)?;
    let enter_point_node_raw: TableInt = consume_u32(rd)?;
    let max_m = consume_usize(rd)?;
    let max_m_0 = consume_usize(rd)?;
    let m = consume_usize(rd)?;
    let mult = consume_f64(rd)?;
    let ef_construction = consume_usize(rd)?;

    if offset_level_0 != 0 {
      return Err(Error::InvalidIndexFormat(format!(
        "unsupported offset_level_0={offset_level_0}"
      )));
    }
    if cur_element_count > max_elements {
      return Err(Error::InvalidIndexFormat(
        "cur_element_count > max_elements".to_string(),
      ));
    }

    let data_level_0_memory = consume_bytes(rd, cur_element_count * size_data_per_element)?;

    let size_links_per_element = max_m * size_of::<TableInt>() + size_of::<LinkListSizeInt>();
    let size_links_level_0 = max_m_0 * size_of::<TableInt>() + size_of::<LinkListSizeInt>();

    let mut label_lookup = HashMap::<LabelType, TableInt>::new();
    let mut element_levels = vec![0; max_elements];
    let mut link_lists = vec![None; max_elements];

    let enter_point_node = (enter_point_node_raw != TableInt::MAX).then_some(enter_point_node_raw);

    let mut num_deleted = 0usize;
    for i in 0..cur_element_count {
      let external_label = get_external_label(
        data_level_0_memory,
        i as TableInt,
        size_data_per_element,
        label_offset,
      )?;
      if label_lookup
        .insert(external_label, i.try_into().unwrap())
        .is_some()
      {
        return Err(Error::InvalidIndexFormat(
          "duplicate external label".to_string(),
        ));
      }

      if Self::is_marked_deleted_raw(
        data_level_0_memory,
        i as TableInt,
        size_data_per_element,
        offset_level_0,
      )? {
        num_deleted += 1;
      }

      let link_list_size: usize = consume_u32(rd)?.try_into().unwrap();
      if link_list_size > 0 {
        if link_list_size % size_links_per_element != 0 {
          return Err(Error::InvalidIndexFormat(
            "invalid linkListSize".to_string(),
          ));
        }
        element_levels[i] = link_list_size / size_links_per_element;
        let link_list = consume_bytes(rd, link_list_size)?;
        link_lists[i] = Some(link_list);
      };
    }

    if !rd.is_empty() {
      return Err(Error::InvalidIndexFormat(
        "trailing bytes after index".to_string(),
      ));
    }

    let rev_size = 1.0 / mult;
    let ef = 10;

    Ok(Self {
      dim,
      max_elements,
      cur_element_count,
      size_data_per_element,
      size_links_per_element,
      num_deleted,
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
    })
  }

  pub fn entry_label(&self) -> Result<LabelType> {
    let ep = self.enter_point_node.ok_or(Error::EmptyIndex)?;
    self.get_external_label(ep)
  }

  fn get_internal_id(&self, label: LabelType) -> Result<TableInt> {
    self
      .label_lookup
      .get(&label)
      .copied()
      .ok_or(Error::LabelNotFound(label))
  }

  fn get_external_label(&self, internal_id: TableInt) -> Result<LabelType> {
    get_external_label(
      self.data_level_0_memory,
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
      .map(|internal_id| self.get_external_label(internal_id).expect("invalid index"))
  }

  fn get_raw_data_by_internal_id(&self, internal_id: TableInt) -> Result<&[u8]> {
    let internal_id: usize = internal_id.try_into().unwrap();
    let pos = internal_id * self.size_data_per_element + self.offset_data;
    let end = pos + self.dim * size_of::<f32>();
    if end > self.data_level_0_memory.len() {
      return Err(Error::InvalidIndexFormat(
        "data offset out of bounds".to_string(),
      ));
    }
    Ok(&self.data_level_0_memory[pos..end])
  }

  pub fn get_raw_data_by_label(&self, label: LabelType) -> Result<&[u8]> {
    self.get_raw_data_by_internal_id(self.get_internal_id(label)?)
  }

  fn get_data_by_internal_id(&self, internal_id: TableInt) -> Result<&[f32]> {
    let bytes = self.get_raw_data_by_internal_id(internal_id)?;
    bytemuck::try_cast_slice(bytes)
      .map_err(|_| Error::InvalidIndexFormat("unaligned f32 vector".to_string()))
  }

  pub fn get_data_by_label(&self, label: LabelType) -> Result<&[f32]> {
    self.get_data_by_internal_id(self.get_internal_id(label)?)
  }

  fn get_link_list(&self, internal_id: TableInt, level: usize) -> Result<&[TableInt]> {
    let ptr = if level == 0 {
      let start = internal_id as usize * self.size_data_per_element + self.offset_level_0;
      let end = start + self.size_links_level_0;
      if end > self.data_level_0_memory.len() {
        return Err(Error::InvalidIndexFormat(
          "linklist0 out of bounds".to_string(),
        ));
      }
      &self.data_level_0_memory[start..end]
    } else {
      let raw = self.link_lists[internal_id as usize]
        .as_ref()
        .ok_or_else(|| Error::InvalidIndexFormat("missing linklist".to_string()))?;
      let start = (level - 1) * self.size_links_per_element;
      let end = start + self.size_links_per_element;
      if end > raw.len() {
        return Err(Error::InvalidIndexFormat(
          "linklist level out of bounds".to_string(),
        ));
      }
      &raw[start..end]
    };
    let cnt = u16::from_le_bytes(ptr[..2].try_into().unwrap()) as usize;
    if level == 0 && cnt > self.max_m_0 {
      return Err(Error::InvalidIndexFormat("linklist0 too large".to_string()));
    }
    if level > 0 && cnt > self.max_m {
      return Err(Error::InvalidIndexFormat("linklist too large".to_string()));
    }
    let neighbors_bytes = &ptr[4..];
    let neighbors: &[TableInt] = bytemuck::try_cast_slice(neighbors_bytes)
      .map_err(|_| Error::InvalidIndexFormat("unaligned neighbor list".to_string()))?;
    Ok(&neighbors[..cnt])
  }

  pub fn get_level_neighbors(
    &self,
    label: LabelType,
    level: usize,
  ) -> Result<impl Iterator<Item = LabelType> + '_> {
    let internal_id = self.get_internal_id(label)?;
    let neighbors = self.get_link_list(internal_id, level)?;
    Ok(
      neighbors
        .iter()
        .map(|&internal_id| self.get_external_label(internal_id).expect("invalid index")),
    )
  }

  pub fn get_merged_neighbors(
    &self,
    label: LabelType,
    min_level: usize,
  ) -> Result<HashSet<LabelType>> {
    let internal_id = self.get_internal_id(label)?;
    let mut out = HashSet::new();
    for level in min_level..=self.get_node_level(label)? {
      let list = self.get_link_list(internal_id, level)?;
      for &internal_id in list {
        let id = self.get_external_label(internal_id)?;
        debug_assert!(self.get_node_level(id)? >= min_level);
        out.insert(id);
      }
    }
    Ok(out)
  }

  pub fn get_node_level(&self, label: LabelType) -> Result<usize> {
    let internal_id = self.get_internal_id(label)?;
    Ok(self.element_levels[internal_id as usize])
  }

  fn is_marked_deleted_raw(
    data_level_0_memory: &[u8],
    internal_id: TableInt,
    size_data_per_element: usize,
    offset_level_0: usize,
  ) -> Result<bool> {
    let start = internal_id as usize * size_data_per_element + offset_level_0;
    let end = start + 4;
    if end > data_level_0_memory.len() {
      return Err(Error::InvalidIndexFormat(
        "linklist header out of bounds".to_string(),
      ));
    }
    Ok((data_level_0_memory[start + 2] & 0x01) != 0)
  }

  pub fn is_marked_deleted(&self, label: LabelType) -> Result<bool> {
    let internal_id = self.get_internal_id(label)?;
    Self::is_marked_deleted_raw(
      self.data_level_0_memory,
      internal_id,
      self.size_data_per_element,
      self.offset_level_0,
    )
  }

  fn search_base_layer_st<S: Space>(
    &self,
    ep_id: TableInt,
    query: &[f32],
    space: &S,
    ef: usize,
    filter: Option<&dyn Fn(LabelType) -> bool>,
    bare_bone_search: bool,
  ) -> Result<BinaryHeap<(OrderedFloat<f32>, TableInt)>> {
    let mut visited = vec![0u16; self.cur_element_count.max(1)];
    let visited_tag = 1u16;

    let mut top_candidates: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();
    let mut candidate_set: BinaryHeap<(OrderedFloat<f32>, TableInt)> = BinaryHeap::new();

    let mut lower_bound;
    let ep_label = self.get_external_label(ep_id)?;
    if bare_bone_search || (!self.is_marked_deleted(ep_label)? && label_allowed(filter, ep_label)) {
      let ep_data = self.get_data_by_internal_id(ep_id)?;
      let dist = space.distance(query, ep_data);
      lower_bound = dist;
      top_candidates.push((OrderedFloat(dist), ep_id));
      candidate_set.push((OrderedFloat(-dist), ep_id));
    } else {
      lower_bound = f32::INFINITY;
      candidate_set.push((OrderedFloat(-lower_bound), ep_id));
    }

    visited[ep_id as usize] = visited_tag;

    while let Some((neg_dist, current_node_id)) = candidate_set.pop() {
      let candidate_dist = -neg_dist.0;
      let flag_stop_search = if bare_bone_search {
        candidate_dist > lower_bound
      } else {
        candidate_dist > lower_bound && top_candidates.len() == ef
      };
      if flag_stop_search {
        break;
      }

      let data = self.get_link_list(current_node_id, 0)?;
      for &candidate_id in data {
        if visited[candidate_id as usize] == visited_tag {
          continue;
        }
        visited[candidate_id as usize] = visited_tag;

        let curr = self.get_data_by_internal_id(candidate_id)?;
        let dist = space.distance(query, curr);
        let flag_consider_candidate = top_candidates.len() < ef || lower_bound > dist;
        if !flag_consider_candidate {
          continue;
        }

        candidate_set.push((OrderedFloat(-dist), candidate_id));

        let cand_label = self.get_external_label(candidate_id)?;
        if bare_bone_search
          || (!self.is_marked_deleted(cand_label)? && label_allowed(filter, cand_label))
        {
          top_candidates.push((OrderedFloat(dist), candidate_id));
        }

        while top_candidates.len() > ef {
          top_candidates.pop();
        }
        if let Some((worst_dist, _)) = top_candidates.peek() {
          lower_bound = worst_dist.0;
        }
      }
    }

    Ok(top_candidates)
  }

  pub fn search_knn<S: Space>(
    &self,
    query: &[f32],
    k: usize,
    space: &S,
    filter: Option<&dyn Fn(LabelType) -> bool>,
  ) -> Result<Vec<(LabelType, f32)>> {
    if space.dim() != self.dim {
      return Err(Error::DimensionMismatch {
        expected: self.dim,
        actual: space.dim(),
      });
    }
    if query.len() != self.dim {
      return Err(Error::DimensionMismatch {
        expected: self.dim,
        actual: query.len(),
      });
    }
    if self.cur_element_count == 0 {
      return Ok(Vec::new());
    }

    let mut curr_obj = self.enter_point_node.ok_or(Error::EmptyIndex)?;
    let mut cur_dist = space.distance(query, self.get_data_by_internal_id(curr_obj)?);

    for level in (1..=self.max_level.max(0) as usize).rev() {
      let mut changed = true;
      while changed {
        changed = false;

        let data = self.get_link_list(curr_obj, level)?;
        for &cand in data {
          let d = space.distance(query, self.get_data_by_internal_id(cand)?);
          if d < cur_dist {
            cur_dist = d;
            curr_obj = cand;
            changed = true;
          }
        }
      }
    }

    let bare_bone_search = self.num_deleted == 0 && filter.is_none();
    let mut top_candidates = self.search_base_layer_st(
      curr_obj,
      query,
      space,
      max(self.ef, k),
      filter,
      bare_bone_search,
    )?;

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
}
