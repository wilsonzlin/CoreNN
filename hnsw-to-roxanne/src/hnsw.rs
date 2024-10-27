use ahash::HashMap;
use ahash::HashMapExt;
use ahash::HashSet;
use ahash::HashSetExt;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use std::io;
use std::io::Read;
use std::mem::size_of;

// These names are copied verbatem (except for casing) from the original source.
pub type TableInt = u32; // This is the internal ID.
pub type LinkListSizeInt = u32;
pub type LabelType = usize; // This is the external ID.

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
  pub max_level: usize, // This is stored as i32 in source, but we store as usize for convenience.

  pub enter_point_node: TableInt,

  pub size_links_level_0: usize,
  pub offset_data: usize,
  pub offset_level_0: usize,
  pub label_offset: usize,

  pub data_level_0_memory: Vec<u8>,
  pub link_lists: Vec<Option<Vec<u8>>>,
  pub element_levels: Vec<usize>, // These are stored as i32 in source, but we store as usize for convenience.

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
    rd.read_exact(&mut data_level_0_memory).unwrap();

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

  pub fn get_raw_data_by_internal_id(&self, internal_id: TableInt) -> &[u8] {
    let internal_id: usize = internal_id.try_into().unwrap();
    let pos = internal_id * self.size_data_per_element + self.offset_data;
    &self.data_level_0_memory[pos..]
  }

  pub fn get_data_by_internal_id(&self, internal_id: TableInt) -> Vec<f32> {
    let ptr = self.get_raw_data_by_internal_id(internal_id);
    let mut vec = vec![0.0f32; self.dim];
    LittleEndian::read_f32_into(&ptr[..4 * self.dim], &mut vec);
    vec
  }

  pub fn get_data_by_label(&self, label: LabelType) -> Vec<f32> {
    let internal_id = self.get_internal_id(label);
    self.get_data_by_internal_id(internal_id)
  }

  fn get_link_list(&self, internal_id: TableInt, level: usize) -> Vec<TableInt> {
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
    let mut lst = vec![0; cnt];
    LittleEndian::read_u32_into(&ptr[4..4 + cnt * size_of::<TableInt>()], &mut lst);
    lst
  }

  // Provide `min_level` > 0 to filter out all nodes at lower levels and edges to them.
  pub fn get_merged_neighbors(&self, label: LabelType, min_level: usize) -> HashSet<LabelType> {
    let internal_id = self.get_internal_id(label);
    let mut out = HashSet::new();
    // Source: searchKnn and searchBaseLayerST (called by searchKnn).
    // Bound to node's own level to avoid out-of-bounds access in get_link_list, which the source doesn't do for some reason: https://github.com/nmslib/hnswlib/issues/595.
    for level in min_level..=self.get_node_level(label) {
      let list = self.get_link_list(internal_id, level);
      for internal_id in list {
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

  pub fn entry_label(&self) -> LabelType {
    self.get_external_label(self.enter_point_node)
  }

  pub fn get_internal_id(&self, label: LabelType) -> TableInt {
    *self.label_lookup.get(&label).unwrap()
  }

  pub fn get_external_label(&self, internal_id: TableInt) -> LabelType {
    get_external_label(
      &self.data_level_0_memory,
      internal_id,
      self.size_data_per_element,
      self.label_offset,
    )
  }

  pub fn internal_ids(&self) -> std::ops::Range<TableInt> {
    0..TableInt::try_from(self.cur_element_count).unwrap()
  }

  pub fn labels(&self) -> impl Iterator<Item = LabelType> + '_ {
    self
      .internal_ids()
      .map(|internal_id| self.get_external_label(internal_id))
  }
}
