//! Load-only support for the original `hnswlib` on-disk format.
//!
//! The legacy format does not store the metric/space name, so callers must provide a `Metric`.

use crate::error::Error;
use crate::error::Result;
use crate::hnsw::Hnsw;
use crate::hnsw::HnswConfig;
use crate::id::NodeId;
use crate::metric::Metric;
use crate::vectors::VectorStore;
use std::mem::size_of;

fn consume<const N: usize>(rd: &mut &[u8]) -> Result<[u8; N]> {
  if rd.len() < N {
    return Err(Error::InvalidIndexFormat("unexpected EOF".to_string()));
  }
  let (bytes, rest) = rd.split_at(N);
  *rd = rest;
  Ok(bytes.try_into().unwrap())
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

fn consume_usize_width(rd: &mut &[u8], width: usize) -> Result<usize> {
  match width {
    8 => Ok(off64::usz!(u64::from_le_bytes(consume::<8>(rd)?))),
    4 => Ok(off64::usz!(u32::from_le_bytes(consume::<4>(rd)?))),
    _ => Err(Error::InvalidIndexFormat(
      "unsupported usize width".to_string(),
    )),
  }
}

#[derive(Debug, Clone)]
struct LegacyHeader {
  header_size: usize,
  offset_level0: usize,
  max_elements: usize,
  cur_element_count: usize,
  size_data_per_element: usize,
  label_offset: usize,
  offset_data: usize,
  max_level: i32,
  entry_point: Option<u32>,
  max_m: usize,
  max_m0: usize,
  m: usize,
  ef_construction: usize,
}

fn parse_header(dim: usize, bytes: &[u8]) -> Result<LegacyHeader> {
  let mut last_err = None;
  for &usize_width in &[8usize, 4usize] {
    match parse_header_with_width(dim, bytes, usize_width) {
      Ok(h) => return Ok(h),
      Err(e) => last_err = Some(e),
    }
  }
  Err(last_err.unwrap_or_else(|| {
    Error::InvalidIndexFormat("failed to parse legacy header".to_string())
  }))
}

fn parse_header_with_width(dim: usize, bytes: &[u8], usize_width: usize) -> Result<LegacyHeader> {
  let rd = &mut &*bytes;

  let offset_level0 = consume_usize_width(rd, usize_width)?;
  let max_elements = consume_usize_width(rd, usize_width)?;
  let cur_element_count = consume_usize_width(rd, usize_width)?;
  let size_data_per_element = consume_usize_width(rd, usize_width)?;
  let label_offset = consume_usize_width(rd, usize_width)?;
  let offset_data = consume_usize_width(rd, usize_width)?;
  let max_level = consume_i32(rd)?;
  let entry_point_raw = consume_u32(rd)?;
  let max_m = consume_usize_width(rd, usize_width)?;
  let max_m0 = consume_usize_width(rd, usize_width)?;
  let m = consume_usize_width(rd, usize_width)?;
  let mult = consume_f64(rd)?;
  let ef_construction = consume_usize_width(rd, usize_width)?;

  let header_size = bytes.len() - rd.len();

  if offset_level0 != 0 {
    return Err(Error::InvalidIndexFormat(format!(
      "unsupported offsetLevel0={offset_level0}"
    )));
  }
  if cur_element_count > max_elements {
    return Err(Error::InvalidIndexFormat(
      "cur_element_count > max_elements".to_string(),
    ));
  }
  if max_elements > off64::usz!(u32::MAX) {
    return Err(Error::InvalidIndexFormat(
      "max_elements exceeds u32::MAX".to_string(),
    ));
  }
  if dim == 0 {
    return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
  }
  if size_data_per_element == 0 {
    return Err(Error::InvalidIndexFormat(
      "size_data_per_element is 0".to_string(),
    ));
  }
  if label_offset > size_data_per_element || offset_data > size_data_per_element {
    return Err(Error::InvalidIndexFormat(
      "label_offset/offset_data out of bounds".to_string(),
    ));
  }
  if m == 0 {
    return Err(Error::InvalidIndexFormat("M is 0".to_string()));
  }

  let entry_point = (entry_point_raw != u32::MAX).then_some(entry_point_raw);
  if cur_element_count == 0 {
    if entry_point.is_some() || max_level != -1 {
      return Err(Error::InvalidIndexFormat(
        "empty index has non-empty entry point/maxlevel".to_string(),
      ));
    }
  } else {
    let Some(ep) = entry_point else {
      return Err(Error::InvalidIndexFormat(
        "non-empty index missing entry point".to_string(),
      ));
    };
    if off64::usz!(ep) >= cur_element_count {
      return Err(Error::InvalidIndexFormat(
        "entry point out of bounds".to_string(),
      ));
    }
    if max_level < 0 {
      return Err(Error::InvalidIndexFormat(
        "non-empty index has maxlevel < 0".to_string(),
      ));
    }
  }

  let size_links_level0 = max_m0
    .checked_mul(size_of::<u32>())
    .and_then(|v| v.checked_add(size_of::<u32>()))
    .ok_or_else(|| Error::InvalidIndexFormat("size_links_level0 overflow".to_string()))?;
  let vector_bytes = dim
    .checked_mul(size_of::<f32>())
    .ok_or_else(|| Error::InvalidIndexFormat("vector size overflow".to_string()))?;
  if offset_data != size_links_level0 {
    return Err(Error::InvalidIndexFormat(
      "unsupported offsetData (expected base linklist size)".to_string(),
    ));
  }
  if label_offset != offset_data + vector_bytes {
    return Err(Error::InvalidIndexFormat(
      "unsupported label_offset (expected after vector data)".to_string(),
    ));
  }
  let label_size = size_data_per_element - label_offset;
  if label_size != 4 && label_size != 8 {
    return Err(Error::InvalidIndexFormat(format!(
      "unsupported label size {label_size} (expected 4 or 8)"
    )));
  }
  if size_data_per_element != size_links_level0 + vector_bytes + label_size {
    return Err(Error::InvalidIndexFormat(
      "size_data_per_element mismatch".to_string(),
    ));
  }

  if !mult.is_finite() || mult <= 0.0 {
    return Err(Error::InvalidIndexFormat(
      "invalid mult".to_string(),
    ));
  }
  if ef_construction < m {
    return Err(Error::InvalidIndexFormat(
      "ef_construction < M".to_string(),
    ));
  }
  if max_m != m {
    return Err(Error::InvalidIndexFormat(
      "maxM != M is not supported".to_string(),
    ));
  }
  if max_m0 != m * 2 {
    return Err(Error::InvalidIndexFormat(
      "maxM0 != 2*M is not supported".to_string(),
    ));
  }

  Ok(LegacyHeader {
    header_size,
    offset_level0,
    max_elements,
    cur_element_count,
    size_data_per_element,
    label_offset,
    offset_data,
    max_level,
    entry_point,
    max_m,
    max_m0,
    m,
    ef_construction,
  })
}

/// Read-only vector store backed by the legacy index file bytes.
#[derive(Debug, Clone)]
pub struct LegacyVectors<'a> {
  dim: usize,
  nodes: usize,
  size_data_per_element: usize,
  offset_data_within_element: usize,
  data_level0_offset: usize,
  bytes: &'a [u8],
}

impl<'a> LegacyVectors<'a> {
  fn raw_vector_bytes(&self, id: NodeId) -> Option<&'a [u8]> {
    let internal_id = id.as_usize();
    if internal_id >= self.nodes {
      return None;
    }
    let start = self
      .data_level0_offset
      .checked_add(internal_id.checked_mul(self.size_data_per_element)?)?
      .checked_add(self.offset_data_within_element)?;
    let end = start.checked_add(self.dim.checked_mul(size_of::<f32>())?)?;
    if end > self.bytes.len() {
      return None;
    }
    Some(&self.bytes[start..end])
  }
}

impl VectorStore for LegacyVectors<'_> {
  type Scalar = f32;
  type Vector<'a> = &'a [f32] where Self: 'a;

  fn dim(&self) -> usize {
    self.dim
  }

  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>> {
    let bytes = self.raw_vector_bytes(id)?;
    let v = bytemuck::try_cast_slice::<u8, f32>(bytes).ok()?;
    debug_assert_eq!(v.len(), self.dim);
    Some(v)
  }
}

/// Loads a legacy `hnswlib` index from bytes.
///
/// The returned graph uses `u64` keys (legacy labels), and preserves the original internal node
/// IDs (dense `0..cur_element_count`) for maximal fidelity.
///
/// The returned `LegacyVectors` is keyed by `NodeId` (the internal IDs).
pub fn load_hnswlib<'a, M: Metric<Scalar = f32>>(
  metric: M,
  dim: usize,
  bytes: &'a [u8],
) -> Result<(Hnsw<u64, M>, LegacyVectors<'a>)> {
  if (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
    return Err(Error::InvalidIndexFormat(
      "index bytes are not aligned to f32".to_string(),
    ));
  }

  let header = parse_header(dim, &bytes)?;
  let size_links_per_element = header
    .max_m
    .checked_mul(size_of::<u32>())
    .and_then(|v| v.checked_add(size_of::<u32>()))
    .ok_or_else(|| Error::InvalidIndexFormat("size_links_per_element overflow".to_string()))?;

  let data_level0_len = header
    .cur_element_count
    .checked_mul(header.size_data_per_element)
    .ok_or_else(|| Error::InvalidIndexFormat("data_level0_len overflow".to_string()))?;
  let data_level0_offset = header.header_size;
  if data_level0_offset + data_level0_len > bytes.len() {
    return Err(Error::InvalidIndexFormat(
      "data_level0_memory out of bounds".to_string(),
    ));
  }
  let data_level0 = &bytes[data_level0_offset..data_level0_offset + data_level0_len];

  let cfg = HnswConfig::new(dim, header.max_elements)
    .m(header.m)
    .ef_construction(header.ef_construction)
    // hnswlib default.
    .ef_search(10)
    .seed(100);
  let graph = Hnsw::new(metric, cfg);
  graph.legacy_start_loading(header.cur_element_count)?;
  let mut deleted_count = 0usize;

  let mut rd = &bytes[data_level0_offset + data_level0_len..];
  for internal_id in 0..header.cur_element_count {
    let elem_off = internal_id * header.size_data_per_element;

    let label_size = header.size_data_per_element - header.label_offset;
    let label_off = elem_off + header.label_offset;
    let label = match label_size {
      8 => u64::from_le_bytes(data_level0[label_off..label_off + 8].try_into().unwrap()),
      4 => off64::u64!(u32::from_le_bytes(
        data_level0[label_off..label_off + 4].try_into().unwrap(),
      )),
      _ => unreachable!("validated in header"),
    };
    graph.legacy_set_node_key(off64::u32!(internal_id), label)?;

    let deleted = (data_level0[elem_off + header.offset_level0 + 2] & 0x01) != 0;
    if deleted {
      deleted_count += 1;
    }
    graph.legacy_set_node_deleted(off64::u32!(internal_id), deleted)?;

    let link_list_size = off64::usz!(consume_u32(&mut rd)?);
    let node_level = if link_list_size == 0 {
      0usize
    } else {
      if link_list_size % size_links_per_element != 0 {
        return Err(Error::InvalidIndexFormat("invalid linkListSize".to_string()));
      }
      link_list_size / size_links_per_element
    };
    if node_level > off64::usz!(i32::MAX) {
      return Err(Error::InvalidIndexFormat(
        "node level too large".to_string(),
      ));
    }
    graph.legacy_set_node_level(off64::u32!(internal_id), off64::i32!(node_level))?;

    // Level 0 neighbors (in the base layer memory block).
    {
      let block = &data_level0[elem_off + header.offset_level0..elem_off + header.offset_level0 + header.offset_data];
      let cnt = off64::usz!(u16::from_le_bytes(block[..2].try_into().unwrap()));
      if cnt > header.max_m0 {
        return Err(Error::InvalidIndexFormat("linklist0 too large".to_string()));
      }
      let mut neighbors = Vec::with_capacity(cnt);
      let mut p = &block[4..];
      for _ in 0..cnt {
        neighbors.push(consume_u32(&mut p)?);
      }
      graph.legacy_set_neighbors(off64::u32!(internal_id), 0, &neighbors)?;
    }

    if link_list_size > 0 {
      let link_list = consume_bytes(&mut rd, link_list_size)?;
      for level in 1..=node_level {
        let start = (level - 1) * size_links_per_element;
        let end = start + size_links_per_element;
        let block = &link_list[start..end];
        let cnt = off64::usz!(u16::from_le_bytes(block[..2].try_into().unwrap()));
        if cnt > header.max_m {
          return Err(Error::InvalidIndexFormat("linklist too large".to_string()));
        }
        let mut neighbors = Vec::with_capacity(cnt);
        let mut p = &block[4..];
        for _ in 0..cnt {
          neighbors.push(consume_u32(&mut p)?);
        }
        graph.legacy_set_neighbors(off64::u32!(internal_id), level, &neighbors)?;
      }
    }
  }

  if !rd.is_empty() {
    return Err(Error::InvalidIndexFormat(
      "trailing bytes after index".to_string(),
    ));
  }
  graph.legacy_finish_loading(header.max_level, header.entry_point, deleted_count)?;

  let vectors = LegacyVectors {
    dim,
    nodes: header.cur_element_count,
    size_data_per_element: header.size_data_per_element,
    offset_data_within_element: header.offset_data,
    data_level0_offset,
    bytes,
  };

  Ok((graph, vectors))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::metric::L2;
  use crate::vectors::VectorStore;

  const FIXTURE_L2_DIM4_N32_B64: &str = "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AAAAAAAAAAAAAAAAAACMQwCAjEMAAI1DAICNQxwAAAAAAAAAAgAAABwAAAAeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACRQwCAkUMAAJJDAICSQx0AAAAAAAAAAgAAAB0AAAAfAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACWQwCAlkMAAJdDAICXQx4AAAAAAAAAAQAAAB4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACbQwCAm0MAAJxDAICcQx8AAAAAAAAAFAAAAAMAAAAJAAAAEAAAABMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAAwAAAAAAAAAQAAAAEwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAAwAAAAAAAAAJAAAAEwAAAAAAAAAAAAAAAAAAABQAAAAEAAAAAAAAAAkAAAAQAAAAFAAAABQAAAACAAAAEwAAABwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAAQAAABQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

  fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> Option<u8> {
      match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => None,
        _ => None,
      }
    }

    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
      let Some(v) = val(c) else {
        break;
      };
      buf = (buf << 6) | off64::u32!(v);
      bits += 6;
      while bits >= 8 {
        bits -= 8;
        out.push(off64::u8!((buf >> bits) & 0xff));
      }
    }
    out
  }

  #[test]
  fn loads_real_hnswlib_fixture() {
    let bytes = b64_decode(FIXTURE_L2_DIM4_N32_B64);
    let (graph, vectors) = load_hnswlib(L2::new(), 4, &bytes).unwrap();

    assert_eq!(graph.dim(), 4);
    assert_eq!(graph.max_nodes(), 32);
    assert_eq!(graph.len(), 32);
    assert_eq!(graph.m(), 4);
    assert_eq!(graph.ef_construction(), 40);
    assert_eq!(graph.ef_search(), 10);

    let labels = graph.keys();
    assert_eq!(labels.len(), 32);
    assert_eq!(labels[0], 0);
    assert_eq!(labels[31], 31);

    let v0_id = graph.node_id(&0).unwrap();
    let v0 = vectors.vector(v0_id).unwrap();
    let v0: &[f32] = v0.as_ref();
    assert_eq!(v0, &[0.0, 1.0, 2.0, 3.0]);
    let v7_id = graph.node_id(&7).unwrap();
    let v7 = vectors.vector(v7_id).unwrap();
    let v7: &[f32] = v7.as_ref();
    assert_eq!(v7, &[70.0, 71.0, 72.0, 73.0]);

    assert!(graph.is_deleted_key(&5).unwrap());
    assert!(!graph.is_deleted_key(&6).unwrap());

    graph.set_ef_search(256);
    for key in [0u64, 1, 2, 7, 31] {
      let id = graph.node_id(&key).unwrap();
      let q = vectors.vector(id).unwrap();
      let hits = graph.search(&vectors, q.as_ref(), 1, None).unwrap();
      assert_eq!(hits.len(), 1);
      assert_eq!(hits[0].key, key);
    }
  }
}
