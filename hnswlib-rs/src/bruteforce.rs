use crate::space::label_allowed;
use crate::LabelType;
use crate::Space;
use ahash::HashMap;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use std::io;
use std::io::Read;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct BruteforceIndex<S: Space> {
  space: S,
  max_elements: usize,
  cur_element_count: usize,
  vectors: Vec<f32>,
  labels: Vec<LabelType>,
  label_to_internal: HashMap<LabelType, usize>,
}

impl<S: Space> BruteforceIndex<S> {
  pub fn new(space: S, max_elements: usize) -> Self {
    let dim = space.dim();
    Self {
      space,
      max_elements,
      cur_element_count: 0,
      vectors: vec![0.0; max_elements * dim],
      labels: vec![0; max_elements],
      label_to_internal: HashMap::default(),
    }
  }

  pub fn len(&self) -> usize {
    self.cur_element_count
  }

  pub fn is_empty(&self) -> bool {
    self.cur_element_count == 0
  }

  pub fn add_point(&mut self, vector: &[f32], label: LabelType) {
    assert_eq!(vector.len(), self.space.dim());

    let idx = if let Some(&existing) = self.label_to_internal.get(&label) {
      existing
    } else {
      if self.cur_element_count >= self.max_elements {
        panic!("The number of elements exceeds the specified limit");
      }
      let idx = self.cur_element_count;
      self.cur_element_count += 1;
      self.label_to_internal.insert(label, idx);
      self.labels[idx] = label;
      idx
    };

    let dim = self.space.dim();
    self.vectors[idx * dim..idx * dim + dim].copy_from_slice(vector);
    self.labels[idx] = label;
  }

  pub fn remove_point(&mut self, label: LabelType) {
    let Some(idx) = self.label_to_internal.remove(&label) else {
      return;
    };

    let last = self.cur_element_count - 1;
    if idx != last {
      let dim = self.space.dim();
      let (before_last, last_and_after) = self.vectors.split_at_mut(last * dim);
      before_last[idx * dim..idx * dim + dim].copy_from_slice(&last_and_after[..dim]);
      let last_label = self.labels[last];
      self.labels[idx] = last_label;
      self.label_to_internal.insert(last_label, idx);
    }
    self.cur_element_count -= 1;
  }

  pub fn search_knn(
    &self,
    query: &[f32],
    k: usize,
    filter: Option<&dyn Fn(LabelType) -> bool>,
  ) -> Vec<(LabelType, f32)> {
    assert_eq!(query.len(), self.space.dim());
    assert!(k <= self.cur_element_count);

    let mut heap: BinaryHeap<(OrderedFloat<f32>, LabelType)> = BinaryHeap::new();
    let dim = self.space.dim();

    for i in 0..self.cur_element_count {
      let label = self.labels[i];
      if !label_allowed(filter, label) {
        continue;
      }
      let v = &self.vectors[i * dim..i * dim + dim];
      let dist = self.space.distance(query, v);
      if heap.len() < k {
        heap.push((OrderedFloat(dist), label));
      } else if dist <= heap.peek().unwrap().0 .0 {
        heap.pop();
        heap.push((OrderedFloat(dist), label));
      }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some((d, l)) = heap.pop() {
      out.push((l, d.0));
    }
    out.reverse();
    out
  }

  pub fn save_to_writer(&self, mut w: impl Write) -> io::Result<()> {
    let max_elements = self.max_elements as u64;
    let size_per_element =
      (self.space.dim() * std::mem::size_of::<f32>() + std::mem::size_of::<LabelType>()) as u64;
    let cur_element_count = self.cur_element_count as u64;

    w.write_all(&max_elements.to_le_bytes())?;
    w.write_all(&size_per_element.to_le_bytes())?;
    w.write_all(&cur_element_count.to_le_bytes())?;

    let dim = self.space.dim();
    for i in 0..self.max_elements {
      let v = &self.vectors[i * dim..i * dim + dim];
      w.write_all(bytemuck::cast_slice(v))?;
      w.write_all(&self.labels[i].to_le_bytes())?;
    }
    Ok(())
  }

  pub fn load_from_reader(space: S, mut r: impl Read) -> io::Result<Self> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    let max_elements = u64::from_le_bytes(buf) as usize;
    r.read_exact(&mut buf)?;
    let _size_per_element = u64::from_le_bytes(buf) as usize;
    r.read_exact(&mut buf)?;
    let cur_element_count = u64::from_le_bytes(buf) as usize;

    let dim = space.dim();
    let mut idx = Self::new(space, max_elements);
    idx.cur_element_count = cur_element_count;

    for i in 0..max_elements {
      let v = &mut idx.vectors[i * dim..i * dim + dim];
      r.read_exact(bytemuck::cast_slice_mut(v))?;

      let mut lbuf = vec![0u8; std::mem::size_of::<LabelType>()];
      r.read_exact(&mut lbuf)?;
      idx.labels[i] = LabelType::from_le_bytes(lbuf.try_into().unwrap());
    }

    idx.label_to_internal.clear();
    for i in 0..idx.cur_element_count {
      idx.label_to_internal.insert(idx.labels[i], i);
    }

    Ok(idx)
  }
}
