use crate::error::Error;
use crate::error::Result;
use crate::id::NodeId;
use arc_swap::ArcSwapOption;
use std::io::Read;
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

pub trait VectorRef {
  fn as_f32_slice(&self) -> &[f32];
}

impl<T> VectorRef for T
where
  T: AsRef<[f32]>,
{
  fn as_f32_slice(&self) -> &[f32] {
    self.as_ref()
  }
}

#[derive(Clone)]
pub struct ArcVector(Arc<Box<[f32]>>);

impl AsRef<[f32]> for ArcVector {
  fn as_ref(&self) -> &[f32] {
    self.0.as_ref().as_ref()
  }
}

pub trait VectorStore: Send + Sync {
  type Vector<'a>: VectorRef
  where
    Self: 'a;

  fn dim(&self) -> usize;
  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>>;
}

pub trait VectorStoreMut: VectorStore {
  fn set(&self, id: NodeId, vector: &[f32]) -> Result<()>;
}

pub struct InMemoryVectorStore {
  dim: usize,
  vectors: Vec<ArcSwapOption<Box<[f32]>>>,
}

impl InMemoryVectorStore {
  pub fn new(dim: usize, max_nodes: usize) -> Self {
    let mut vectors = Vec::with_capacity(max_nodes);
    vectors.resize_with(max_nodes, ArcSwapOption::empty);
    Self { dim, vectors }
  }

  pub fn max_nodes(&self) -> usize {
    self.vectors.len()
  }

  pub fn save_to<W: Write>(&self, w: &mut W, node_count: usize) -> Result<()> {
    #[cfg(not(target_endian = "little"))]
    {
      return Err(Error::InvalidIndexFormat(
        "InMemoryVectorStore persistence requires little-endian".to_string(),
      ));
    }

    if node_count > self.max_nodes() {
      return Err(Error::InvalidIndexFormat(
        "node_count exceeds vector store capacity".to_string(),
      ));
    }
    if self.dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }

    for internal_id in 0..node_count {
      let v = self
        .vectors
        .get(internal_id)
        .and_then(|slot| slot.load_full())
        .ok_or(Error::MissingVector)?;
      let slice: &[f32] = v.as_ref().as_ref();
      if slice.len() != self.dim {
        return Err(Error::DimensionMismatch {
          expected: self.dim,
          actual: slice.len(),
        });
      }
      w.write_all(bytemuck::cast_slice(slice))?;
    }
    w.flush()?;
    Ok(())
  }

  pub fn load_from<R: Read>(
    r: &mut R,
    dim: usize,
    max_nodes: usize,
  ) -> Result<(Self, usize)> {
    #[cfg(not(target_endian = "little"))]
    {
      return Err(Error::InvalidIndexFormat(
        "InMemoryVectorStore persistence requires little-endian".to_string(),
      ));
    }

    if dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if max_nodes > off64::usz!(u32::MAX) {
      return Err(Error::InvalidIndexFormat(
        "max_nodes exceeds u32::MAX".to_string(),
      ));
    }

    let bytes_per_vector = dim
      .checked_mul(size_of::<f32>())
      .ok_or_else(|| Error::InvalidIndexFormat("vector byte size overflow".to_string()))?;

    let store = InMemoryVectorStore::new(dim, max_nodes);
    let mut node_count = 0usize;
    loop {
      let mut v = vec![0f32; dim];
      let bytes = bytemuck::cast_slice_mut(&mut v);
      debug_assert_eq!(bytes.len(), bytes_per_vector);

      let mut read = 0usize;
      while read < bytes.len() {
        let n = r.read(&mut bytes[read..])?;
        if n == 0 {
          if read == 0 {
            return Ok((store, node_count));
          }
          return Err(Error::InvalidIndexFormat(
            "vector data truncated".to_string(),
          ));
        }
        read += n;
      }

      if node_count >= max_nodes {
        return Err(Error::InvalidIndexFormat(
          "node_count exceeds max_nodes".to_string(),
        ));
      }
      store.vectors[node_count].store(Some(Arc::new(v.into_boxed_slice())));
      node_count += 1;
    }
  }
}

impl VectorStore for InMemoryVectorStore {
  type Vector<'a> = ArcVector where Self: 'a;

  fn dim(&self) -> usize {
    self.dim
  }

  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>> {
    self
      .vectors
      .get(id.as_usize())?
      .load_full()
      .map(ArcVector)
  }
}

impl VectorStoreMut for InMemoryVectorStore {
  fn set(&self, id: NodeId, vector: &[f32]) -> Result<()> {
    if vector.len() != self.dim {
      return Err(Error::DimensionMismatch {
        expected: self.dim,
        actual: vector.len(),
      });
    }
    let slot = self
      .vectors
      .get(id.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node id out of bounds".to_string()))?;
    slot.store(Some(Arc::new(vector.to_vec().into_boxed_slice())));
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
	  fn in_memory_vector_store_save_load_roundtrip() {
	    let dir = tempdir().unwrap();
	    let path = dir.path().join("vectors.bin");

    let dim = 8;
    let max_nodes = 100;
    let node_count = 25;

    let store = InMemoryVectorStore::new(dim, max_nodes);
	    for i in 0..node_count {
	      let v = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
	      store.set(NodeId::new(off64::u32!(i)), &v).unwrap();
	    }

	    {
	      let mut f = std::fs::File::create(&path).unwrap();
	      store.save_to(&mut f, node_count).unwrap();
	    }

	    let (loaded, loaded_count) = {
	      let mut f = std::fs::File::open(&path).unwrap();
	      InMemoryVectorStore::load_from(&mut f, dim, max_nodes).unwrap()
	    };
	    assert_eq!(loaded_count, node_count);
	    assert_eq!(loaded.dim(), dim);
	    assert_eq!(loaded.max_nodes(), max_nodes);

    for i in 0..node_count {
      let got = loaded.vector(NodeId::new(off64::u32!(i))).unwrap();
      let got = got.as_f32_slice();
      let expected = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
      assert_eq!(got, expected.as_slice());
    }
  }

  #[test]
	  fn save_errors_on_missing_vector() {
	    let dir = tempdir().unwrap();
	    let path = dir.path().join("vectors.bin");

	    let dim = 4;
	    let max_nodes = 10;
	    let store = InMemoryVectorStore::new(dim, max_nodes);
	    store.set(NodeId::new(0), &[1.0, 2.0, 3.0, 4.0]).unwrap();
	    let err = {
	      let mut f = std::fs::File::create(&path).unwrap();
	      store.save_to(&mut f, 2).unwrap_err()
	    };
	    assert!(matches!(err, Error::MissingVector));
	  }
}
