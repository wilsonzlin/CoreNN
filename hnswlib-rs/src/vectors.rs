use crate::error::Error;
use crate::error::Result;
use crate::id::NodeId;
use crate::scalar::Dtype;
use crate::scalar::Scalar;
use crate::vector::Dense;
use crate::vector::Qi8;
use crate::vector::Qi8Ref;
use crate::vector::VectorFamily;
use crate::vector::VectorView;
use arc_swap::ArcSwapOption;
use std::io::Read;
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

const VECTOR_STORE_FILE_VERSION: u32 = 1;

#[derive(serde::Serialize, serde::Deserialize)]
struct VectorStoreHeader {
  version: u32,
  dtype: Dtype,
  dim: u32,
  max_nodes: u32,
  node_count: u32,
}

#[derive(Clone)]
pub struct ArcVector<S: Scalar>(Arc<Box<[S]>>);

impl<S: Scalar> VectorView<Dense<S>> for ArcVector<S> {
  fn view<'a>(&'a self) -> &'a [S] {
    self.0.as_ref().as_ref()
  }
}

#[derive(Clone)]
struct Qi8Vector {
  data: Box<[i8]>,
  scale: f32,
  zero_point: i8,
}

#[derive(Clone)]
pub struct ArcQi8Vector(Arc<Qi8Vector>);

impl VectorView<Qi8> for ArcQi8Vector {
  fn view<'a>(&'a self) -> Qi8Ref<'a> {
    Qi8Ref {
      data: self.0.data.as_ref(),
      scale: self.0.scale,
      zero_point: self.0.zero_point,
    }
  }
}

pub trait VectorStore: Send + Sync {
  type Family: VectorFamily;
  type Vector<'a>: VectorView<Self::Family>
  where
    Self: 'a;

  fn dim(&self) -> usize;
  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>>;
}

pub trait VectorStoreMut: VectorStore {
  fn set<'a>(&self, id: NodeId, vector: <Self::Family as VectorFamily>::Ref<'a>) -> Result<()>;
}

pub struct InMemoryVectorStore<S: Scalar> {
  dim: usize,
  vectors: Vec<ArcSwapOption<Box<[S]>>>,
}

impl<S: Scalar> InMemoryVectorStore<S> {
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
        "vector store persistence requires little-endian".to_string(),
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

    let header = VectorStoreHeader {
      version: VECTOR_STORE_FILE_VERSION,
      dtype: S::DTYPE,
      dim: self
        .dim
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("dim exceeds u32::MAX".to_string()))?,
      max_nodes: self
        .max_nodes()
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("max_nodes exceeds u32::MAX".to_string()))?,
      node_count: node_count
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("node_count exceeds u32::MAX".to_string()))?,
    };
    bincode::serialize_into(&mut *w, &header)
      .map_err(|e| Error::InvalidIndexFormat(format!("bincode error: {e}")))?;

    for internal_id in 0..node_count {
      let v = self
        .vectors
        .get(internal_id)
        .and_then(|slot| slot.load_full())
        .ok_or(Error::MissingVector)?;
      let slice: &[S] = v.as_ref().as_ref();
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

  pub fn load_from<R: Read>(r: &mut R) -> Result<(Self, usize)> {
    #[cfg(not(target_endian = "little"))]
    {
      return Err(Error::InvalidIndexFormat(
        "vector store persistence requires little-endian".to_string(),
      ));
    }

    let header: VectorStoreHeader = bincode::deserialize_from(&mut *r)
      .map_err(|e| Error::InvalidIndexFormat(format!("bincode error: {e}")))?;
    if header.version != VECTOR_STORE_FILE_VERSION {
      return Err(Error::InvalidIndexFormat(format!(
        "unsupported vector store version {}",
        header.version
      )));
    }
    if header.dtype != S::DTYPE {
      return Err(Error::InvalidIndexFormat(format!(
        "dtype mismatch: file has {:?}, but this store is {:?}",
        header.dtype,
        S::DTYPE
      )));
    }

    let dim = off64::usz!(header.dim);
    let max_nodes = off64::usz!(header.max_nodes);
    let node_count = off64::usz!(header.node_count);

    if dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if node_count > max_nodes {
      return Err(Error::InvalidIndexFormat(
        "node_count exceeds max_nodes".to_string(),
      ));
    }

    let bytes_per_vector = dim
      .checked_mul(size_of::<S>())
      .ok_or_else(|| Error::InvalidIndexFormat("vector byte size overflow".to_string()))?;

    let store = InMemoryVectorStore::new(dim, max_nodes);
    for internal_id in 0..node_count {
      let mut v = vec![S::from_f32(0.0); dim];
      let bytes = bytemuck::cast_slice_mut(&mut v);
      debug_assert_eq!(bytes.len(), bytes_per_vector);
      r.read_exact(bytes)?;
      store.vectors[internal_id].store(Some(Arc::new(v.into_boxed_slice())));
    }

    Ok((store, node_count))
  }
}

impl<S: Scalar> VectorStore for InMemoryVectorStore<S> {
  type Family = Dense<S>;
  type Vector<'a>
    = ArcVector<S>
  where
    Self: 'a;

  fn dim(&self) -> usize {
    self.dim
  }

  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>> {
    self.vectors.get(id.as_usize())?.load_full().map(ArcVector)
  }
}

impl<S: Scalar> VectorStoreMut for InMemoryVectorStore<S> {
  fn set<'a>(&self, id: NodeId, vector: &'a [S]) -> Result<()> {
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

pub struct InMemoryQi8VectorStore {
  dim: usize,
  vectors: Vec<ArcSwapOption<Qi8Vector>>,
}

impl InMemoryQi8VectorStore {
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
        "vector store persistence requires little-endian".to_string(),
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

    let header = VectorStoreHeader {
      version: VECTOR_STORE_FILE_VERSION,
      dtype: Dtype::QI8,
      dim: self
        .dim
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("dim exceeds u32::MAX".to_string()))?,
      max_nodes: self
        .max_nodes()
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("max_nodes exceeds u32::MAX".to_string()))?,
      node_count: node_count
        .try_into()
        .map_err(|_| Error::InvalidIndexFormat("node_count exceeds u32::MAX".to_string()))?,
    };
    bincode::serialize_into(&mut *w, &header)
      .map_err(|e| Error::InvalidIndexFormat(format!("bincode error: {e}")))?;

    for internal_id in 0..node_count {
      let v = self
        .vectors
        .get(internal_id)
        .and_then(|slot| slot.load_full())
        .ok_or(Error::MissingVector)?;
      if v.data.len() != self.dim {
        return Err(Error::DimensionMismatch {
          expected: self.dim,
          actual: v.data.len(),
        });
      }
      w.write_all(&v.scale.to_le_bytes())?;
      w.write_all(&[v.zero_point as u8])?;
      w.write_all(bytemuck::cast_slice(v.data.as_ref()))?;
    }
    w.flush()?;
    Ok(())
  }

  pub fn load_from<R: Read>(r: &mut R) -> Result<(Self, usize)> {
    #[cfg(not(target_endian = "little"))]
    {
      return Err(Error::InvalidIndexFormat(
        "vector store persistence requires little-endian".to_string(),
      ));
    }

    let header: VectorStoreHeader = bincode::deserialize_from(&mut *r)
      .map_err(|e| Error::InvalidIndexFormat(format!("bincode error: {e}")))?;
    if header.version != VECTOR_STORE_FILE_VERSION {
      return Err(Error::InvalidIndexFormat(format!(
        "unsupported vector store version {}",
        header.version
      )));
    }
    if header.dtype != Dtype::QI8 {
      return Err(Error::InvalidIndexFormat(format!(
        "dtype mismatch: file has {:?}, but this store is {:?}",
        header.dtype,
        Dtype::QI8
      )));
    }

    let dim = off64::usz!(header.dim);
    let max_nodes = off64::usz!(header.max_nodes);
    let node_count = off64::usz!(header.node_count);

    if dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if node_count > max_nodes {
      return Err(Error::InvalidIndexFormat(
        "node_count exceeds max_nodes".to_string(),
      ));
    }

    let store = InMemoryQi8VectorStore::new(dim, max_nodes);
    for internal_id in 0..node_count {
      let mut scale_bytes = [0u8; 4];
      r.read_exact(&mut scale_bytes)?;
      let scale = f32::from_le_bytes(scale_bytes);

      let mut zp = [0u8; 1];
      r.read_exact(&mut zp)?;
      let zero_point = zp[0] as i8;

      let mut data = vec![0i8; dim];
      r.read_exact(bytemuck::cast_slice_mut(&mut data))?;
      store.vectors[internal_id].store(Some(Arc::new(Qi8Vector {
        data: data.into_boxed_slice(),
        scale,
        zero_point,
      })));
    }

    Ok((store, node_count))
  }
}

impl VectorStore for InMemoryQi8VectorStore {
  type Family = Qi8;
  type Vector<'a>
    = ArcQi8Vector
  where
    Self: 'a;

  fn dim(&self) -> usize {
    self.dim
  }

  fn vector<'a>(&'a self, id: NodeId) -> Option<Self::Vector<'a>> {
    self.vectors.get(id.as_usize())?.load_full().map(ArcQi8Vector)
  }
}

impl VectorStoreMut for InMemoryQi8VectorStore {
  fn set<'a>(&self, id: NodeId, vector: Qi8Ref<'a>) -> Result<()> {
    if vector.data.len() != self.dim {
      return Err(Error::DimensionMismatch {
        expected: self.dim,
        actual: vector.data.len(),
      });
    }
    let slot = self
      .vectors
      .get(id.as_usize())
      .ok_or_else(|| Error::InvalidIndexFormat("node id out of bounds".to_string()))?;
    slot.store(Some(Arc::new(Qi8Vector {
      data: vector.data.to_vec().into_boxed_slice(),
      scale: vector.scale,
      zero_point: vector.zero_point,
    })));
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;
  use crate::vector::VectorRef as _;
  use tempfile::tempdir;

  #[test]
  fn in_memory_vector_store_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vectors.bin");

    let dim = 8;
    let max_nodes = 100;
    let node_count = 25;

    let store = InMemoryVectorStore::<f32>::new(dim, max_nodes);
    for i in 0..node_count {
      let v = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
      store.set(NodeId::new(off64::u32!(i)), v.as_slice()).unwrap();
    }

    {
      let mut f = std::fs::File::create(&path).unwrap();
      store.save_to(&mut f, node_count).unwrap();
    }

    let (loaded, loaded_count) = {
      let mut f = std::fs::File::open(&path).unwrap();
      InMemoryVectorStore::<f32>::load_from(&mut f).unwrap()
    };
    assert_eq!(loaded_count, node_count);
    assert_eq!(loaded.dim(), dim);
    assert_eq!(loaded.max_nodes(), max_nodes);

    for i in 0..node_count {
      let got = loaded.vector(NodeId::new(off64::u32!(i))).unwrap();
      let got = got.view();
      let expected = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
      assert_eq!(got, expected.as_slice());
    }
  }

  #[test]
  fn qi8_vector_store_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vectors_qi8.bin");

    let dim = 16;
    let max_nodes = 100;
    let node_count = 10;

    let store = InMemoryQi8VectorStore::new(dim, max_nodes);
    for i in 0..node_count {
      let data = (0..dim).map(|j| (i * 3 + j) as i8).collect::<Vec<_>>();
      store
        .set(
          NodeId::new(off64::u32!(i)),
          Qi8Ref {
            data: data.as_slice(),
            scale: 0.01 * (i as f32 + 1.0),
            zero_point: -3,
          },
        )
        .unwrap();
    }

    {
      let mut f = std::fs::File::create(&path).unwrap();
      store.save_to(&mut f, node_count).unwrap();
    }

    let (loaded, loaded_count) = {
      let mut f = std::fs::File::open(&path).unwrap();
      InMemoryQi8VectorStore::load_from(&mut f).unwrap()
    };
    assert_eq!(loaded_count, node_count);
    assert_eq!(loaded.dim(), dim);
    assert_eq!(loaded.max_nodes(), max_nodes);

    for i in 0..node_count {
      let got = loaded.vector(NodeId::new(off64::u32!(i))).unwrap();
      let got = got.view();
      assert_eq!(got.len(), dim);
      assert_eq!(got.zero_point, -3);
      assert_relative_eq!(got.scale, 0.01 * (i as f32 + 1.0), epsilon = 1e-6);
    }
  }

  #[test]
  fn save_errors_on_missing_vector() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vectors.bin");

    let store = InMemoryVectorStore::<f32>::new(4, 10);
    store.set(NodeId::new(0), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let err = {
      let mut f = std::fs::File::create(&path).unwrap();
      store.save_to(&mut f, 2).unwrap_err()
    };
    assert!(matches!(err, Error::MissingVector));
  }
}
