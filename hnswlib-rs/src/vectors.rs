use crate::error::Error;
use crate::error::Result;
use crate::id::NodeId;
use arc_swap::ArcSwapOption;
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
