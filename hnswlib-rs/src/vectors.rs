use crate::error::Error;
use crate::error::Result;
use crate::id::NodeId;
use arc_swap::ArcSwapOption;
use std::io::SeekFrom;
use std::mem::size_of;
use std::sync::Arc;
use std::path::Path;
use tokio::fs::OpenOptions;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::BufWriter;
use tokio::task::JoinSet;

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

  pub async fn save_to_file(
    self: &Arc<Self>,
    path: impl AsRef<Path>,
    node_count: usize,
    concurrency: usize,
  ) -> Result<()> {
    if node_count > self.max_nodes() {
      return Err(Error::InvalidIndexFormat(
        "node_count exceeds vector store capacity".to_string(),
      ));
    }
    if self.dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if concurrency == 0 {
      return Err(Error::InvalidIndexFormat(
        "concurrency must be > 0".to_string(),
      ));
    }

    let bytes_per_vector = self
      .dim
      .checked_mul(size_of::<f32>())
      .ok_or_else(|| Error::InvalidIndexFormat("vector byte size overflow".to_string()))?;
    let total_bytes = node_count
      .checked_mul(bytes_per_vector)
      .ok_or_else(|| Error::InvalidIndexFormat("file length overflow".to_string()))?;

    let path = path.as_ref().to_path_buf();

    {
      let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .await?;
      file.set_len(off64::u64!(total_bytes)).await?;
    }

    if node_count == 0 {
      return Ok(());
    }

    let store = Arc::clone(self);
    let tasks = concurrency.min(node_count).max(1);
    let mut set = JoinSet::<Result<()>>::new();
    for task_idx in 0..tasks {
      let store = Arc::clone(&store);
      let path = path.clone();
      let start = task_idx * node_count / tasks;
      let end = (task_idx + 1) * node_count / tasks;
      set.spawn(async move {
        let mut file = OpenOptions::new().write(true).open(&path).await?;
        file
          .seek(SeekFrom::Start(off64::u64!(start * bytes_per_vector)))
          .await?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);

        for internal_id in start..end {
          let v = store
            .vectors
            .get(internal_id)
            .and_then(|slot| slot.load_full())
            .ok_or(Error::MissingVector)?;
          let slice: &[f32] = v.as_ref().as_ref();
          if slice.len() != store.dim {
            return Err(Error::DimensionMismatch {
              expected: store.dim,
              actual: slice.len(),
            });
          }
          w.write_all(bytemuck::cast_slice(slice)).await?;
        }

        w.flush().await?;
        Ok(())
      });
    }

    while let Some(res) = set.join_next().await {
      match res {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
          set.abort_all();
          return Err(err);
        }
        Err(e) => {
          set.abort_all();
          return Err(Error::InvalidIndexFormat(format!(
            "save_to_file task failed: {e}"
          )));
        }
      }
    }

    Ok(())
  }

  pub async fn load_from_file(
    path: impl AsRef<Path>,
    dim: usize,
    max_nodes: usize,
    concurrency: usize,
  ) -> Result<(Arc<Self>, usize)> {
    if dim == 0 {
      return Err(Error::InvalidIndexFormat("dim must be > 0".to_string()));
    }
    if max_nodes > off64::usz!(u32::MAX) {
      return Err(Error::InvalidIndexFormat(
        "max_nodes exceeds u32::MAX".to_string(),
      ));
    }
    if concurrency == 0 {
      return Err(Error::InvalidIndexFormat(
        "concurrency must be > 0".to_string(),
      ));
    }

    let bytes_per_vector = dim
      .checked_mul(size_of::<f32>())
      .ok_or_else(|| Error::InvalidIndexFormat("vector byte size overflow".to_string()))?;

    let path = path.as_ref().to_path_buf();
    let meta = tokio::fs::metadata(&path).await?;
    let file_len = off64::usz!(meta.len());
    if file_len % bytes_per_vector != 0 {
      return Err(Error::InvalidIndexFormat(
        "vector file length is not a multiple of dim*sizeof(f32)".to_string(),
      ));
    }
    let node_count = file_len / bytes_per_vector;
    if node_count > max_nodes {
      return Err(Error::InvalidIndexFormat(
        "node_count exceeds max_nodes".to_string(),
      ));
    }

    let store = Arc::new(InMemoryVectorStore::new(dim, max_nodes));
    if node_count == 0 {
      return Ok((store, 0));
    }

    let tasks = concurrency.min(node_count).max(1);
    let mut set = JoinSet::<Result<()>>::new();
    for task_idx in 0..tasks {
      let store = Arc::clone(&store);
      let path = path.clone();
      let start = task_idx * node_count / tasks;
      let end = (task_idx + 1) * node_count / tasks;
      set.spawn(async move {
        let mut file = OpenOptions::new().read(true).open(&path).await?;
        file
          .seek(SeekFrom::Start(off64::u64!(start * bytes_per_vector)))
          .await?;
        let mut r = BufReader::with_capacity(8 * 1024 * 1024, file);

        for internal_id in start..end {
          let mut v = vec![0f32; dim];
          r.read_exact(bytemuck::cast_slice_mut(&mut v)).await?;
          store.vectors[internal_id].store(Some(Arc::new(v.into_boxed_slice())));
        }

        Ok(())
      });
    }

    while let Some(res) = set.join_next().await {
      match res {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
          set.abort_all();
          return Err(err);
        }
        Err(e) => {
          set.abort_all();
          return Err(Error::InvalidIndexFormat(format!(
            "load_from_file task failed: {e}"
          )));
        }
      }
    }

    Ok((store, node_count))
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
    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    rt.block_on(async {
      let dir = tempdir().unwrap();
      let path = dir.path().join("vectors.bin");

      let dim = 8;
      let max_nodes = 100;
      let node_count = 25;
      let concurrency = 4;

      let store = Arc::new(InMemoryVectorStore::new(dim, max_nodes));
      for i in 0..node_count {
        let v = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
        store.set(NodeId::new(off64::u32!(i)), &v).unwrap();
      }

      store
        .save_to_file(&path, node_count, concurrency)
        .await
        .unwrap();

      let (loaded, loaded_count) =
        InMemoryVectorStore::load_from_file(&path, dim, max_nodes, concurrency)
          .await
          .unwrap();
      assert_eq!(loaded_count, node_count);
      assert_eq!(loaded.dim(), dim);
      assert_eq!(loaded.max_nodes(), max_nodes);

      for i in 0..node_count {
        let got = loaded.vector(NodeId::new(off64::u32!(i))).unwrap();
        let got = got.as_f32_slice();
        let expected = (0..dim).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
        assert_eq!(got, expected.as_slice());
      }
    });
  }

  #[test]
  fn save_errors_on_missing_vector() {
    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    rt.block_on(async {
      let dir = tempdir().unwrap();
      let path = dir.path().join("vectors.bin");

      let dim = 4;
      let max_nodes = 10;
      let concurrency = 4;
      let store = Arc::new(InMemoryVectorStore::new(dim, max_nodes));
      store.set(NodeId::new(0), &[1.0, 2.0, 3.0, 4.0]).unwrap();
      let err = store
        .save_to_file(&path, 2, concurrency)
        .await
        .unwrap_err();
      assert!(matches!(err, Error::MissingVector));
    });
  }
}
