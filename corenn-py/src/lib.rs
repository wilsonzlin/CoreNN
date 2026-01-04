use half::bf16;
use half::f16;
use libcorenn::cfg::Cfg;
use libcorenn::CoreNN as CoreNNNative;
use numpy::PyReadonlyArray2;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::pymodule;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use serde_pyobject::from_pyobject;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass]
struct CoreNN(Arc<CoreNNNative>);

#[pymethods]
impl CoreNN {
  #[staticmethod]
  pub fn create(dir: PathBuf, cfg: Bound<PyAny>) -> PyResult<CoreNN> {
    let cfg: Cfg = from_pyobject(cfg)?;
    let db = Arc::new(CoreNNNative::create(dir, cfg));
    Ok(CoreNN(db))
  }

  #[staticmethod]
  pub fn open(dir: PathBuf) -> PyResult<CoreNN> {
    let db = Arc::new(CoreNNNative::open(dir));
    Ok(CoreNN(db))
  }

  #[staticmethod]
  pub fn new_in_memory(cfg: Bound<PyAny>) -> PyResult<CoreNN> {
    let cfg: Cfg = from_pyobject(cfg)?;
    let db = Arc::new(CoreNNNative::new_in_memory(cfg));
    Ok(CoreNN(db))
  }

  pub fn insert_bf16(&self, keys: Vec<String>, vectors: PyReadonlyArray2<bf16>) {
    zip(keys, vectors.as_array().rows())
      .par_bridge()
      .for_each(|(k, v)| {
        self.0.insert(&k, v.as_slice().unwrap());
      });
  }

  pub fn insert_f16(&self, keys: Vec<String>, vectors: PyReadonlyArray2<f16>) {
    zip(keys, vectors.as_array().rows())
      .par_bridge()
      .for_each(|(k, v)| {
        self.0.insert(&k, v.as_slice().unwrap());
      });
  }

  pub fn insert_f32(&self, keys: Vec<String>, vectors: PyReadonlyArray2<f32>) {
    zip(keys, vectors.as_array().rows())
      .par_bridge()
      .for_each(|(k, v)| {
        self.0.insert(&k, v.as_slice().unwrap());
      });
  }

  pub fn query_bf16(&self, queries: PyReadonlyArray2<bf16>, k: usize) -> Vec<Vec<(String, f32)>> {
    queries
      .as_array()
      .rows()
      .into_iter()
      .par_bridge()
      .map(|q| self.0.query(q.as_slice().unwrap(), k))
      .collect()
  }

  pub fn query_f16(&self, queries: PyReadonlyArray2<f16>, k: usize) -> Vec<Vec<(String, f32)>> {
    queries
      .as_array()
      .rows()
      .into_iter()
      .par_bridge()
      .map(|q| self.0.query(q.as_slice().unwrap(), k))
      .collect()
  }

  pub fn query_f32(&self, queries: PyReadonlyArray2<f32>, k: usize) -> Vec<Vec<(String, f32)>> {
    queries
      .as_array()
      .rows()
      .into_iter()
      .par_bridge()
      .map(|q| self.0.query(q.as_slice().unwrap(), k))
      .collect()
  }

}

#[pymodule]
fn corenn_py(m: &Bound<PyModule>) -> PyResult<()> {
  // Sends tracing messages to Python logging.
  pyo3_log::init();
  m.add_class::<CoreNN>()?;
  Ok(())
}
