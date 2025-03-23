use half::f16;
use itertools::Itertools;
use libroxanne::Roxanne as RoxanneNative;
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::pymodule;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass]
struct Roxanne(Arc<RoxanneNative>);

#[pymethods]
impl Roxanne {
  #[staticmethod]
  pub fn open(py: Python, dir: PathBuf) -> PyResult<Bound<PyAny>> {
    // We must use pyo3_async_runtimes as we need to have the Tokio runtime running, as libroxanne uses tokio::spawn.
    // (PyO3 does support async methods, but won't start a Rust runtime.)
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let db = RoxanneNative::open(dir).await;
      Ok(Roxanne(db))
    })
  }

  pub fn insert<'a>(
    &self,
    py: Python<'a>,
    keys: Vec<String>,
    vectors: PyReadonlyArray2<'a, f16>,
  ) -> PyResult<Bound<'a, PyAny>> {
    let db = self.0.clone();
    // We must clone as PyReadonlyArray2 (borrowed Python value) cannot be sent between threads safely.
    let entries = keys
      .into_iter()
      .enumerate()
      .map(|(i, k)| {
        let v = vectors.as_array().row(i).to_vec();
        (k, Array1::from(v))
      })
      .collect_vec();
    pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(db.insert(entries).await) })
  }

  pub fn query<'a>(
    &self,
    py: Python<'a>,
    query: PyReadonlyArray1<'a, f16>,
    k: usize,
  ) -> PyResult<Bound<'a, PyAny>> {
    let db = self.0.clone();
    // We must clone as PyReadonlyArray1 (borrowed Python value) cannot be sent between threads safely.
    let query = query.as_slice().unwrap().to_vec();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      Ok(db.query(&Array1::from(query).view(), k).await)
    })
  }
}

#[pymodule]
fn roxanne_py(m: &Bound<PyModule>) -> PyResult<()> {
  pyo3_log::init();
  m.add_class::<Roxanne>()?;
  Ok(())
}
