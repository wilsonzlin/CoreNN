use crate::LabelType;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
  #[error("dimension mismatch: expected {expected}, got {actual}")]
  DimensionMismatch { expected: usize, actual: usize },

  #[error("label not found: {0}")]
  LabelNotFound(LabelType),

  #[error("index is full (max_elements={max_elements})")]
  IndexFull { max_elements: usize },

  #[error("index is empty")]
  EmptyIndex,

  #[error("invalid index format: {0}")]
  InvalidIndexFormat(String),

  #[error(transparent)]
  Io(#[from] std::io::Error),
}
