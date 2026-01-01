pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
  #[error("dimension mismatch: expected {expected}, got {actual}")]
  DimensionMismatch { expected: usize, actual: usize },

  #[error("key not found")]
  KeyNotFound,

  #[error("key already exists")]
  KeyAlreadyExists,

  #[error("index is full (max_nodes={max_nodes})")]
  IndexFull { max_nodes: usize },

  #[error("index is empty")]
  EmptyIndex,

  #[error("missing vector")]
  MissingVector,

  #[error("invalid index format: {0}")]
  InvalidIndexFormat(String),

  #[error(transparent)]
  Io(#[from] std::io::Error),
}
