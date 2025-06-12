use hnswlib_rs::HnswIndex;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use num_traits::ToPrimitive;
use std::path::Path;

pub mod cmd;

pub fn new_pb_with_template(len: impl ToPrimitive, template: &'static str) -> ProgressBar {
  let pb = ProgressBar::new(len.to_u64().unwrap());
  pb.set_style(
    ProgressStyle::with_template(template)
      .unwrap()
      .progress_chars("#>-"),
  );
  pb
}

/// Create a new progress bar that will show a custom message instead of the progress ratio.
/// This custom message can be set using `.set_message(...)`.
pub fn new_pb_with_msg(len: impl ToPrimitive) -> ProgressBar {
  new_pb_with_template(
    len,
    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} ({eta})",
  )
}

/// Create a new progress bar.
pub fn new_pb(len: impl ToPrimitive) -> ProgressBar {
  new_pb_with_template(
    len,
    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
  )
}

pub fn load_hnsw(dim: usize, path: impl AsRef<Path>) -> HnswIndex {
  let raw = std::fs::File::open(path).unwrap();
  let mut rd = std::io::BufReader::new(raw);
  HnswIndex::load(dim, &mut rd)
}
