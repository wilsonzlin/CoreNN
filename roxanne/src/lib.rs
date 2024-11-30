use indicatif::ProgressBar;
use indicatif::ProgressStyle;

pub mod cmd;

pub fn new_pb(len: usize) -> ProgressBar {
  let pb = ProgressBar::new(len.try_into().unwrap());
  pb.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-"),
  );
  pb
}
