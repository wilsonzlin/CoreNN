use ahash::HashSet;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use libroxanne::RoxanneDbReadOnly;
use libroxanne_search::GreedySearchable;
use libroxanne_search::Id;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Path to packed u64 vector representing ID of each query.
  #[arg(long)]
  queries: PathBuf,

  /// Path to packed matrix of u64 vectors representing ID of nearest neighbors for each query.
  #[arg(long)]
  knn: PathBuf,

  /// Override query search list cap.
  #[arg(long)]
  query_search_list_cap: Option<usize>,

  /// Override k (must be less than or equal to KNN vector dimensions).
  #[arg(long)]
  k: Option<usize>,

  /// Load as in-memory index.
  #[arg(long)]
  in_memory: bool,
}

fn new_pb(len: usize) -> ProgressBar {
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

fn main() {
  let args = Args::parse();

  // TODO in_memory.
  let mut idx = RoxanneDbReadOnly::open(&args.path);
  println!("Loaded index");

  let query_ids = {
    let raw = fs::read(&args.queries).unwrap();
    assert_eq!(raw.len() % 8, 0);
    let mut out = vec![0; raw.len() / 8];
    LittleEndian::read_u64_into(&raw, &mut out);
    out
  };
  println!("Loaded query IDs");
  let knn_ids = {
    let raw = fs::read(&args.knn).unwrap();
    assert_eq!(raw.len() % (8 * query_ids.len()), 0);
    let k = raw.len() / 8 / query_ids.len();
    (0..query_ids.len())
      .map(|i| {
        let start = i * k * 8;
        let end = (i + 1) * k * 8;
        let mut out = vec![0; k];
        LittleEndian::read_u64_into(&raw[start..end], &mut out);
        out
      })
      .collect::<Vec<_>>()
  };
  println!("Loaded KNN IDs");
  let k = args.k.unwrap_or(knn_ids[0].len());
  if let Some(q) = args.query_search_list_cap {
    idx.params_mut().query_search_list_cap = q;
  }

  let pb = new_pb(query_ids.len());
  let correct = AtomicUsize::new(0);
  let total = AtomicUsize::new(0);
  query_ids
    .into_par_iter()
    .zip(knn_ids)
    .for_each(|(query_id, knn_expected)| {
      let query_id: Id = query_id.try_into().unwrap();

      let knn_expected = knn_expected
        .into_iter()
        .take(k)
        .map(|id| Id::try_from(id).unwrap())
        .collect::<HashSet<_>>();
      let knn_got = idx
        .query(&idx.datastore().get_point(query_id).view(), k)
        .into_iter()
        .map(|e| e.id)
        .collect::<HashSet<_>>();

      correct.fetch_add(
        knn_expected.intersection(&knn_got).count(),
        Ordering::Relaxed,
      );
      total.fetch_add(k, Ordering::Relaxed);

      let correct = correct.load(Ordering::Relaxed);
      let total = total.load(Ordering::Relaxed);
      let pc = correct as f64 / total as f64 * 100.0;
      pb.set_message(format!("Correct: {pc:.2}% ({correct}/{total})"));
      pb.inc(1);
    });
  pb.finish();
  println!("All done!");
}
