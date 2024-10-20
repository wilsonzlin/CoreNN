use ahash::AHashSet;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use ndarray::Array1;
use ndarray::ArrayView1;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(long)]
  dims: usize,

  /// Path to the matrix of f32 vectors.
  #[arg(long)]
  vecs: PathBuf,

  /// Path to the matrix of f32 vectors.
  #[arg(long)]
  qs: PathBuf,

  /// Path to the matrix of k-NN vectors.
  #[arg(long)]
  knns: PathBuf,

  #[arg(long)]
  k: usize,
}

fn read_vectors<T: Copy + Default + Send, Reader: Fn(&[u8], &mut [T]) + Sync>(
  path: &Path,
  dims: usize,
  reader: Reader,
) -> Vec<(u32, Array1<T>)> {
  let raw = std::fs::read(path).unwrap();

  let bytes_per_vec = 4 + dims * size_of::<T>();
  assert_eq!(raw.len() % bytes_per_vec, 0);
  let n = raw.len() / bytes_per_vec;

  (0..n)
    .into_par_iter()
    .map(|i| {
      let start = i * bytes_per_vec;
      let end = (i + 1) * bytes_per_vec;
      let raw = &raw[start..end];
      let (dims_raw, raw) = raw.split_at(4);
      assert_eq!(
        u32::from_le_bytes(dims_raw.try_into().unwrap()) as usize,
        dims
      );
      let mut vec = vec![T::default(); dims];
      reader(raw, vec.as_mut_slice());
      (i as u32, Array1::from_vec(vec))
    })
    .collect::<Vec<_>>()
}

fn main() {
  let args = Args::parse();

  let vecs = read_vectors(&args.vecs, args.dims, LittleEndian::read_f32_into);
  let qs = read_vectors(&args.qs, args.dims, LittleEndian::read_f32_into);
  let knns = read_vectors(&args.knns, args.k, LittleEndian::read_u32_into);
  let qn = qs.len();
  assert_eq!(knns.len(), qn);

  fn euclidian(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f64 {
    let diff = a.mapv(|x| x as f64) - b.mapv(|x| x as f64);
    (&diff * &diff).sum().sqrt()
  }

  let params = VamanaParams {
    degree_bound: (vecs.len() as f64).ln() as usize,
    distance_threshold: 1.1,
    insert_batch_size: 64,
    medoid_sample_size: 10_000,
    search_list_cap: (args.k as f64 * 1.33) as usize,
  };

  let graph = InMemoryVamana::init(vecs, euclidian, params);

  let correct: usize = qs
    .into_par_iter()
    .zip(knns)
    .map(|((_, vec), (_, knn_expected))| {
      let knn_expected = AHashSet::from_iter(knn_expected);
      let knn_got = graph
        .query(&vec.view(), args.k)
        .into_iter()
        .map(|pd| pd.id)
        .collect::<AHashSet<_>>();
      knn_expected.intersection(&knn_got).count()
    })
    .sum();

  println!(
    "Correct: {:.2}% ({}/{})",
    correct as f64 / (qn * args.k) as f64 * 100.0,
    correct,
    qn * args.k
  );
}
