use ahash::HashSet;
use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::common::metric_euclidean;
use libroxanne::common::Id;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use ndarray::Array1;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::mem::size_of;

// This program takes a dataset from http://corpus-texmex.irisa.fr/ and calculates the recall accuracy.

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
  /// Path prefix of files base.fvecs, groundtruth.ivecs, and query.fvecs.
  #[arg()]
  path: String,

  #[arg(long, default_value_t = 1)]
  beam_width: usize,

  #[arg(long)]
  degree_bound: usize,

  #[arg(long, default_value_t = 1.1)]
  distance_threshold: f64,

  #[arg(long, default_value_t = 64)]
  insert_batch_size: usize,

  #[arg(long, default_value_t = 1.33)]
  search_list_cap_mul: f64,
}

fn read_vectors<T: Copy + Default + Send, Reader: Fn(&[u8], &mut [T]) + Sync>(
  path: &str,
  reader: Reader,
) -> Vec<(Id, Array1<T>)> {
  let raw = std::fs::read(path).unwrap();
  let dims = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;

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
      (i, Array1::from_vec(vec))
    })
    .collect::<Vec<_>>()
}

fn main() {
  let args = Args::parse();

  let vecs = read_vectors(
    &format!("{}base.fvecs", args.path),
    LittleEndian::read_f32_into,
  );
  let qs = read_vectors(
    &format!("{}query.fvecs", args.path),
    LittleEndian::read_f32_into,
  );
  let knns = read_vectors(
    &format!("{}groundtruth.ivecs", args.path),
    LittleEndian::read_u32_into,
  );
  let dims = vecs[0].1.len();
  let k = knns[0].1.len();
  let qn = qs.len();
  assert_eq!(qs[0].1.len(), dims);
  assert_eq!(knns.len(), qn);

  let params = VamanaParams {
    beam_width: args.beam_width,
    degree_bound: args.degree_bound,
    distance_threshold: args.distance_threshold,
    insert_batch_size: args.insert_batch_size,
    medoid_sample_size: 10_000,
    search_list_cap: (k as f64 * args.search_list_cap_mul) as usize,
  };
  println!("Params: {params:?}");

  let graph = InMemoryVamana::build_index(vecs, metric_euclidean, params, None);
  println!("Indexed");

  let correct: usize = qs
    .into_par_iter()
    .zip(knns)
    .map(|((_, vec), (_, knn_expected))| {
      let knn_expected = HashSet::from_iter(knn_expected.mapv(|v| v as Id));
      let knn_got = graph
        .query(&vec.view(), k)
        .into_iter()
        .map(|pd| pd.id)
        .collect::<HashSet<_>>();
      knn_expected.intersection(&knn_got).count()
    })
    .sum();

  println!(
    "Correct: {:.2}% ({}/{})",
    correct as f64 / (qn * k) as f64 * 100.0,
    correct,
    qn * k
  );
}
