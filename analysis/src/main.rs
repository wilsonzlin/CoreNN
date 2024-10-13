use byteorder::ByteOrder;
use byteorder::LittleEndian;
use clap::Parser;
use libroxanne::vamana::InMemoryVamana;
use libroxanne::vamana::VamanaParams;
use ndarray::Array1;
use ndarray::ArrayView1;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
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
}

fn main() {
  let args = Args::parse();

  let raw = std::fs::read(args.vecs).unwrap();
  println!("Read vectors: complete");

  let bytes_per_vec = 4 + args.dims * 4;
  assert_eq!(raw.len() % bytes_per_vec, 0);
  let n = raw.len() / bytes_per_vec;

  let vecs = (0..n)
    .into_par_iter()
    .map(|i| {
      let start = i * bytes_per_vec;
      let end = (i + 1) * bytes_per_vec;
      let raw = &raw[start..end];
      let (dims_raw, raw) = raw.split_at(4);
      let dims = u32::from_le_bytes(dims_raw.try_into().unwrap()) as usize;
      assert_eq!(dims, args.dims);
      let mut vec = vec![0.0f32; dims];
      LittleEndian::read_f32_into(raw, &mut vec);
      (i as u32, Array1::from_vec(vec))
    })
    .collect::<Vec<_>>();
  drop(raw);
  println!("Load vectors: complete");

  fn euclidian(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f64 {
    let diff = a.mapv(|x| x as f64) - b.mapv(|x| x as f64);
    (&diff * &diff).sum().sqrt()
  }

  let params = VamanaParams {
    medoid_sample_size: 10_000,
    beam_width: 10,
    distance_threshold: 1.1,
    degree_bound: f64::ln(n as f64) as usize,
  };

  let graph = InMemoryVamana::init(vecs, euclidian, params);
}
