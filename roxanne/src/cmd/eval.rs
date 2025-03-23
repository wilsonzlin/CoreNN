use crate::new_pb_with_msg;
use ahash::HashSet;
use bytemuck::cast_slice;
use clap::Args;
use clap::ValueEnum;
use half::f16;
use libroxanne::util::AsyncConcurrentIteratorExt;
use libroxanne::util::AtomUsz;
use libroxanne::Roxanne;
use ndarray::Array1;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::read;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Dtype {
  F16,
  F32,
}

#[derive(Args)]
pub struct EvalArgs {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Data type of provided insert and query vectors.
  #[arg(long, short = 't', value_enum, default_value_t = Dtype::F16)]
  dtype: Dtype,

  /// Dimension of vectors. Only provide this for `vectors`.
  #[arg(long, short)]
  dim: Option<usize>,

  /// Optionally insert vectors from a packed matrix where each row is a vector. Row numbers are used as IDs.
  #[arg(long, short)]
  vectors: Option<PathBuf>,

  /// Path to packed matrix where each row is a query vector.
  #[arg(long, short)]
  queries: PathBuf,

  /// Number of nearest neighbors to query for.
  #[arg(long, short)]
  k: usize,

  /// Path to packed u32 matrix representing IDs of the `k` nearest neighbors for each query. (For now, IDs must be integers.)
  #[arg(long, short)]
  results: PathBuf,
}

async fn read_vecs(path: &PathBuf, dtype: Dtype, dim: usize) -> Vec<Array1<f16>> {
  let raw = read(path).await.unwrap();
  let dtype_size = match dtype {
    Dtype::F16 => size_of::<f16>(),
    Dtype::F32 => size_of::<f32>(),
  };
  raw.chunks(dtype_size * dim)
    .map(|raw_vec| {
      let raw: Vec<f16> = match dtype {
        Dtype::F16 => cast_slice(raw_vec).to_vec(),
        Dtype::F32 => {
          let raw_f32: &[f32] = cast_slice(raw_vec);
          raw_f32.iter().map(|&x| f16::from_f32(x)).collect()
        }
      };
      Array1::from_vec(raw)
    })
    .collect()
}

impl EvalArgs {
  pub async fn exec(self) {
    let rox = Arc::new(Roxanne::open(self.path).await);
    tracing::info!("loaded database");

    if let Some(vectors_path) = &self.vectors {
      let vectors: Vec<Array1<f16>> = read_vecs(vectors_path, self.dtype, self.dim.unwrap()).await;
      let n = vectors.len();

      rox.insert(
        vectors.into_iter().enumerate().map(|(id, v)| (id.to_string(), v)),
      ).await;
      tracing::info!(n, "inserted vectors");
    };

    let queries: Vec<Array1<f16>> = read_vecs(&self.queries, self.dtype, rox.dim()).await;
    tracing::info!(n = queries.len(), "loaded query vectors");

    let knn_ids: Vec<HashSet<u32>> = {
      let raw = read(&self.results).await.unwrap();
      let raw_vec_len = size_of::<u32>() * self.k;
      assert_eq!(raw.len(), raw_vec_len * queries.len());
      raw
        .chunks(raw_vec_len)
        .map(|raw_vec| cast_slice(raw_vec).iter().copied().collect())
        .collect()
    };
    tracing::info!("loaded k-NN");

    let pb = new_pb_with_msg(queries.len());
    let correct = Arc::new(AtomUsz::new(0));
    let total = Arc::new(AtomUsz::new(0));
    zip(queries, knn_ids)
      .map(|(q, k)| {
        (
          rox.clone(),
          pb.clone(),
          correct.clone(),
          total.clone(),
          q,
          k,
        )
      })
      .spawn_for_each(
        |(rox, pb, correct, total, query, knn_expected)| async move {
          let k = knn_expected.len();
          let knn_got: HashSet<u32> = rox
            .query(&query.view(), k)
            .await
            .into_iter()
            .map(|e| e.0.parse().unwrap())
            .collect();

          let correct = correct.inc(knn_expected.intersection(&knn_got).count());
          let total = total.inc(k);
          let pc = correct as f64 / total as f64 * 100.0;
          pb.set_message(format!("Correct: {pc:.2}% ({correct}/{total})"));
          pb.inc(1);
        },
      )
      .await;
    pb.finish();

    let correct = correct.get();
    let total = total.get();
    let pc = correct as f64 / total as f64 * 100.0;
    tracing::info!(correct, total, correct_percent = pc, "all done!");
  }
}
