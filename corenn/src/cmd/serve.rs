use axum::extract::State;
use axum::routing::get;
use axum::routing::post;
use axum::Json;
use axum::Router;
use clap::Args;
use clap::ValueEnum;
use half::bf16;
use half::f16;
use libcorenn::vec::VecData;
use libcorenn::CoreNN;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Dtype {
  BF16,
  F16,
  F32,
}

#[derive(Args)]
pub struct ServeArgs {
  /// Path to a CoreNN DB.
  #[arg()]
  path: PathBuf,

  /// Port to listen on.
  #[arg(long, default_value = "4224")]
  port: u16,

  /// Address to bind to.
  #[arg(long, default_value = "IpAddr::from([127, 0, 0, 1])")]
  addr: IpAddr,

  /// Dtype of the vectors.
  #[arg(long, default_value = "F32")]
  dtype: Dtype,
}

struct Ctx {
  db: Arc<CoreNN>,
  dtype: Dtype,
}

impl Ctx {
  pub fn vec(&self, vector: Vec<f32>) -> VecData {
    match self.dtype {
      Dtype::BF16 => VecData::BF16(vector.into_iter().map(bf16::from_f32).collect()),
      Dtype::F16 => VecData::F16(vector.into_iter().map(f16::from_f32).collect()),
      Dtype::F32 => VecData::F32(vector),
    }
  }
}

#[derive(Deserialize)]
struct PostInsertReqVector {
  key: String,
  vector: Vec<f32>,
}

#[derive(Deserialize)]
struct PostInsertReq {
  vectors: Vec<PostInsertReqVector>,
}

async fn handle_post_insert(State(ctx): State<Arc<Ctx>>, Json(req): Json<PostInsertReq>) {
  req.vectors.into_par_iter().for_each(|v| {
    let vec = ctx.vec(v.vector);
    ctx.db.insert_vec(&v.key, vec);
  });
}

#[derive(Deserialize)]
struct PostQueryReq {
  vector: Vec<f32>,
  k: usize,
}

#[derive(Serialize)]
struct PostQueryRes {
  results: Vec<(String, f32)>,
}

async fn handle_post_query(
  State(ctx): State<Arc<Ctx>>,
  Json(req): Json<PostQueryReq>,
) -> Json<PostQueryRes> {
  let vec = ctx.vec(req.vector);
  let results = ctx.db.query_vec(vec, req.k);
  Json(PostQueryRes { results })
}

impl ServeArgs {
  pub async fn exec(self) {
    let db = Arc::new(CoreNN::open(self.path));

    let ctx = Ctx {
      db,
      dtype: self.dtype,
    };

    let app = Router::new()
      .route("/healthz", get(|| async { "OK " }))
      .route("/insert", post(handle_post_insert))
      .route("/query", post(handle_post_query))
      .with_state(Arc::new(ctx));

    let listener = tokio::net::TcpListener::bind((self.addr, self.port))
      .await
      .unwrap();
    tracing::info!(
      addr = self.addr.to_string(),
      port = self.port,
      "server started"
    );
    axum::serve(listener, app).await.unwrap();
  }
}
