use axum::extract::State;
use axum::routing::get;
use axum::routing::post;
use axum::Json;
use axum::Router;
use clap::Args;
use half::f16;
use libroxanne::Roxanne;
use ndarray::Array1;
use serde::Deserialize;
use serde::Serialize;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Args)]
pub struct ServeArgs {
  /// Path to a Roxanne DB.
  #[arg()]
  path: PathBuf,

  /// Port to listen on.
  #[arg(long, default_value = "4224")]
  port: u16,

  /// Address to bind to.
  #[arg(long, default_value = "IpAddr::from([127, 0, 0, 1])")]
  addr: IpAddr,
}

struct Ctx {
  db: Arc<Roxanne>,
}

#[derive(Deserialize)]
struct PostInsertReqVector {
  key: String,
  vector: Vec<f16>,
}

#[derive(Deserialize)]
struct PostInsertReq {
  vectors: Vec<PostInsertReqVector>,
}

async fn handle_post_insert(State(ctx): State<Arc<Ctx>>, Json(req): Json<PostInsertReq>) {
  ctx
    .db
    .insert(
      req
        .vectors
        .into_iter()
        .map(|v| (v.key, Array1::from(v.vector))),
    )
    .await;
}

#[derive(Deserialize)]
struct PostQueryReq {
  vector: Vec<f16>,
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
  let results = ctx.db.query(&Array1::from(req.vector).view(), req.k).await;
  Json(PostQueryRes { results })
}

impl ServeArgs {
  pub async fn exec(self) {
    let db = Roxanne::open(self.path).await;

    let ctx = Ctx { db };

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
