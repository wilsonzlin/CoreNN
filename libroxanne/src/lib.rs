#![feature(async_closure)]
#![feature(duration_millis_float)]
#![feature(path_add_extension)]
#![warn(clippy::future_not_send)]
#![allow(async_fn_in_trait)]

use crate::util::AsyncConcurrentIteratorExt;
use bf::BruteForceIndex;
use blob::BlobStore;
use cfg::RoxanneDbCfg;
use common::nan_to_num;
use common::to_calc;
use common::Dtype;
use common::DtypeCalc;
use common::Id;
use common::Metric;
use common::PrecomputedDists;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use dashmap::DashSet;
use db::Db;
use db::DbIndexMode;
use flume::Sender;
use futures::StreamExt;
use ndarray::Array1;
use ndarray::ArrayView1;
use parking_lot::Mutex;
use parking_lot::RwLock;
use pq::ProductQuantizer;
use search::GreedySearchParams;
use search::GreedySearchable;
use search::GreedySearchableAsync;
use search::INeighbors;
use search::IPoint;
use search::Points;
use search::Query;
use signal_future::SignalFuture;
use std::cmp::max;
use std::iter::zip;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use updater::updater_thread;
use updater::Update;
use util::ArcMap;
use vamana::Vamana;
use vamana::VamanaParams;

pub mod bf;
pub mod blob;
pub mod cfg;
pub mod common;
pub mod db;
pub mod hnsw;
pub mod in_memory;
pub mod pq;
pub mod queue;
pub mod search;
pub mod updater;
pub mod util;
pub mod vamana;

pub struct RoxanneDbDir {
  root: PathBuf,
}

impl RoxanneDbDir {
  pub fn new(root: PathBuf) -> Self {
    Self { root }
  }

  pub fn blobs(&self) -> PathBuf {
    self.root.join("blobs")
  }

  pub fn db(&self) -> PathBuf {
    self.root.join("db")
  }

  pub fn cfg(&self) -> PathBuf {
    self.root.join("roxanne.toml")
  }
}

#[derive(Debug)]
pub enum RoxanneDbError {
  Busy,
}

enum Mode<T: Dtype, C: DtypeCalc> {
  InMemory {
    graph: Arc<DashMap<Id, Vec<Id>>>,
    vectors: Arc<DashMap<Id, Array1<T>>>,
  },
  LTI {
    pq_vecs: DashMap<Id, Array1<u8>>,
    pq: ProductQuantizer<C>,
  },
}

struct Index<T: Dtype, C: DtypeCalc> {
  // We use a brute force index for the first few points, because:
  // - we want a decent sized sample to calculate the medoid
  // - it's fast enough to query
  // - inserting into a Vamana graph with zero or only a few nodes has complications (e.g. RobustPrune may actually make a node unreachable)
  bf: RwLock<Option<Arc<BruteForceIndex<T, C>>>>,
  db: Arc<Db>,
  mode: RwLock<Mode<T, C>>,
  // This changes when we transition from BF to InMemory.
  medoid: AtomicUsize,
  metric: Metric<C>,
  vamana_params: VamanaParams,
  additional_out_neighbors: DashMap<Id, Vec<Id>>,
  additional_edge_count: AtomicUsize,
  // Only used for a temporary short period during updater_thread when inserting new vectors.
  temp_nodes: DashMap<Id, (Vec<Id>, Array1<T>)>,
}

enum IndexPoint<'a, T, C> {
  LTI(Array1<C>),
  Temp(Ref<'a, Id, (Vec<Id>, Array1<T>)>),
  // I'd prefer to use MappedRwLockReadGuard but we're blocked on https://github.com/Amanieu/parking_lot/issues/289.
  InMemory(ArcMap<DashMap<Id, Array1<T>>, Ref<'a, Id, Array1<T>>>),
}

impl<'a, T: Dtype, C: DtypeCalc> IPoint<T, C> for IndexPoint<'a, T, C> {
  fn point(&self) -> ArrayView1<T> {
    match self {
      IndexPoint::InMemory(v) => v.value().view(),
      IndexPoint::LTI(_) => panic!("tried to get Dtype view of PQ decoded DtypeCalc vector"),
      IndexPoint::Temp(v) => v.value().1.view(),
    }
  }

  fn into_calc(self) -> Array1<C> {
    if let IndexPoint::LTI(v) = self {
      v
    } else {
      to_calc(&self.point())
    }
  }
}

enum IndexNeighborsBase<'a, T> {
  Temp(Ref<'a, Id, (Vec<Id>, Array1<T>)>),
  // I'd prefer to use MappedRwLockReadGuard but we're blocked on https://github.com/Amanieu/parking_lot/issues/289.
  InMemory(ArcMap<DashMap<Id, Vec<Id>>, Ref<'a, Id, Vec<Id>>>),
  LTI(Vec<Id>),
}

struct IndexNeighbors<'a, T> {
  base: IndexNeighborsBase<'a, T>,
  add: Option<Ref<'a, Id, Vec<Id>>>,
}

impl<'a, T> IndexNeighbors<'a, T> {
  fn iter_add(&self) -> impl Iterator<Item = Id> + '_ {
    match &self.add {
      Some(e) => e.value().as_slice(),
      None => &[],
    }
    .iter()
    .map(|e| *e)
  }
}

impl<'a, T> INeighbors for IndexNeighbors<'a, T> {
  fn neighbors(&self) -> impl Iterator<Item = Id> {
    match &self.base {
      IndexNeighborsBase::InMemory(e) => e.value().as_slice(),
      IndexNeighborsBase::LTI(v) => v.as_slice(),
      IndexNeighborsBase::Temp(e) => e.value().0.as_slice(),
    }
    .iter()
    .map(|e| *e)
    .chain(self.iter_add())
  }
}

impl<T: Dtype, C: DtypeCalc> Points<T, C> for Index<T, C> {
  type Point<'a> = IndexPoint<'a, T, C>;

  fn metric(&self) -> Metric<C> {
    self.metric
  }

  fn get_point<'a>(&'a self, id: Id) -> Self::Point<'a> {
    if let Some(temp) = self.temp_nodes.get(&id) {
      return IndexPoint::Temp(temp);
    }
    match &*self.mode.read() {
      Mode::InMemory { vectors, .. } => {
        IndexPoint::InMemory(ArcMap::map(vectors.clone(), |v| v.get(&id).unwrap()))
      }
      Mode::LTI { pq, pq_vecs } => IndexPoint::LTI({
        let pq_vec = pq_vecs.get(&id).unwrap();
        pq.decode_1(&pq_vec.view())
      }),
    }
  }

  fn precomputed_dists(&self) -> Option<&PrecomputedDists> {
    None
  }
}

impl<T: Dtype, C: DtypeCalc> GreedySearchable<T, C> for Index<T, C> {
  type FullVec = Array1<T>;
  type Neighbors<'a> = IndexNeighbors<'a, T>;

  fn medoid(&self) -> Id {
    self.medoid.load(Ordering::Relaxed)
  }
}

impl<T: Dtype, C: DtypeCalc> GreedySearchableAsync<T, C> for Index<T, C> {
  async fn get_out_neighbors_async<'a>(
    &'a self,
    id: Id,
  ) -> (Self::Neighbors<'a>, Option<Self::FullVec>) {
    if let Some(temp) = self.temp_nodes.get(&id) {
      // We don't need to add additional_out_neighbors, as temp_nodes is a full override.
      // We don't need to return a full vector as it's not from the disk.
      return (
        IndexNeighbors {
          base: IndexNeighborsBase::Temp(temp),
          add: None,
        },
        None,
      );
    }
    // Don't use `match` as we can't hold lock across await in Mode::LTI branch.
    if let Mode::InMemory { graph, .. } = &*self.mode.read() {
      return (
        IndexNeighbors {
          base: IndexNeighborsBase::InMemory(ArcMap::map(Arc::clone(&graph), |g| {
            g.get(&id).unwrap()
          })),
          add: self.additional_out_neighbors.get(&id),
        },
        None,
      );
    };
    let n = self.db.read_node(id).await;
    (
      IndexNeighbors {
        base: IndexNeighborsBase::LTI(n.neighbors),
        add: self.additional_out_neighbors.get(&id),
      },
      Some(Array1::from(n.vector)),
    )
  }
}

// We only implement Vamana to be able to use compute_robust_pruned during updater_thread.
// We don't impl VamanaSync and use set_* as the update process is more sophisticated (see updater_thread).
impl<T: Dtype, C: DtypeCalc> Vamana<T, C> for Index<T, C> {
  fn params(&self) -> &vamana::VamanaParams {
    &self.vamana_params
  }
}

pub struct RoxanneDb<T: Dtype, C: DtypeCalc> {
  blobs: BlobStore,
  cfg: RoxanneDbCfg,
  db: Arc<Db>,
  deleted: DashSet<Id>,
  index: Index<T, C>,
  update_sender: Sender<Update<T>>,
}

impl<T: Dtype, C: DtypeCalc> RoxanneDb<T, C> {
  pub async fn open(dir: PathBuf) -> Arc<RoxanneDb<T, C>> {
    let dir = RoxanneDbDir::new(dir);

    let blobs = BlobStore::open(dir.blobs()).await;
    let cfg_raw = tokio::fs::read_to_string(dir.cfg()).await.unwrap();
    let cfg: RoxanneDbCfg = toml::from_str(&cfg_raw).unwrap();
    let metric = cfg.metric.get_fn::<C>();
    let vamana_params = VamanaParams {
      degree_bound: cfg.degree_bound,
      distance_threshold: cfg.distance_threshold,
      update_batch_size: cfg.update_batch_size,
      update_search_list_cap: cfg.update_search_list_cap,
    };

    let db = Arc::new(Db::open(dir.db()).await);
    let next_id = Mutex::new(0);
    let deleted = DashSet::new();
    db.iter_deleted()
      .for_each(async |id| {
        deleted.insert(id);
        let mut next_id = next_id.lock();
        *next_id = max(id + 1, *next_id);
      })
      .await;
    db.iter_ids()
      .for_each(async |(id, _)| {
        let mut next_id = next_id.lock();
        *next_id = max(id + 1, *next_id);
      })
      .await;
    let next_id = next_id.into_inner();

    let additional_out_neighbors = DashMap::new();
    let additional_edge_count = AtomicUsize::new(0);
    db.iter_additional_out_neighbors()
      .for_each(async |(id, add)| {
        additional_edge_count.fetch_add(add.len(), Ordering::Relaxed);
        additional_out_neighbors.insert(id, add);
      })
      .await;

    let mode = db.read_index_mode().await;

    let index = Index {
      bf: RwLock::new(if mode == DbIndexMode::BruteForce {
        let bf = BruteForceIndex::new(metric);
        db.iter_nodes::<T>()
          .for_each(async |(id, n)| {
            bf.insert(id, Array1::from_vec(n.vector));
          })
          .await;
        Some(Arc::new(bf))
      } else {
        None
      }),
      db: db.clone(),
      // In BruteForce mode (e.g. init), there is no medoid.
      medoid: db.maybe_read_medoid().await.unwrap_or(Id::MAX).into(),
      metric,
      vamana_params,
      additional_out_neighbors,
      additional_edge_count,
      temp_nodes: DashMap::new(),
      mode: RwLock::new(if mode == DbIndexMode::LongTerm {
        let pq = blobs.read_pq_model().await;
        // TODO Build pq_vecs from existing nodes.
        Mode::LTI {
          pq,
          pq_vecs: DashMap::new(),
        }
      } else {
        // This also handles BruteForce mode (this'll be empty.)
        let graph = Arc::new(DashMap::new());
        let vectors = Arc::new(DashMap::new());
        db.iter_nodes()
          .for_each(async |(id, n)| {
            graph.insert(id, n.neighbors);
            vectors.insert(id, Array1::from_vec(n.vector));
          })
          .await;
        Mode::InMemory { graph, vectors }
      }),
    };

    let (send, recv) = flume::unbounded();
    let roxanne = Arc::new(RoxanneDb {
      blobs,
      cfg,
      db,
      deleted,
      index,
      update_sender: send,
    });
    tokio::spawn({
      let roxanne = roxanne.clone();
      async move {
        updater_thread(roxanne, recv, next_id).await;
      }
    });
    roxanne
  }

  pub async fn query<'a>(&'a self, query: &'a ArrayView1<'a, T>, k: usize) -> Vec<(String, f64)> {
    // We must (cheaply) clone to avoid holding lock across await.
    let bf = self.index.bf.read().clone();
    let res = if let Some(bf) = bf {
      // Don't run this in spawn_blocking; it's not I/O, it's CPU bound, and moving to a separate thread doesn't unblock that CPU that would be used.
      bf.query(query, k, |n| !self.deleted.contains(&n))
    } else {
      self
        .index
        .greedy_search_async(GreedySearchParams {
          query: Query::Vec(query),
          k,
          search_list_cap: self.cfg.query_search_list_cap,
          beam_width: self.cfg.beam_width,
          start: self.index.medoid(),
          filter: |n| !self.deleted.contains(&n),
          out_visited: None,
          out_metrics: None,
          ground_truth: None,
        })
        .await
    };
    let keys = res
      .iter()
      .map_concurrent(|r| self.db.maybe_read_key(r.id))
      .collect::<Vec<_>>()
      .await;
    zip(keys, res)
      // A node may have been deleted during the query (already collected, so didn't get filtered), or literally just after the end of the query but before here.
      // TODO DOCUMENT: it's possible to get less than k for the above reason.
      .filter_map(|(k, r)| k.map(|k| (k, r.dist)))
      .collect()
  }

  // Considerations:
  // - We don't want to keep piling on to a brute force index (e.g. while compactor struggles to keep up) and degrade query performance exponentially.
  // - We don't want to keep buffering vectors in memory, acknowledging inserts but in reality there's backpressure while the graph is being updated.
  // - Both above points means that we should have some mechanism of returning a "busy" signal to the caller, instead of just accepting into unbounded memory or blocking the caller.
  // - We don't want to perform any compaction within the function (instead running it in a background thread), as otherwise that negatively affects the insertion call latency for the one unlucky caller.
  // Require a Map to ensure there are no duplicate keys (which would otherwise complicate insertion).
  pub async fn insert(
    &self,
    entries: impl IntoIterator<Item = (String, Array1<T>)>,
  ) -> Result<(), RoxanneDbError> {
    let (signal, ctl) = SignalFuture::new();
    self
      .update_sender
      .send_async(Update::Insert(
        entries
          .into_iter()
          // NaN values cause infinite loops while PQ training and vector querying, amongst other things. This replaces NaN values with 0 and +/- infinity with min/max finite values.
          .map(|(k, v)| (k, v.mapv(nan_to_num)))
          .collect(),
        ctl,
      ))
      .await
      .unwrap();
    signal.await;
    Ok(())
  }

  pub async fn delete(&self, key: &str) -> Result<(), RoxanneDbError> {
    let (signal, ctl) = SignalFuture::new();
    self
      .update_sender
      .send_async(Update::Delete(key.to_string(), ctl))
      .await
      .unwrap();
    signal.await;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use crate::cfg::RoxanneDbCfg;
  use crate::common::metric_euclidean;
  use crate::common::StdMetric;
  use crate::db::DbIndexMode;
  use crate::in_memory::calc_approx_medoid;
  use crate::util::AsyncConcurrentIteratorExt;
  use crate::util::AsyncConcurrentStreamExt;
  use crate::Mode;
  use crate::RoxanneDb;
  use crate::RoxanneDbDir;
  use ahash::HashSet;
  use dashmap::DashSet;
  use futures::stream::iter;
  use futures::StreamExt;
  use itertools::Itertools;
  use ndarray::Array1;
  use ordered_float::OrderedFloat;
  use rand::seq::IteratorRandom;
  use rand::thread_rng;
  use rand::Rng;
  use rayon::iter::IntoParallelIterator;
  use rayon::iter::ParallelIterator;
  use std::fs;
  use std::iter::once;
  use std::path::PathBuf;
  use std::sync::atomic::Ordering;
  use std::thread::sleep;
  use std::time::Duration;
  use std::time::Instant;

  // Unlike tokio::main, tokio::test by default uses only one thread: https://docs.rs/tokio/latest/tokio/attr.test.html#multi-threaded-runtime.
  #[tokio::test(flavor = "multi_thread")]
  async fn test_roxanne() {
    tracing_subscriber::fmt()
      .with_max_level(tracing::Level::DEBUG)
      .init();

    // Parameters.
    let dir_raw = PathBuf::from("/dev/shm/roxannedb-test");
    let dir = RoxanneDbDir::new(dir_raw.clone());
    let dim = 512;
    let degree_bound = 40;
    let max_degree_bound = 80;
    let search_list_cap = 100;
    let pq_subspaces = 256;
    let gen = || {
      Array1::from_vec(
        (0..dim)
          .map(|_| thread_rng().gen_range(-1.0..1.0))
          .collect_vec(),
      )
    };
    let n = 10_000;
    let vecs = (0..n).map(|_| gen()).collect_vec();
    let nn = (0..n)
      .into_par_iter()
      .map(|i| {
        (0..n)
          .map(|j| (j, metric_euclidean(&vecs[i].view(), &vecs[j].view())))
          .sorted_unstable_by_key(|e| OrderedFloat(e.1))
          .map(|e| e.0)
          .collect_vec()
      })
      .collect::<Vec<_>>();
    tracing::info!("calculated nearest neighbors");

    // Create and open DB.
    if fs::exists(&dir_raw).unwrap() {
      fs::remove_dir_all(&dir_raw).unwrap();
    }
    fs::create_dir(&dir_raw).unwrap();
    fs::write(
      dir.cfg(),
      toml::to_string(&RoxanneDbCfg {
        brute_force_index_cap: 1000,
        degree_bound,
        dim,
        in_memory_index_cap: 2500,
        max_degree_bound,
        merge_threshold_deletes: 200,
        metric: StdMetric::L2,
        query_search_list_cap: search_list_cap,
        update_batch_size: num_cpus::get(),
        update_search_list_cap: search_list_cap,
        pq_subspaces,
        ..Default::default()
      })
      .unwrap(),
    )
    .unwrap();
    let rx = RoxanneDb::open(dir_raw).await;
    tracing::info!("opened database");

    // First test: an empty DB should provide no results.
    let res = rx.query(&gen().view(), 100).await;
    assert_eq!(res.len(), 0);

    let deleted = DashSet::new();
    macro_rules! assert_accuracy {
      ($below_i:expr, $exp_acc:expr) => {{
        tracing::debug!(n = $below_i, "running queries");
        let mode = rx.db.read_index_mode().await;
        let start = Instant::now();
        let mut correct = 0;
        let mut total = 0;
        let results = (0..$below_i)
          .map_concurrent(async |i| {
            tokio::spawn({
              let rx = rx.clone();
              let vec = vecs[i].clone();
              async move { rx.query(&vec.view(), 100).await }
            })
            .await
            .unwrap()
          })
          .collect_vec()
          .await;
        for (i, res) in results.into_iter().enumerate() {
          let got = res
            .iter()
            .map(|e| e.0.parse::<usize>().unwrap())
            .collect::<HashSet<_>>();
          let want = nn[i]
            .iter()
            .cloned()
            .filter(|&n| n < $below_i && !deleted.contains(&n))
            .take(100)
            .collect::<HashSet<_>>();
          total += want.len();
          correct += want.intersection(&got).count();
        }
        let accuracy = correct as f64 / total as f64;
        let exec_ms = start.elapsed().as_millis_f64();
        tracing::info!(
          mode = format!("{:?}", mode),
          accuracy,
          qps = $below_i as f64 / exec_ms * 1000.0,
          "accuracy"
        );
        assert!(accuracy >= $exp_acc);
      }};
    }

    // Insert below brute force index cap.
    rx.insert([("0".to_string(), vecs[0].clone())])
      .await
      .unwrap();
    let res = rx.query(&vecs[0].view(), 100).await;
    assert_eq!(res.len(), 1);
    assert_eq!(&res[0].0, "0");

    // Still below brute force index cap.
    rx.insert((1..734).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    // It's tempting to directly compare against `nn[i]` given BF's 100% accuracy, but it's still complicated due to 1) non-deterministic DB internal IDs assigned, and 2) unstable sorting (i.e. two same dists).
    // Accuracy can be very slightly less than 1.0 due to non-deterministic sorting of equal dist points.
    assert_accuracy!(734, 0.99);
    tracing::info!("inserted 734 so far");

    // Delete some keys.
    // Deleting non-existent keys should do nothing.
    for i in [5, 90, 734, 735, 900, 10002] {
      rx.delete(&i.to_string()).await.unwrap();
    }
    assert_eq!(rx.deleted.len(), 2);
    deleted.insert(5);
    deleted.insert(90);

    // Still below brute force index cap.
    rx.insert((734..1000).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    // Accuracy can be very slightly less than 1.0 due to non-deterministic sorting of equal dist points.
    assert_accuracy!(1000, 0.99);
    tracing::info!("inserted 1000 so far");
    // Ensure the updater_thread isn't doing anything.
    sleep(Duration::from_secs(1));

    assert_eq!(rx.deleted.len(), 2);
    assert_eq!(rx.index.bf.read().clone().unwrap().len(), 1000);
    assert_eq!(rx.index.additional_out_neighbors.len(), 0);
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 0);
        assert_eq!(vectors.len(), 0);
      }
      Mode::LTI { .. } => unreachable!(),
    };

    // Now, it should build the in-memory index.
    rx.insert((1000..1313).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    tracing::info!("inserted 1313 so far");
    let expected_medoid_i = calc_approx_medoid(
      &(0..1313).map(|i| (i, vecs[i].clone())).collect(),
      metric_euclidean,
      10_000,
      None,
    );
    let expected_medoid_key = expected_medoid_i.to_string();
    tracing::info!("calculated expected medoid");
    let expected_medoid_id = rx.db.maybe_read_id(&expected_medoid_key).await.unwrap();
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.additional_out_neighbors.len(), 0);
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 1313);
        assert_eq!(vectors.len(), 1313);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(rx.db.maybe_read_medoid().await, Some(expected_medoid_id));
    assert_eq!(rx.db.read_index_mode().await, DbIndexMode::InMemory);
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 1313);
    let missing_nodes = rx
      .db
      .iter_nodes::<f32>()
      .filter_map(async |(internal_id, n)| {
        // We don't know the internal ID so we can't just skip if it matches our ID.
        let Some(key) = rx.db.maybe_read_key(internal_id).await else {
          return Some(());
        };
        let i = key.parse::<usize>().unwrap();
        assert_eq!(n.vector, vecs[i].to_vec());
        None
      })
      .count()
      .await;
    assert_eq!(missing_nodes, deleted.len());

    // Delete some keys.
    // Yes, some of these will be duplicates, which we want to also test.
    // Also delete the medoid. Medoids are never permanently deleted, which will be important later once we test merging.
    let to_delete = (0..40)
      .map(|_| thread_rng().gen_range(0..1313))
      .chain(once(expected_medoid_i))
      .collect::<Vec<_>>();
    iter(&to_delete)
      .for_each_concurrent(None, async |&i| {
        rx.delete(&i.to_string()).await.unwrap();
        deleted.insert(i); // Handles already deleted, duplicate delete requests.
      })
      .await;
    assert_eq!(rx.deleted.len(), deleted.len());

    // Still under in-memory index cap.
    rx.insert((1313..2090).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    tracing::info!("inserted 2090 so far");
    // Medoid must not have changed.
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { graph, vectors } => {
        assert_eq!(graph.len(), 2090);
        assert_eq!(vectors.len(), 2090);
      }
      Mode::LTI { .. } => unreachable!(),
    };
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 2090);
    assert_accuracy!(2090, 0.95);

    // Now, it should transition to LTI.
    rx.insert((2090..2671).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    // Do another insert to wait on the updater_thread to complete the transition.
    rx.insert((2671..2722).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    assert_eq!(rx.index.medoid.load(Ordering::Relaxed), expected_medoid_id);
    assert!(rx.index.bf.read().is_none());
    assert_eq!(rx.index.temp_nodes.len(), 0);
    match &*rx.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq_vecs, .. } => {
        assert_eq!(pq_vecs.len(), 2722);
      }
    };
    assert_eq!(rx.db.maybe_read_medoid().await, Some(expected_medoid_id));
    assert_eq!(rx.db.read_index_mode().await, DbIndexMode::LongTerm);
    assert_eq!(rx.db.iter_nodes::<f32>().count().await, 2722);
    assert_accuracy!(2722, 0.85);

    // Trigger merge due to excessive deletes.
    let to_delete = (0..2722)
      .filter(|&i| !deleted.contains(&i))
      .choose_multiple(&mut thread_rng(), 200);
    iter(&to_delete)
      .for_each_concurrent(None, async |&i| {
        rx.delete(&i.to_string()).await.unwrap();
        deleted.insert(i);
      })
      .await;
    assert_eq!(rx.deleted.len(), deleted.len());
    tracing::info!(n = to_delete.len(), "deleted vectors");
    assert_accuracy!(2722, 0.84);
    // Do insert to trigger the updater_thread to start the merge.
    rx.insert((2722..2799).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    // Do another insert to wait for the updater_thread to finish the merge.
    rx.insert([("2799".to_string(), vecs[2799].clone())])
      .await
      .unwrap();
    // The medoid is never permanently deleted, so add 1.
    let expected_post_merge_nodes = 2800 - deleted.len() + 1;
    match &*rx.index.mode.read() {
      Mode::InMemory { .. } => {
        unreachable!();
      }
      Mode::LTI { pq_vecs, .. } => {
        assert_eq!(pq_vecs.len(), expected_post_merge_nodes);
      }
    };
    assert_eq!(
      rx.db.iter_nodes::<f32>().count().await,
      expected_post_merge_nodes
    );
    // The medoid is never permanently deleted.
    assert_eq!(rx.deleted.len(), 1);
    assert_eq!(rx.db.iter_deleted().count().await, 1);
    // NOTE: We don't update our `deleted` as they are deleted, it's not the same as the soft-delete markers `db.deleted`.
    assert_accuracy!(2800, 0.84);

    // Finally, insert all remaining vectors.
    rx.insert((2800..n).map(|i| (i.to_string(), vecs[i].clone())))
      .await
      .unwrap();
    tracing::info!("inserted all vectors");
    // Do final query.
    // We expect a dramatic drop in accuracy as we're inserting uniformly random data (i.e. noise) that cannot be compressed or fitted to, and we've now inserted ~300% more data so our PQ model is now very poor.
    assert_accuracy!(n, 0.77);
  }
}
