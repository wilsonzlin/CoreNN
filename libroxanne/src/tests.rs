use crate::cfg::Cfg;
use crate::common::dist_l2;
use crate::common::StdMetric;
use crate::util::AsyncConcurrentIteratorExt;
use crate::util::AsyncConcurrentStreamExt;
use crate::Mode;
use crate::Roxanne;
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
        .map(|j| (j, dist_l2(&vecs[i].view(), &vecs[j].view())))
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
    toml::to_string(&Cfg {
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
  let rx = Roxanne::<f32, f32>::open(dir_raw).await;
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
  let expected_medoid_i = calc_approx_medoid::<f32, f32>(
    &(0..1313).map(|i| (i, vecs[i].clone())).collect(),
    dist_l2,
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
