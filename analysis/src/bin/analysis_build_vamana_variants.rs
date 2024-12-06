use libroxanne::common::PrecomputedDists;
use libroxanne::in_memory::InMemoryIndex;
use rmp_serde::to_vec_named;
use roxanne_analysis::eval;
use roxanne_analysis::export_index;
use roxanne_analysis::new_pb;
use roxanne_analysis::Dataset;
use std::cmp::max;
use std::fs;
use std::sync::Arc;

fn main() {
  let ds = Dataset::init();

  let vecs = ds.read_vectors();
  let qs = ds.read_queries();
  let knns = ds.read_results();
  println!("Loaded vectors");
  let n = ds.info.n;
  let precomputed_dists = {
    let dists = ds.read_dists();
    println!("Loaded dists");
    Some(Arc::new(PrecomputedDists::new(
      (0..n).map(|i| (i, i)).collect(),
      dists,
    )))
  };

  // update_search_list_cap is not that interesting to vary, as during building, higher always results in better graphs, it just takes longer.
  for degree_bound in [16, 32, 48, 64, 80, 128, 160, 256, 512] {
    let update_search_list_cap = max(128, degree_bound * 2);
    for distance_threshold in [
      1.0, 1.01, 1.05, 1.1, 1.12, 1.15, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 4.0,
    ] {
      let out_dir = format!(
        "vamana-{}M-{}ef-{}a",
        degree_bound, update_search_list_cap, distance_threshold
      );
      fs::create_dir_all(format!("dataset/{}/out/{out_dir}", ds.name)).unwrap();

      println!("Building {out_dir}");
      let pb = new_pb(n);
      let index = InMemoryIndex::builder(
        (0..n).collect(),
        (0..n).map(|i| vecs.row(i).to_owned()).collect(),
      )
      .degree_bound(degree_bound)
      .distance_threshold(distance_threshold)
      .update_batch_size(64)
      .update_search_list_cap(update_search_list_cap)
      .medoid_sample_size(10_000)
      .precomputed_dists(precomputed_dists.clone())
      .on_progress(|completed| pb.set_position(completed as u64))
      .build();
      pb.finish();

      export_index(&ds, &out_dir, &index.graph, index.medoid);

      let e = eval(
        &index,
        &qs.view(),
        &knns.view(),
        128, // Unlike update_search_list_cap, this must be kept invariant as otherwise the larger graphs benefit from a larger search cap too.
        1,
      );
      fs::write(
        format!("dataset/{}/out/{out_dir}/query_metrics.msgpack", ds.name),
        to_vec_named(&e.query_metrics).unwrap(),
      )
      .unwrap();

      println!(
        "Correct: {:.2}% ({}/{})",
        e.ratio() * 100.0,
        e.correct,
        e.total,
      );
    }
  }

  println!("All done!");
}
