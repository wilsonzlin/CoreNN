use crate::db::DbTransaction;
use crate::tests::create_test_rox;
use crate::tests::node;
use half::f16;

#[tokio::test]
async fn test_get_node_for_uncompressed() {
  let rox = create_test_rox("get_node_for_uncompressed").await;

  // Insert a node.
  let mut txn = DbTransaction::new();
  txn.write_node(33, &node(vec![40], vec![0.97]));
  txn.commit(&rox.db).await;

  let vec = rox.get_point(33).await;
  assert_eq!(vec[0], f16::from_f32(0.97));

  let node = rox.get_node(33).await;
  assert_eq!(node.neighbors, vec![40]);
  assert_eq!(node.vector[0], f16::from_f32(0.97));
}

#[tokio::test]
async fn test_dist() {
  let rox = create_test_rox("dist").await;

  // Insert two nodes.
  let mut txn = DbTransaction::new();
  txn.write_node(33, &node(vec![40], vec![0.2]));
  txn.write_node(40, &node(vec![33], vec![0.8]));
  txn.commit(&rox.db).await;

  let dist = rox.dist(33, 40).await;
  // L2 distance: sqrt((0.2 - 0.8)^2) = sqrt(0.36) = 0.6
  assert!((dist - 0.6).abs() < 1e-3, "dist = {dist}");
}

#[tokio::test]
async fn test_dist2() {
  let rox = create_test_rox("dist2").await;

  // Insert a node.
  let mut txn = DbTransaction::new();
  txn.write_node(33, &node(vec![40], vec![0.2]));
  txn.commit(&rox.db).await;

  // Query vector.
  let query = ndarray::Array1::from(vec![f16::from_f32(0.8)]);

  let dist = rox.dist2(33, &query.view()).await;
  // L2 distance: sqrt((0.2 - 0.8)^2) = sqrt(0.36) = 0.6
  assert!((dist - 0.6).abs() < 1e-3, "dist = {dist}");
}
