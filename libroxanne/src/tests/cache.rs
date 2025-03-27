use crate::Cache;
use std::time::Duration;
use tokio::join;
use tokio::time::sleep;

#[tokio::test]
async fn test_cache() {
  let cache = Cache::<usize>::new();
  cache.insert(0, 0);
  assert_eq!(cache.get_or_compute(0, async || 42).await, 0);
  let (a, b) = join! {
    cache.get_or_compute(1, async || {
      sleep(Duration::from_secs(1)).await;
      50
    }),
    cache.get_or_compute(1, async || {
      sleep(Duration::from_millis(100)).await;
      999
    }),
  };
  assert_eq!(a, 50);
  assert_eq!(b, 50);
}
