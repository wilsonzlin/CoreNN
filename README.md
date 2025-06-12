# CoreNN

Database for [querying **billions** of vectors and embeddings](https://en.wikipedia.org/wiki/Nearest_neighbor_search) in sublinear time on commodity machines.

Read the [accompanying blog post](https://blog.wilsonl.in/corenn) for an accessible introduction and in-depth technical report.

## Getting started

### Rust

```rust
fn main() {
  let db = CoreNN::create("/path/to/db", Cfg {
    // Specify the dimensionality of your vectors.
    dim: 3,
    // All others are optional.
    ...Default::default()
  });
  let key = "my_entry".to_string();
  // This can be bf16, f16, f32, or f64.
  let vec = vec![0.3, 0.6, 0.9];
  db.insert(&key, &vec);

  // Later...
  let db = CoreNN::open("/path/to/db");
  let query = vec![1.0, 1.3, 1.7];
  // Returns Vec of (key, distance) pairs.
  let k100 = db.query(&query, 100);
  assert_eq!(k100[0].0.as_str(), "my_entry");
}
```

### Python

```python
from corenn_py import CoreNN

db = CoreNN.create("/path/to/db", {
  "dim": 3,
})
# Or insert_bf16, insert_f16, insert_f64
db.insert_f32("my_entry", np.array([0.3, 0.6, 0.9]))

# Later...
db = CoreNN.open("/path/to/db")
# Returns a list of (key, distance) tuples
# for each query vector
# (so returns a list of lists).
k100 = db.query_f32(np.array([
  [1.0, 1.3, 1.7],
  [7.3, 2.5, 0.0],
]), 100)
```
