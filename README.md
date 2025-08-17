# CoreNN

Database for querying **billions** of vectors and embeddings in sublinear time on commodity machines.

Read the [accompanying blog post](https://blog.wilsonl.in/corenn/) for details and an accessible deep dive and in-depth technical report.

## Getting started

### Rust

```rust
extern crate blas_src;
use libcorenn::{CoreNN, cfg::Cfg};

fn main() {
  let db = CoreNN::create("/path/to/db", Cfg {
    // Specify the dimensionality of your vectors.
    dim: 3,
    // All other config options are optional.
    ..Default::default()
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
  # Specify the dimensionality of your vectors.
  "dim": 3,
  # All other config options are optional.
})
keys = [
  "my_entry_1",
  "my_entry_2",
]
vectors = np.array(
    [
        [0.3, 0.6, 0.9],
        [0.4, 1.1, 0.0],
    ],
    dtype=np.float32,
)
# Or insert_bf16, insert_f16, insert_f64
db.insert_f32(keys, vectors)

# Later...
db = CoreNN.open("/path/to/db")
queries = np.array([
  [1.0, 1.3, 1.7],
  [7.3, 2.5, 0.0],
])
# Returns a list of (key, distance) tuples
# for each query vector
# (so returns a list of lists).
k100 = db.query_f32(queries, 100)
```

### Node.js

```typescript
import { CoreNN } from "@corenn/node";

const db = CoreNN.create("/path/to/db", {
  // Specify the dimensionality of your vectors.
  dim: 3,
  // All other config options are optional.
});
db.insert([
  {
    key: "my_entry_1",
    vector: new Float32Array([0.5, 4.1, 2.2]),
  },
  {
    key: "my_entry_2",
    vector: new Float32Array([3.1, 7.7, 6.4]),
  },
]);

// Later...
const db = CoreNN.open("/path/to/db");
// Array of { key, distance } objects.
const results = db.query(
  new Float32Array([0.0, 1.1, 2.2]),
  100,
);
```
