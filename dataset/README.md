Place in this folder one folder per dataset. Each dataset should contain these files:

- **info.toml**: TOML file containing these properties:
  - *dtype*: one of {float{16,32,64},{u,}int{8,16,32,64}}
  - *dim*: dimensions
  - *n*: vectors
  - *q*: queries
  - *k*: KNNs
- **vectors.bin**: packed little-endian numbers representing (n, dim) matrix of vectors
- **dists.bin**: packed little-endian f16s representing (n, n) matrix of precomputed pairwise distances
- **queries.bin**: packed little-endian numbers representing (q, dim) matrix of eval query vectors
- **results.bin**: packed little-endian u32s representing (q, k) matrix of k nearest neighbor indices for each eval query vector

To use a dataset with an `analysis` tool, set the `DS=$folder_name` environment variable. Analysis outputs will be placed under `$folder_name/out`.
