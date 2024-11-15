Place in this folder one folder per dataset. Each dataset should contain these files:

- **info.toml**: TOML file containing these properties:
  - *dtype*: one of {float{16,32,64},{u,}int{8,16,32,64}}
  - *dim*: dimensions
  - *n*: vectors
  - *q*: queries
  - *k*: KNNs
- **vectors.bin**: packed little-endian numbers representing (n, dim) matrix of vectors
- **dists.bin**: packed little-endian f16s representing (n, n) matrix of precomputed pairwise distances
  - This can be generated using [calc_dists.py](../tools/calc_dists.py)
- **queries.bin**: packed little-endian numbers representing (q, dim) matrix of eval query vectors
- **results.bin**: packed little-endian u32s representing (q, k) matrix of k nearest neighbor indices for each eval query vector
  - This can be generated from dataset vectors using [calc_dists.py](../tools/calc_dists.py), although ideally queries are not merely vectors in the training dataset

To use a dataset with an `analysis` tool, set the `DS=$folder_name` environment variable. Analysis outputs will be placed under `$folder_name/out`.

Convert corpus-texmex.irisa.fr vectors to these files using [convert_texmex.py](../tools/convert_texmex.py).
