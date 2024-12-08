import argparse
from functools import partial
import math
from typing import Optional
from jax import jit
import jax.random as rand
import jax.numpy as np
from jaxtyping import Float16, Array, UInt32
from jax.numpy.linalg import norm
from tqdm import tqdm

NULL_ID = np.iinfo(np.uint32).max

def batches(n: int, sz: int):
    for start in range(0, n, sz):
        end = min(start + sz, n)
        # Last batch may be smaller than `sz`.
        yield (start, end, end - start)

import numpy as np

def count_non_nan_prefix_columns(arr: Float16[Array, "x y"]):
    # Create a boolean mask where True indicates non-NaN values
    mask = ~np.isnan(arr)

    # Find the last non-NaN position in each row
    last_valid_indices = np.max(mask * np.arange(arr.shape[1]), axis=1)

    # Get the maximum index needed to preserve all non-NaN values
    return int(np.max(last_valid_indices)) + 1

@jit
def pair_matrix_vector(M, V):
    """
    Creates a 2D matrix of pairs from matrix M and vector V.
    Args:
        M: numpy array of shape (b, m)
        V: numpy array of shape (b,)
    Returns:
        numpy array of shape (b*m, 2) where each row is a pair (M[i,j], V[i])
    """
    return np.column_stack((M.ravel(), np.repeat(V, M.shape[1])))

@jit
def group_by_x(P: UInt32[Array, "x y"]):
    """
    Group a sorted-by-x array of distinct (x, y) pairs by x into a non-jagged 2D matrix filled with uint32.max.

    Parameters
    ----------
    P : np.ndarray
        A (N, 2) NumPy array of distinct (x, y) pairs, sorted by the first column (x).

    Returns
    -------
    x_values : np.ndarray
        A 1D array of the distinct x-values.
    result : np.ndarray
        A 2D array of shape (len(x_values), max_count) where max_count is the maximum
        number of pairs per distinct x. Missing positions are filled with uint32.max.
    """
    # Extract unique x-values, indices for their first occurrences, and the counts per x.
    x_values, idx_start, counts = np.unique(P[:, 0], return_index=True, return_counts=True)

    max_count = np.max(counts)
    uint32_max = np.iinfo(np.uint32).max

    # Prepare the result matrix filled with uint32.max
    result = np.full((len(x_values), max_count), uint32_max, dtype=np.uint32)

    # We have all y-values in order since P is sorted by x
    y_values = P[:, 1]

    # Determine the row index for each element in P.
    # We know the rows are aligned with unique x-values in ascending order,
    # and that P is sorted by x. The repetition pattern of row indices matches the counts.
    row_idx = np.repeat(np.arange(len(x_values)), counts)

    # Determine the column indices.
    # For each group i, we need columns [0, 1, 2, ..., counts[i]-1].
    # Construct column indices by creating a continuous range and then offsetting by group starts.

    # The starting position for each block of y-values is given by cumulative sums of counts.
    block_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # Global sequence of indices for all y-values
    global_indices = np.arange(np.sum(counts))
    # Subtract the per-group start index to get within-group indices
    col_idx = global_indices - np.repeat(block_starts, counts)

    # Place y_values into result matrix using fancy indexing
    result[row_idx, col_idx] = y_values

    return x_values, result

# Handles NULL_ID indices.
def select_vecs(
  vecs: Float16[Array, "n d"],
  indices: np.ndarray,
):
  return vecs.at(indices).get("fill", fill_value=np.nan)

@partial(jit, static_argnames=["sample_size", "seed"])
def calc_approx_medoid(
  *,
  vecs: Float16[Array, "n d"],
  sample_size: int,
  seed: int,
):
  n, _ = vecs.shape
  rk = rand.key(seed)
  indices = rand.choice(rk, np.arange(n), shape=(sample_size,), replace=False) # (sample_size,)
  sample = vecs[indices] # (sample_size, d)
  # Calculate between samples only.
  dists = norm(sample[:, None, :] - sample[None, :, :], axis=2) # (sample_size, sample_size)
  return indices[np.argmin(dists.sum(axis=0))]

@partial(jit, static_argnames=["n", "m", "m_max", "seed"])
def init_random_graph(
  *,
  n: int,
  m: int,
  m_max: int,
  seed: int,
):
  rk = rand.key(seed)
  # Allow node to have up to m_max edges so we don't have to compute_robust_pruned after adding every single backedge.
  graph = np.full((n, m_max), NULL_ID, dtype=np.uint32)
  for i in range(n):
    graph[i, :m] = rand.choice(rk, np.delete(np.arange(n), i), shape=(m,), replace=False)
  return graph

@partial(jit, static_argnames=["k", "iterations", "id_start"])
def greedy_search(
  *,
  graph: UInt32[Array, "n m_max"],
  vecs: Float16[Array, "n d"],
  id_targets: UInt32[Array, "b"], # Batched.
  k: Optional[int], # If None, return all visited node IDs instead of top k.
  iterations: int,
  id_start: int,
):
  n, _ = vecs.shape
  b, = id_targets.shape

  # Cache outside loop.
  b_id_target_vecs = vecs[id_targets, None, :] # (b, 1, d)
  b_row_indices = np.arange(b)[:, None] # (b, 1)

  visited_ids = np.full((b, iterations), NULL_ID, dtype=np.uint32) # (b, iterations)
  visited_dists = np.full((b, iterations), float("inf"), dtype=np.float16) # (b, iterations)

  cand_ids = np.full((b, 1), id_start, dtype=np.uint32) # (b, *)
  cand_dists = norm(vecs[id_targets] - vecs[id_start], axis=1, keepdims=True) # (b, *)
  # Add 1 so that NULL_ID gets bounded to it (and won't clobber any legitimate node ID).
  cand_seen = np.full((b, n + 1), False, dtype=bool) # (b, n + 1)
  for i in range(iterations):
    # Get nodes to expand and copy to visited list.
    to_expand = cand_ids[:, 0] # (b,)
    visited_ids[:, i] = cand_ids[:, 0]
    visited_dists[:, i] = cand_dists[:, 0]
    # Get neighbors of expanded nodes and calculate distance.
    new_cand_ids = graph[to_expand] # (b, m)
    # NULL_ID will get bounded to `n`. (See comment on `cand_seen`.)
    new_cand_seen = cand_seen[b_row_indices, new_cand_ids] # (b, m)
    # Filter out seen. Removing is not possible as rows have different seen counts and then it's no longer a matrix. This also ensures we'll never have an empty `cand_ids`. The vector at this special ID should be all NaNs to ensure its distance is always last.
    new_cand_ids[new_cand_seen] = NULL_ID
    not_seen = ~new_cand_seen
    cand_seen[b_row_indices[not_seen], new_cand_ids[not_seen]] = True

    # Calculate distance to new candidates.
    new_cand_vecs = select_vecs(vecs, new_cand_ids) # (b, m, d)
    new_cand_dists = norm(b_id_target_vecs - new_cand_vecs, axis=2) # (b, m)
    # Remove expanded nodes and append new candidates to search list.
    cand_ids = np.hstack([cand_ids[:, 1:], new_cand_ids]) # (b, * - 1 + m)
    cand_dists = np.hstack([cand_dists[:, 1:], new_cand_dists]) # (b, * - 1 + m)
    # We cannot remove duplicates as different rows have different duplicate counts, breaking matrix. We shouldn't need to anyway, because we don't insert seen and new candidates come from neighbors of a single node which should not have duplicates.

    # Sort candidates.
    sort_indices = np.argsort(cand_dists, axis=1) # NULL_IDs use NaNs which will be sorted to the end.
    cand_ids = cand_ids[b_row_indices, sort_indices]
    cand_dists = cand_dists[b_row_indices, sort_indices]
    # Remove NULL_IDs where possible.
    cols = count_non_nan_prefix_columns(cand_dists)
    cand_ids = cand_ids[:, :cols]
    cand_dists = cand_dists[:, :cols]

  if k is None:
    return visited_ids # (b, iterations)
  return visited_ids[np.arange(b), np.argpartition(visited_dists, k, axis=1)[:, :k]] # (b, k)

# Returns (b, m_max) but will only ever fill (b, m); the rest are conveniently filled with NULL_ID.
@partial(jit, static_argnames=["m", "dist_thresh"])
def compute_robust_pruned(
  *,
  vecs: Float16[Array, "n d"],
  node_ids: UInt32[Array, "b"], # Batched.
  cand_ids: UInt32[Array, "b c"], # It's safe to have interspersed NULL_IDs and duplicates in here.
  m: int,
  dist_thresh: float,
):
  b, _ = cand_ids.shape

  b_node_vecs = vecs[node_ids, None, :] # (b, 1, d)

  res = np.full((b, m), NULL_ID, np.uint32) # (b, m)

  # Calculate distance from node to each candidate.
  cand_vecs = select_vecs(vecs, cand_ids) # (b, c, d)
  cand_dists = norm(b_node_vecs - cand_vecs, axis=2) # (b, c); NULL_IDs will be NaNs which will be sorted to the end.
  # Sort candidates by distance to node.
  cand_dists, cand_i = np.argsort(cand_dists, axis=1)
  # IDs will be set to NULL_ID as they are picked or pruned.
  cand_ids = cand_ids[cand_i] # (b, c)
  for i in range(m):
    # If we've run out of candidates for a node, this will just pick the first (closest) candidate again as argmax returns the smallest index on conflicts, which is now NULL_ID (as we've previously picked it and masked it with NULL_ID). This is correct as we want to append NULL_ID to `res`. It won't prune anything since the NaNs will propagate when calculating dists and not be `<=` to anything.
    p_star_indices = (cand_ids != NULL_ID).argmax(axis=1) # (b,)
    p_star = cand_ids[np.arange(b), p_star_indices] # (b,)
    res[:, i] = p_star
    cand_ids[np.arange(b), p_star_indices] = NULL_ID

    p_star_vecs = vecs[p_star, None, :] # (b, 1, d)
    # p_star_dists[p_star][candidate] is distance from p_star to candidate with distance threshold multiplier.
    # If there are duplicates, this will handle it as their distance will be 0.
    p_star_dists = dist_thresh * norm(p_star_vecs - cand_vecs, axis=2) # (b, c)
    # cand_dists[node][candidate] is distance from node to candidate.
    to_prune = p_star_dists <= cand_dists # (b, c)
    cand_ids[to_prune] = NULL_ID

  return res # (b, m)

def optimize_graph(
  *,
  graph: UInt32[Array, "n m_max"],
  vecs: Float16[Array, "n d"],
  id_medoid: int,
  m: int,
  ef: int,
  dist_thresh: float,
  update_batch_size: int,
  seed: int,
):
  n, _ = graph.shape
  rk = rand.key(seed)
  nodes = rand.permutation(rk, n).astype(np.uint32)
  pb = tqdm(total=n)
  for start, end, b in batches(n, update_batch_size):
    batch_nodes = nodes[start:end] # (b,)
    visited = greedy_search(
      graph=graph,
      vecs=vecs,
      id_targets=batch_nodes,
      k=None,
      iterations=ef,
      id_start=id_medoid,
    ) # (b, ef)
    new_neighbors = compute_robust_pruned(
      cand_ids=np.hstack([visited, graph[batch_nodes, :]]),
      dist_thresh=dist_thresh,
      m=m,
      node_ids=batch_nodes,
      vecs=vecs,
    ) # (b, m)
    # No need to clear m:m_max as compute_robust_pruned already returns m_max with NULL_ID filled after m.
    graph[batch_nodes] = new_neighbors

    # `new_neighbors` is a matrix (b, m) where we add an edge b->m[j] (i.e. `m` is the out-neighbors for `b`).
    # Build "list" of backedge (m[j], b) pairs.
    backedges = pair_matrix_vector(new_neighbors, batch_nodes) # (b * ef, 2)
    # Sort by `m[j]` as group_by_x depends on it.
    backedges = backedges[backedges[:, 0].argsort()]
    back_nodes, back_add_neighbors = group_by_x(backedges) # (*,), (*, * <= b)
    # compute_robust_pruned will handle duplicates.
    graph[back_nodes] = compute_robust_pruned(
      cand_ids=np.hstack([back_add_neighbors, graph[back_nodes, :]]),
      dist_thresh=dist_thresh,
      m=m,
      node_ids=back_nodes,
      vecs=vecs,
    )

    pb.update(b)
  pb.close()


def main():
  parser = argparse.ArgumentParser(
      description="Build a Vamana graph using the GPU."
  )
  parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
  parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
  parser.add_argument("--dim", type=int, help="Vector dimensions")
  parser.add_argument("--m", type=int, help="Degree bound")
  parser.add_argument("--ef", type=int, help="Search list cap")
  parser.add_argument("--alpha", type=float, help="Distance threshold")
  args = parser.parse_args()

  dtype = np.dtype(args.dtype)
  with open(args.input, "rb") as f:
    # Cast to float16 for faster calculations.
    vecs = np.frombuffer(f.read(), dtype=dtype).reshape((-1, args.dim)).astype(np.float16)
  n, _ = vecs.shape
  m = args.m
  m_max = m * 2
  ef = args.ef
  update_batch_size = 64
  dist_thresh = args.alpha
  seed = 0
  medoid_sample_size = 10_000

  medoid = calc_approx_medoid(
    vecs=vecs,
    sample_size=medoid_sample_size,
    seed=seed,
  )
  graph = init_random_graph(
    n=n,
    m=m,
    m_max=m_max,
    seed=seed,
  )
  graph = optimize_graph(
    graph=graph,
    vecs=vecs,
    id_medoid=medoid,
    m=m,
    ef=ef,
    dist_thresh=dist_thresh,
    update_batch_size=update_batch_size,
    seed=seed,
  )
  print("All done!")

if __name__ == "__main__":
  main()
