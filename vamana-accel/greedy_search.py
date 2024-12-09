from functools import partial
from jax import jit
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from typing import Optional
from util import arange
from util import NULL_ID
from util import select_nodes
from util import select_vecs
import jax.numpy as np


@partial(jit, static_argnames=["k", "ef", "iterations"])
def greedy_search(
    *,
    graph: UInt32[Array, "n m"],
    vecs: Float16[Array, "n d"],
    id_targets: UInt32[Array, "b"],  # Batched.
    k: Optional[int],  # If None, return all visited node IDs instead of top k.
    ef: int,
    iterations: int,
    id_start: UInt32[Array, ""],
):
    n, m = graph.shape
    (b,) = id_targets.shape

    # Cache outside loop.
    b_id_target_vecs = vecs[id_targets, None, :]  # (b, 1, d)
    b_row_indices = arange(b)[:, None]  # (b, 1)

    visited_ids = np.full((b, iterations), NULL_ID, dtype=np.uint32)  # (b, iterations)
    visited_dists = np.full(
        (b, iterations), np.nan, dtype=np.float16
    )  # (b, iterations)

    cand_ids = np.full((b, ef + m), NULL_ID, dtype=np.uint32)  # (b, ef + m)
    cand_dists = np.full((b, ef + m), np.nan, dtype=np.float16)  # (b, ef + m)

    cand_ids = cand_ids.at[:, 0].set(id_start)
    cand_dists = cand_dists.at[:, 0].set(
        norm(vecs[id_targets] - vecs[id_start], axis=1)
    )

    cand_seen = np.full((b, n), False, dtype=bool)  # (b, n)
    for i in range(iterations):
        # Get nodes to expand and copy to visited list.
        to_expand = cand_ids[:, 0]  # (b,)
        visited_ids = visited_ids.at[:, i].set(cand_ids[:, 0])
        visited_dists = visited_dists.at[:, i].set(cand_dists[:, 0])
        # Get neighbors of expanded nodes and calculate distance.
        # m new candidates for each query node in b.
        new_cand_ids = select_nodes(graph, to_expand)  # (b, m)
        new_cand_seen = cand_seen[b_row_indices, new_cand_ids]  # (b, m)
        # Filter out seen. Removing is not possible as rows have different seen counts and then it's no longer a matrix. This also ensures we'll never have an empty `cand_ids`. The vector at this special ID should be all NaNs to ensure its distance is always last.
        new_cand_ids = np.where(new_cand_seen, NULL_ID, new_cand_ids)  # (b, m)
        # Mark resulting new candidates as seen. JAX doesn't support using boolean masks, but we don't need to, for simplicity we can just set all as True.
        # NULL_IDs will be out-of-bounds and not assign to anything.
        cand_seen = cand_seen.at[b_row_indices, new_cand_ids].set(True)

        # Calculate distance to new candidates.
        new_cand_vecs = select_vecs(vecs, new_cand_ids)  # (b, m, d)
        new_cand_dists = norm(b_id_target_vecs - new_cand_vecs, axis=2)  # (b, m)
        # Remove expanded nodes.
        cand_ids = cand_ids.at[:, 0].set(NULL_ID)
        cand_dists = cand_dists.at[:, 0].set(np.nan)
        # Append new candidates to search list.
        # Avoid using hstack to avoid repeated memory allocation. (We do this each iteration, which gets unrolled by JIT.)
        cand_ids = cand_ids.at[:, ef:].set(new_cand_ids)
        cand_dists = cand_dists.at[:, ef:].set(new_cand_dists)
        # We cannot remove duplicates as different rows have different duplicate counts, breaking matrix. We shouldn't need to anyway, because we don't insert seen and new candidates come from neighbors of a single node which should not have duplicates.

        # Sort candidates.
        # NULL_IDs use NaNs which will be sorted to the end.
        sort_indices = np.argsort(cand_dists, axis=1)
        cand_ids = cand_ids[b_row_indices, sort_indices]
        cand_dists = cand_dists[b_row_indices, sort_indices]

    if k is None:
        return visited_ids  # (b, iterations)
    return visited_ids[
        arange(b), np.argpartition(visited_dists, k, axis=1)[:, :k]
    ]  # (b, k)
