from functools import partial
from jax import jit
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from util import arange
from util import NULL_ID
from util import select_vecs
import jax.numpy as np


@partial(jit, static_argnames=["m"])
def compute_robust_pruned(
    *,
    vecs: Float16[Array, "n d"],
    # Batched. May contain NULL_IDs; corresponding output row contains junk and should be ignored.
    node_ids: UInt32[Array, "b"],
    # It's safe to have interspersed NULL_IDs and duplicates in here.
    cand_ids: UInt32[Array, "b c"],
    m: int,
    dist_thresh: Float16[Array, ""],
):
    b, _ = cand_ids.shape

    b_node_vecs = select_vecs(vecs, node_ids)[:, None, :]  # (b, 1, d)
    b_row_indices = arange(b)[:, None]  # (b, 1)

    res = np.full((b, m), NULL_ID, np.uint32)  # (b, m)

    # Calculate distance from node to each candidate.
    cand_vecs = select_vecs(vecs, cand_ids)  # (b, c, d)
    # NULL_IDs will be NaNs which will be sorted to the end.
    cand_dists = norm(b_node_vecs - cand_vecs, axis=2)  # (b, c)
    # Sort candidates by distance to node.
    cand_i = np.argsort(cand_dists, axis=1)
    cand_dists = cand_dists[b_row_indices, cand_i]
    # This must be sorted too as we'll later use it to compare to `p_star_dists`.
    cand_vecs = cand_vecs[b_row_indices, cand_i]
    # IDs will be set to NULL_ID as they are picked or pruned.
    cand_ids = cand_ids[b_row_indices, cand_i]  # (b, c)
    for i in range(m):
        # If we've run out of candidates for a node, this will just pick the first (closest) candidate again as argmax returns the smallest index on conflicts, which is now NULL_ID (as we've previously picked it and masked it with NULL_ID). This is correct as we want to append NULL_ID to `res`. It won't prune anything since the NaNs will propagate when calculating dists and not be `<=` to anything.
        p_star_indices = (cand_ids != NULL_ID).argmax(axis=1)  # (b,)
        p_star = cand_ids[arange(b), p_star_indices]  # (b,)
        res = res.at[:, i].set(p_star)
        cand_ids = cand_ids.at[arange(b), p_star_indices].set(NULL_ID)

        p_star_vecs = select_vecs(vecs, p_star)[:, None, :]  # (b, 1, d)
        # p_star_dists[p_star][candidate] is distance from p_star to candidate with distance threshold multiplier.
        # If there are duplicates, this will handle it as their distance will be 0.
        p_star_dists = dist_thresh * norm(p_star_vecs - cand_vecs, axis=2)  # (b, c)
        # cand_dists[node][candidate] is distance from node to candidate.
        to_prune = p_star_dists <= cand_dists  # (b, c)
        cand_ids = np.where(to_prune, NULL_ID, cand_ids)  # type: ignore

    return res  # (b, m)
