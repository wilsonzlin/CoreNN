from argparse import ArgumentParser
from functools import partial
from jax import jit
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from tqdm import tqdm
from typing import Optional
import jax
import jax.numpy as np
import jax.random as rand
import msgpack

# Use int32 as some functions use/return int32, which if we provide the uint32 max value, will overflow to -1, which is a legal index.
NULL_ID = np.uint32(np.iinfo(np.int32).max)


# By default, np.arange returns int32 which causes a lot of issues with JAX.
@partial(jit, static_argnames=["n"])
def arange(n: int):
    return np.arange(n, dtype=np.uint32)


def batches(n: int, sz: int):
    for start in range(0, n, sz):
        end = min(start + sz, n)
        # Last batch may be smaller than `sz`.
        yield (start, end, end - start)


@jit
def count_non_nan_prefix_columns(arr: Float16[Array, "x y"]):
    # Create a boolean mask where True indicates non-NaN values
    mask = ~np.isnan(arr)

    # Find the last non-NaN position in each row
    last_valid_indices = np.max(mask * arange(arr.shape[1]), axis=1)

    # Get the maximum column index needed to preserve all non-NaN values
    return np.max(last_valid_indices) + 1


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
def insert_values_into_filler_positions(
    A: UInt32[Array, "n m"],
    rows: UInt32[Array, "k"],
    unique_rows: UInt32[Array, "n"],  # Padded with NULL_ID up to n.
    values: UInt32[Array, "k"],
):
    n, m = A.shape

    # Extract subset; since we pad with NULL_ID, make sure to drop out-of-bounds rows
    A_subset = A.at[unique_rows].get(mode="drop")  # shape (r, m)

    # Find filler positions only in the relevant subset
    # nonzero returns int32, not unsigned, so make sure NULL_ID is int32.max, not uint32.max which overflows to -1 which is a legal index.
    sub_fill_rows, sub_fill_cols = np.nonzero(
        A_subset == NULL_ID, size=n * m, fill_value=NULL_ID
    )
    global_fill_rows = unique_rows.at[sub_fill_rows].get(mode="drop")

    # Sort insertions by row
    insert_sort_idx = np.argsort(rows)
    sorted_rows = rows[insert_sort_idx]
    sorted_vals = values[insert_sort_idx]

    # Sort filler positions by row
    fill_sort_idx = np.lexsort((sub_fill_cols, global_fill_rows))
    fr = global_fill_rows[fill_sort_idx]
    fc = sub_fill_cols[fill_sort_idx]

    insertion_counts = np.zeros((n,), dtype=np.uint32).at[sorted_rows].add(1)
    filler_counts = np.zeros((n,), dtype=np.uint32).at[fr].add(1)

    insertion_starts = np.cumsum(
        np.concatenate([np.array([0], dtype=np.uint32), insertion_counts[:-1]])
    )
    filler_starts = np.cumsum(
        np.concatenate([np.array([0], dtype=np.uint32), filler_counts[:-1]])
    )

    insertion_row_starts = insertion_starts[sorted_rows]
    insertion_pos_in_row = arange(sorted_rows.size) - insertion_row_starts
    fill_row_starts_for_insertions = filler_starts[sorted_rows]
    filler_idx_for_insertions = fill_row_starts_for_insertions + insertion_pos_in_row

    final_rows = fr[filler_idx_for_insertions]
    final_cols = fc[filler_idx_for_insertions]

    return A.at[final_rows, final_cols].set(sorted_vals)


# Handles NULL_ID indices.
@jit
def select_vecs(
    vecs: Float16[Array, "n d"],
    indices: np.ndarray,
):
    return vecs.at[indices].get(mode="fill", fill_value=np.nan)


@partial(jit, static_argnames=["sample_size", "seed"])
def calc_approx_medoid(
    *,
    vecs: Float16[Array, "n d"],
    sample_size: int,
    seed: int,
):
    n, _ = vecs.shape
    rk = rand.key(seed)
    indices = rand.choice(
        rk, arange(n), shape=(sample_size,), replace=False
    )  # (sample_size,)
    sample = vecs[indices]  # (sample_size, d)
    # Calculate between samples only.
    dists = norm(
        sample[:, None, :] - sample[None, :, :], axis=2
    )  # (sample_size, sample_size)
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
    # Duplicates are fine, we'll compute_robust_pruned them out later.
    graph = rand.choice(rk, arange(n), shape=(n, m)).astype(np.uint32)
    # Remove self-edges.
    row_indices = arange(n)[:, None]
    graph = np.where(graph == row_indices, NULL_ID, graph)
    # Allow node to have up to m_max edges so we don't have to compute_robust_pruned after adding every single backedge.
    graph = np.pad(
        graph,
        pad_width=((0, 0), (0, m_max - m)),
        mode="constant",
        constant_values=NULL_ID,
    )
    return graph


@partial(jit, static_argnames=["k", "ef", "iterations"])
def greedy_search(
    *,
    graph: UInt32[Array, "n m_max"],
    vecs: Float16[Array, "n d"],
    id_targets: UInt32[Array, "b"],  # Batched.
    k: Optional[int],  # If None, return all visited node IDs instead of top k.
    ef: int,
    iterations: int,
    id_start: UInt32[Array, ""],
):
    n, _ = vecs.shape
    (b,) = id_targets.shape

    # Cache outside loop.
    b_id_target_vecs = vecs[id_targets, None, :]  # (b, 1, d)
    b_row_indices = arange(b)[:, None]  # (b, 1)

    visited_ids = np.full((b, iterations), NULL_ID, dtype=np.uint32)  # (b, iterations)
    visited_dists = np.full(
        (b, iterations), float("inf"), dtype=np.float16
    )  # (b, iterations)

    cand_ids = np.full((b, 1), id_start, dtype=np.uint32)  # (b, *)
    cand_dists = norm(
        vecs[id_targets] - vecs[id_start], axis=1, keepdims=True
    )  # (b, *)
    cand_seen = np.full((b, n), False, dtype=bool)  # (b, n + 1)
    for i in range(iterations):
        # Get nodes to expand and copy to visited list.
        to_expand = cand_ids[:, 0]  # (b,)
        visited_ids = visited_ids.at[:, i].set(cand_ids[:, 0])
        visited_dists = visited_dists.at[:, i].set(cand_dists[:, 0])
        # Get neighbors of expanded nodes and calculate distance.
        new_cand_ids = graph[
            to_expand
        ]  # (b, m_max); m_max new candidates for each query node in b.
        # NULL_ID will get bounded to `n`. (See comment on `cand_seen`.)
        new_cand_seen = cand_seen[b_row_indices, new_cand_ids]  # (b, m_max)
        # Filter out seen. Removing is not possible as rows have different seen counts and then it's no longer a matrix. This also ensures we'll never have an empty `cand_ids`. The vector at this special ID should be all NaNs to ensure its distance is always last.
        new_cand_ids = np.where(new_cand_seen, NULL_ID, new_cand_ids)  # (b, m_max)
        # Mark resulting new candidates as seen. JAX doesn't support using boolean masks, but we don't need to, for simplicity we can just set all as True.
        # NULL_IDs will be out-of-bounds and not assign to anything.
        cand_seen = cand_seen.at[b_row_indices, new_cand_ids].set(True)

        # Calculate distance to new candidates.
        new_cand_vecs = select_vecs(vecs, new_cand_ids)  # (b, m_max, d)
        new_cand_dists = norm(b_id_target_vecs - new_cand_vecs, axis=2)  # (b, m_max)
        # Remove expanded nodes and append new candidates to search list.
        cand_ids = np.hstack([cand_ids[:, 1:], new_cand_ids])  # (b, * - 1 + m_max)
        cand_dists = np.hstack(
            [cand_dists[:, 1:], new_cand_dists]
        )  # (b, * - 1 + m_max)
        # We cannot remove duplicates as different rows have different duplicate counts, breaking matrix. We shouldn't need to anyway, because we don't insert seen and new candidates come from neighbors of a single node which should not have duplicates.

        # Sort candidates.
        sort_indices = np.argsort(
            cand_dists, axis=1
        )  # NULL_IDs use NaNs which will be sorted to the end.
        cand_ids = cand_ids[b_row_indices, sort_indices]
        cand_dists = cand_dists[b_row_indices, sort_indices]
        cand_ids = cand_ids[:, :ef]
        cand_dists = cand_dists[:, :ef]

    if k is None:
        return visited_ids  # (b, iterations)
    return visited_ids[
        arange(b), np.argpartition(visited_dists, k, axis=1)[:, :k]
    ]  # (b, k)


# Returns (b, m_max) but will only ever fill (b, m); the rest are conveniently filled with NULL_ID.
@partial(jit, static_argnames=["m", "m_max"])
def compute_robust_pruned(
    *,
    vecs: Float16[Array, "n d"],
    node_ids: UInt32[Array, "b"],  # Batched.
    cand_ids: UInt32[
        Array, "b c"
    ],  # It's safe to have interspersed NULL_IDs and duplicates in here.
    m: int,
    m_max: int,
    dist_thresh: Float16[Array, ""],
):
    b, _ = cand_ids.shape

    b_node_vecs = vecs[node_ids, None, :]  # (b, 1, d)

    res = np.full((b, m_max), NULL_ID, np.uint32)  # (b, m_max)

    # Calculate distance from node to each candidate.
    cand_vecs = select_vecs(vecs, cand_ids)  # (b, c, d)
    cand_dists = norm(
        b_node_vecs - cand_vecs, axis=2
    )  # (b, c); NULL_IDs will be NaNs which will be sorted to the end.
    # Sort candidates by distance to node.
    cand_i = np.argsort(cand_dists, axis=1)
    cand_dists = np.take_along_axis(cand_dists, cand_i, axis=1)
    # IDs will be set to NULL_ID as they are picked or pruned.
    cand_ids = np.take_along_axis(cand_ids, cand_i, axis=1)  # (b, c)
    for i in range(m):
        # If we've run out of candidates for a node, this will just pick the first (closest) candidate again as argmax returns the smallest index on conflicts, which is now NULL_ID (as we've previously picked it and masked it with NULL_ID). This is correct as we want to append NULL_ID to `res`. It won't prune anything since the NaNs will propagate when calculating dists and not be `<=` to anything.
        p_star_indices = (cand_ids != NULL_ID).argmax(axis=1)  # (b,)
        p_star = cand_ids[arange(b), p_star_indices]  # (b,)
        res = res.at[:, i].set(p_star)
        cand_ids = cand_ids.at[arange(b), p_star_indices].set(NULL_ID)

        p_star_vecs = vecs[p_star, None, :]  # (b, 1, d)
        # p_star_dists[p_star][candidate] is distance from p_star to candidate with distance threshold multiplier.
        # If there are duplicates, this will handle it as their distance will be 0.
        p_star_dists = dist_thresh * norm(p_star_vecs - cand_vecs, axis=2)  # (b, c)
        # cand_dists[node][candidate] is distance from node to candidate.
        to_prune = p_star_dists <= cand_dists  # (b, c)
        cand_ids = np.where(to_prune, NULL_ID, cand_ids)

    return res  # (b, m_max)


@partial(
    jit,
    static_argnames=["m", "m_max", "ef"],
)
def optimize_graph_batch(
    *,
    graph: UInt32[Array, "n m_max"],
    vecs: Float16[Array, "n d"],
    batch_nodes: UInt32[Array, "b"],
    id_medoid: UInt32[Array, ""],
    m: int,
    m_max: int,
    ef: int,
    dist_thresh: Float16[Array, ""],
):
    n, _ = vecs.shape
    visited = greedy_search(
        graph=graph,
        vecs=vecs,
        id_targets=batch_nodes,
        k=None,
        ef=ef,
        iterations=ef,
        id_start=id_medoid,
    )  # (b, ef)
    new_neighbors = compute_robust_pruned(
        cand_ids=np.hstack([visited, graph[batch_nodes, :]]),
        dist_thresh=dist_thresh,
        m=m,
        m_max=m_max,
        node_ids=batch_nodes,
        vecs=vecs,
    )  # (b, m_max)
    # No need to clear m:m_max as compute_robust_pruned already returns m_max with NULL_ID filled after m.
    graph = graph.at[batch_nodes].set(new_neighbors)

    # `new_neighbors` is a matrix (b, m) where we add an edge b->m[j] (i.e. `m` is the out-neighbors for `b`).
    # Build "list" of backedge (m[j], b) pairs.
    backedges = pair_matrix_vector(new_neighbors, batch_nodes)  # (b * ef, 2)
    rows = backedges[:, 0]
    unique_rows, row_freq = np.unique(
        rows, return_counts=True, size=n, fill_value=NULL_ID
    )
    # These rows have no more capacity for their pending-backedges, so we need to prune them.
    rows_to_prune = np.where(
        (graph[unique_rows] == NULL_ID).sum(axis=1) < row_freq, NULL_ID, unique_rows
    )
    # compute_robust_pruned will handle duplicates.
    graph = graph.at[rows_to_prune].set(
        compute_robust_pruned(
            cand_ids=graph.at[rows_to_prune, :].get(mode="fill", fill_value=NULL_ID),
            dist_thresh=dist_thresh,
            m=m,
            m_max=m_max,
            node_ids=rows_to_prune,
            vecs=vecs,
        )
    )
    # TODO This requires batch_size to be less than m_max - m. The ideal operation is: group backedges by row (max batch_size columns), select those graph rows, hstack concat (as we possibly exceed m_max), then 1) for prune rows, prune then update graph 2) for non-prune rows, sort (so we can truncate to m_max and remove NULL_ID) then update graph.
    graph = insert_values_into_filler_positions(
        A=graph,
        rows=rows,
        unique_rows=unique_rows,
        values=backedges[:, 1],
    )

    return graph


# WARNING: Do not JIT this function, as otherwise it will unwrap the loop and create a giant function that takes forever to compile.
def optimize_graph(
    *,
    graph: UInt32[Array, "n m_max"],
    vecs: Float16[Array, "n d"],
    id_medoid: UInt32[Array, ""],
    m: int,
    m_max: int,
    ef: int,
    dist_thresh: Float16[Array, ""],
    update_batch_size: int,
    seed: int,
):
    n, _ = graph.shape
    rk = rand.key(seed)
    nodes = rand.permutation(rk, n).astype(np.uint32)
    pb = tqdm(total=n, desc="Optimizing graph", unit="nodes")
    for start, end, b in batches(n, update_batch_size):
        batch_nodes = nodes[start:end]  # (b,)
        graph = optimize_graph_batch(
            graph=graph,
            vecs=vecs,
            batch_nodes=batch_nodes,
            id_medoid=id_medoid,
            m=m,
            m_max=m_max,
            ef=ef,
            dist_thresh=dist_thresh,
        )
        pb.update(b)
    pb.close()
    return graph


def main():
    parser = ArgumentParser(description="Build a Vamana graph using the GPU.")
    parser.add_argument("vectors", type=str, help="Path to a packed matrix of vectors")
    parser.add_argument("--out", type=str, help="Graph output path")
    parser.add_argument("--out-medoid", type=str, help="Medoid output path")
    parser.add_argument(
        "--profile", type=str, help="Profile a batch optimization to this directory"
    )
    parser.add_argument("--dtype", type=str, help="Data type name e.g. float32")
    parser.add_argument("--dim", type=int, help="Vector dimensions")
    parser.add_argument("--m", type=int, help="Degree bound")
    parser.add_argument("--ef", type=int, help="Search list cap")
    parser.add_argument("--alpha", type=float, help="Distance threshold")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    with open(args.vectors, "rb") as f:
        # Cast to float16 for faster calculations.
        vecs = (
            np.frombuffer(f.read(), dtype=dtype)
            .reshape((-1, args.dim))
            .astype(np.float16)
        )
    n, _ = vecs.shape
    assert n < NULL_ID
    m = args.m
    m_max = m * 2
    ef = args.ef
    update_batch_size = 64
    dist_thresh = np.float16(args.alpha)
    seed = 0
    medoid_sample_size = 10_000

    print("Calculating approx. medoid")
    medoid = calc_approx_medoid(
        vecs=vecs,
        sample_size=medoid_sample_size,
        seed=seed,
    )
    print("Initializing random graph")
    graph = init_random_graph(
        n=n,
        m=m,
        m_max=m_max,
        seed=seed,
    )
    if args.profile:
        with jax.profiler.trace(args.profile, create_perfetto_link=True):
            print("Profiling")
            optimize_graph_batch(
                graph=graph,
                vecs=vecs,
                batch_nodes=arange(update_batch_size),
                id_medoid=medoid,
                m=m,
                m_max=m_max,
                ef=ef,
                dist_thresh=dist_thresh,
            ).block_until_ready()
            print("Computation complete")
    else:
        graph = optimize_graph(
            graph=graph,
            vecs=vecs,
            id_medoid=medoid,
            m=m,
            m_max=m_max,
            ef=ef,
            dist_thresh=dist_thresh,
            update_batch_size=update_batch_size,
            seed=seed,
        ).block_until_ready()
        print("Saving")
        with open(args.out, "wb") as f:
            msgpack.dump(
                {
                    i: [n for n in nodes if n != NULL_ID]
                    for i, nodes in enumerate(graph.tolist())
                },
                f,
            )
        with open(args.out_medoid, "w") as f:
            f.write(str(medoid.item()))
        print("All done!")


if __name__ == "__main__":
    main()
