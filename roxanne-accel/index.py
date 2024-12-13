from functools import partial
from greedy_search import greedy_search_ids
from jax import jit
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import BFloat16
from jaxtyping import UInt32
from robust_prune import compute_robust_pruned
from tqdm import tqdm
from util import arange
from util import batches
from util import group_by_y_id
from util import NULL_ID
from util import select_nodes
import jax.numpy as np
import jax.random as rand


@partial(jit, static_argnames=["sample_size", "seed"])
def calc_approx_medoid(
    *,
    vecs: BFloat16[Array, "n d"],
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


@partial(jit, static_argnames=["n", "m", "seed"])
def init_random_graph(
    *,
    n: int,
    m: int,
    seed: int,
):
    rk = rand.key(seed)
    # Duplicates are fine, we'll compute_robust_pruned them out later.
    graph = rand.choice(rk, arange(n), shape=(n, m)).astype(np.uint32)
    # Remove self-edges.
    row_indices = arange(n)[:, None]
    graph = np.where(graph == row_indices, NULL_ID, graph)
    return graph


@partial(
    jit,
    static_argnames=["m", "ef", "search_iter"],
)
def optimize_graph_batch(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    batch_nodes: UInt32[Array, "b"],
    id_medoid: UInt32[Array, ""],
    m: int,
    ef: int,
    search_iter: int,
    dist_thresh: BFloat16[Array, ""],
):
    # NOTE: While the original greedy_search doesn't have a fixed search_iter and can drop visited from the search list, so it seems like we could get very early distant nodes that would otherwise be pruned, we only want the set of visited nodes (as per the official insert/optimize algorithm), not the nearest ef/search_iter visited nodes. For example, even in the original implementation, the medoid is always returned.
    visited = greedy_search_ids(
        graph=graph,
        vecs=vecs,
        id_targets=batch_nodes,
        k=None,
        ef=ef,
        iterations=search_iter,
        id_start=id_medoid,
    )  # (b, search_iter)
    # Always prune as we have more than m candidates, unless search_iter is set to a tiny value less than m which should never be done.
    new_neighbors = compute_robust_pruned(
        # Include visited as there may already have backedges added from previous iterations.
        cand_ids=np.hstack([visited, graph[batch_nodes]]),
        dist_thresh=dist_thresh,
        m=m,
        node_ids=batch_nodes,
        vecs=vecs,
    )  # (b, m)
    graph = graph.at[batch_nodes].set(new_neighbors)

    # The maximum possible distinct backedge target nodes is b * search_iter, where every batch node has all distinct neighbors not shared by any other batch node.
    back_nodes, back_add_neighbors = group_by_y_id(
        M_id=batch_nodes, M=new_neighbors
    )  # (b * search_iter, b)
    back_new_neighbors = np.hstack(
        [select_nodes(graph, back_nodes), back_add_neighbors]
    )  # (b * search_iter, m + b)
    # Move NULL_IDs to end so we can truncate to m if we're not pruning.
    sort_i = np.argsort(back_new_neighbors, axis=1)
    back_new_neighbors = back_new_neighbors[
        arange(back_new_neighbors.shape[0])[:, None], sort_i
    ]
    # Figure out which nodes need pruning.
    # This is not a speed optimization: conditionally pruning would mean dynamic computation, not possible under JIT. We always do the same amount of computation here.
    # Instead, this is for accuracy: we don't want to prune a node too early (i.e. before it has m out-neighbors), as that may remove vital edges.
    back_needs_prune = (back_new_neighbors != NULL_ID).sum(
        axis=1
    ) >= m  # (b * search_iter,)
    back_new_neighbors_pruned = compute_robust_pruned(
        cand_ids=np.where(back_needs_prune[:, None], back_new_neighbors, NULL_ID),
        dist_thresh=dist_thresh,
        m=m,
        node_ids=np.where(back_needs_prune, back_nodes, NULL_ID),
        vecs=vecs,
    )
    back_new_neighbors = np.where(
        back_needs_prune[:, None], back_new_neighbors_pruned, back_new_neighbors[:, :m]
    )
    graph = graph.at[back_nodes].set(back_new_neighbors)

    return graph


# WARNING: Do not JIT this function, as otherwise it will unwrap the loop and create a giant function that takes forever to compile.
def optimize_graph(
    *,
    graph: UInt32[Array, "n m"],
    vecs: BFloat16[Array, "n d"],
    id_medoid: UInt32[Array, ""],
    m: int,
    ef: int,
    search_iter: int,
    dist_thresh: BFloat16[Array, ""],
    update_batch_size: int,
    seed: int,
):
    n, _ = graph.shape
    rk = rand.key(seed)
    nodes = rand.permutation(rk, n).astype(np.uint32)

    print("Compiling optimizer")
    for b in [update_batch_size, n % update_batch_size]:
        if not b:
            continue
        optimize_graph_batch.lower(
            graph=graph,
            vecs=vecs,
            batch_nodes=nodes[:b],
            id_medoid=id_medoid,
            m=m,
            ef=ef,
            search_iter=search_iter,
            dist_thresh=dist_thresh,
        ).compile()

    pb = tqdm(total=n, desc="Optimizing graph", unit="nodes")
    for start, end, b in batches(n, update_batch_size):
        batch_nodes = nodes[start:end]  # (b,)
        graph = optimize_graph_batch(
            graph=graph,
            vecs=vecs,
            batch_nodes=batch_nodes,
            id_medoid=id_medoid,
            m=m,
            ef=ef,
            search_iter=search_iter,
            dist_thresh=dist_thresh,
        )
        pb.update(b)
    graph = graph.block_until_ready()
    pb.close()
    return graph
