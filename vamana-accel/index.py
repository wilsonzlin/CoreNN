from functools import partial
from greedy_search import greedy_search
from jax import jit
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
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
    vecs: Float16[Array, "n d"],
    batch_nodes: UInt32[Array, "b"],
    id_medoid: UInt32[Array, ""],
    m: int,
    ef: int,
    search_iter: int,
    dist_thresh: Float16[Array, ""],
):
    visited = greedy_search(
        graph=graph,
        vecs=vecs,
        id_targets=batch_nodes,
        k=None,
        ef=ef,
        iterations=search_iter,
        id_start=id_medoid,
    )  # (b, ef)
    new_neighbors = compute_robust_pruned(
        cand_ids=np.hstack([visited, graph[batch_nodes, :]]),
        dist_thresh=dist_thresh,
        m=m,
        node_ids=batch_nodes,
        vecs=vecs,
    )  # (b, m)
    graph = graph.at[batch_nodes].set(new_neighbors)

    # The maximum possible distinct backedge target nodes is b * ef, where every batch node has all distinct neighbors not shared by any other batch node.
    back_nodes, back_add_neighbors = group_by_y_id(
        M_id=batch_nodes, M=new_neighbors
    )  # (b * ef, b)
    back_new_neighbors = np.hstack(
        [select_nodes(graph, back_nodes), back_add_neighbors]
    )  # (b * ef, m + b)
    # While we could have a mechanism where we only prune upon reaching m_max, that would mean dynamic computation: selective compute_robust_pruned. This isn't possible under JIT; even if we mask rows not needing prune with NULL_ID, compute_robust_pruned will still do the fixed amount of calculations anyway. Therefore, we always compute_robust_pruned all rows every batch.
    # compute_robust_pruned will handle duplicates.
    graph = graph.at[back_nodes].set(
        compute_robust_pruned(
            cand_ids=back_new_neighbors,
            dist_thresh=dist_thresh,
            m=m,
            node_ids=back_nodes,
            vecs=vecs,
        )
    )

    return graph


# WARNING: Do not JIT this function, as otherwise it will unwrap the loop and create a giant function that takes forever to compile.
def optimize_graph(
    *,
    graph: UInt32[Array, "n m"],
    vecs: Float16[Array, "n d"],
    id_medoid: UInt32[Array, ""],
    m: int,
    ef: int,
    search_iter: int,
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
            ef=ef,
            search_iter=search_iter,
            dist_thresh=dist_thresh,
        )
        pb.update(b)
    pb.close()
    return graph
