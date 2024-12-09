from greedy_search import greedy_search
from index import init_random_graph
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from robust_prune import compute_robust_pruned
from typing import List
from util import NULL_ID
import jax.numpy as np
import jax.random as rand


def ref_compute_robust_pruned(
    vecs: Float16[Array, "n d"],
    p: int,
    candidates: List[int],
    distance_threshold: Float16[Array, ""],
    degree_bound: int,
):
    candidates.sort(key=lambda n: norm(vecs[p] - vecs[n]))
    new_neighbors: List[int] = []
    i = 0
    while candidates:
        i += 1
        p_star = candidates.pop(0)
        new_neighbors.append(p_star)
        if len(new_neighbors) == degree_bound:
            break
        new_candidates = []
        pruned = []
        for p_prime in candidates:
            if distance_threshold * norm(vecs[p_star] - vecs[p_prime]) <= norm(
                vecs[p] - vecs[p_prime]
            ):
                pruned.append(p_prime)
                continue
            new_candidates.append(p_prime)
        candidates = new_candidates
    return new_neighbors


def test_compute_robust_pruned():
    n = 15_000
    degree_bound = 32
    d = 128
    node = 0
    seed = 0
    dist_thresh = np.float16(1.1)
    candidate_count = 70
    start = 0
    query = 7777
    g = init_random_graph(n=n, m=degree_bound, seed=seed)
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    candidates = greedy_search(
        vecs=vecs,
        id_start=start,
        graph=g,
        id_targets=np.array([query]),
        k=None,
        ef=degree_bound,
        iterations=candidate_count,
    ).squeeze()

    exp = ref_compute_robust_pruned(
        vecs=vecs,
        p=node,
        candidates=candidates.tolist(),
        distance_threshold=dist_thresh,
        degree_bound=degree_bound,
    )
    got = [
        v
        for v in compute_robust_pruned(
            vecs=vecs,
            node_ids=np.array([node]),
            cand_ids=candidates[None, :],
            m=degree_bound,
            dist_thresh=dist_thresh,
        )
        .squeeze()
        .tolist()
        if v != NULL_ID.item()
    ]
    assert got == exp
