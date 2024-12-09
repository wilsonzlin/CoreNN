from greedy_search import greedy_search
from index import init_random_graph
from jax.numpy.linalg import norm
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from typing import List
from typing import Set
from typing import Tuple
from util import NULL_ID
import jax.numpy as np
import jax.random as rand


# Reference implementation of GreedySearch, according to the paper. Intentionally unoptimized to keep as close to correct reference code as possible.
# We compare this to our variation to see any differences in results.
def ref_golden_greedy_search(
    *,
    vecs: Float16[Array, "n d"],
    start: int,
    g: UInt32[Array, "n m"],
    q: int,
    k: int,
    search_list_cap: int,
) -> Tuple[List[int], List[int]]:
    search_list = [(start, norm(vecs[q] - vecs[start]))]
    seen: Set[int] = set([start])
    visited: List[int] = []
    while True:
        p_star = next(filter(lambda x: x[0] not in visited, search_list), None)
        if p_star is None:
            break
        p_star_id, _ = p_star
        for n in g[p_star_id].tolist():
            if n not in seen:
                search_list.append((n, norm(vecs[q] - vecs[n])))
                seen.add(n)
        visited.append(p_star_id)
        search_list.sort(key=lambda o: o[1])
        del search_list[search_list_cap:]
    return (
        [x for x, _ in search_list[:k]],
        visited,
    )


def ref_greedy_search(
    *,
    vecs: Float16[Array, "n d"],
    start: int,
    g: UInt32[Array, "n m"],
    q: int,
    k: int,
    search_list_cap: int,
    search_iter: int,
) -> Tuple[List[int], List[int]]:
    search_list: List[Tuple[int, float]] = []
    search_list.append((start, norm(vecs[q] - vecs[start])))
    seen: Set[int] = set([start])
    visited: List[Tuple[int, float]] = []
    for _ in range(search_iter):
        p_star = search_list.pop(0) if search_list else None
        if p_star is None:
            break
        p_star_id, _ = p_star
        for n in g[p_star_id].tolist():
            if n not in seen:
                search_list.append((n, norm(vecs[q] - vecs[n])))
                seen.add(n)
        visited.append(p_star)
        search_list.sort(key=lambda o: o[1])
        del search_list[search_list_cap:]
    return (
        [x for x, _ in sorted(visited, key=lambda x: x[1])[:k]],
        [x for x, _ in visited],
    )


def test_greedy_search_top_k():
    n = 15_000
    degree_bound = 12
    d = 128
    search_list_cap = 100
    search_iter = 150
    k = 10
    start = 0
    query = 7777
    seed = 0
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    g = init_random_graph(n=n, m=degree_bound, seed=seed)

    exp = set(
        ref_greedy_search(
            vecs=vecs,
            start=start,
            g=g,
            q=query,
            k=k,
            search_list_cap=search_list_cap,
            search_iter=search_iter,
        )[0]
    )
    got = set(
        greedy_search(
            vecs=vecs,
            id_start=start,
            graph=g,
            id_targets=np.array([query]),
            k=k,
            ef=search_list_cap,
            iterations=search_iter,
        )
        .squeeze()
        .tolist()
    )
    assert exp == got


def test_greedy_search_visited():
    n = 15_000
    degree_bound = 12
    d = 128
    search_list_cap = 100
    search_iter = 150
    start = 0
    query = 7777
    seed = 0
    vecs = rand.uniform(
        rand.PRNGKey(seed), (n, d), dtype=np.float16, minval=-1, maxval=1
    )
    g = init_random_graph(n=n, m=degree_bound, seed=seed)

    exp = ref_greedy_search(
        vecs=vecs,
        start=start,
        g=g,
        q=query,
        k=0,
        search_list_cap=search_list_cap,
        search_iter=search_iter,
    )[1]
    ideal = ref_golden_greedy_search(
        vecs=vecs,
        start=start,
        g=g,
        q=query,
        k=0,
        search_list_cap=search_list_cap,
    )[1]
    got = [
        v
        for v in greedy_search(
            vecs=vecs,
            id_start=start,
            graph=g,
            id_targets=np.array([query]),
            k=None,
            ef=search_list_cap,
            iterations=search_iter,
        )
        .squeeze()
        .tolist()
        if v != NULL_ID
    ]
    assert exp == got
    # With enough search iterations, the visited nodes of ours should be a superset.
    assert set(ideal) == set(got)
