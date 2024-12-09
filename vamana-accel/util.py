from functools import partial
from jax import jit
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from typing import Tuple
import jax.numpy as np

# Use int32.max as some functions use/return int32, which if we provide the uint32 max value, will overflow to -1, which is a legal index.
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
def group_by_y_id(
    M_id: UInt32[Array, "x"],
    M: UInt32[Array, "x y"],
) -> Tuple[UInt32[Array, "k"], UInt32[Array, "k x"]]:
    """
    Group rows by the distinct IDs appearing in M's columns.

    Given:
      - M: a matrix of shape (x, y) containing opaque uint32 IDs. Duplicate IDs within a row are ignored.
      - M_id: a vector of shape (x,) containing one "ID" per row in M.

    Returns:
      - N_id: shape (k_max,), the sorted k unique IDs appearing in M, padded to k_max with NULL_ID.
      - N: shape (k_max, x), where N[i] contains the M_id values of all rows in M where N_id[i] appears.
        Rows where N_id[i] does not appear are filled with NULL_ID.
        NOTE: The NULL_ID values in N can be in any position, not just at the end.
        Padding rows (rows after the kth) contain junk and must not be used.

    Example:
      M:
        [13 18 23]
        [22 18 25]
        [40 13 18]
        [77 11 18]
        [22 18 13]
      Input M_id:
        [5 7 6 2 8]
      Unique IDs (N_id):
        [11 13 18 22 23 25 40 77]
      Resulting N:
        [2147483647 2147483647 2147483647          2 2147483647]
        [         5 2147483647          6 2147483647          8]
        [         5          7          6          2          8]
        [2147483647          7 2147483647 2147483647          8]
        [         5 2147483647 2147483647 2147483647 2147483647]
        [2147483647          7 2147483647 2147483647 2147483647]
        [2147483647 2147483647          6 2147483647 2147483647]
        [2147483647 2147483647 2147483647          2 2147483647]
    """
    # The worst case scenario is that every element in M (x, y) is unique, so there are x * y groups of 1.
    k_max = M.shape[0] * M.shape[1]
    N_id = np.unique(M, size=k_max, fill_value=NULL_ID)
    mask = M[None, :, :] == N_id[:, None, None]
    row_mask = np.any(mask, axis=2)
    N = np.where(row_mask, M_id[None, :], NULL_ID)
    return N_id, N  # type: ignore


# Handles NULL_ID indices.
@jit
def select_nodes(
    graph: UInt32[Array, "n d"],
    indices: np.ndarray,
):
    return graph.at[indices].get(mode="fill", fill_value=NULL_ID)


# Handles NULL_ID indices.
@jit
def select_vecs(
    vecs: Float16[Array, "n d"],
    indices: np.ndarray,
):
    return vecs.at[indices].get(mode="fill", fill_value=np.nan)
