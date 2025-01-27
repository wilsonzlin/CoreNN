from functools import partial
from jax import jit
from jaxtyping import Array
from jaxtyping import BFloat16
from jaxtyping import UInt32
from typing import Tuple
import jax.numpy as np

# Use int32.max as some functions use/return int32, which if we provide the uint32 max value, will overflow to -1, which is a legal index.
# We also sometimes need int32 arrays ourselves and will use this value in them.
NULL_ID = np.uint32(np.iinfo(np.int32).max)


def read_vecs(path: str, dim: int, dtype: np.dtype):
    with open(path, "rb") as f:
        # Cast to bfloat16 for faster calculations.
        # WARNING: Do not use float16, it only has a range of +/-65504, which means a max dist of +/-sqrt(65504) => +/-255 (as we use L2 norms) which is too small.
        return (
            np.frombuffer(f.read(), dtype=dtype).reshape((-1, dim)).astype(np.bfloat16)
        )


# By default, np.arange returns int32 which causes a lot of issues with JAX.
@partial(jit, static_argnames=["n"])
def arange(n: int):
    return np.arange(n, dtype=np.uint32)


def flatten_by_anti_diagonals(arr):
    """
    Flatten a 2D array by anti-diagonals.
    Example:
      [[1 2 3]
       [4 5 6]
       [7 8 9]]
    will be flattened as:
      [1 4 2 7 5 3 8 6 9]
    """
    # arr is of shape (N, M)
    N, M = arr.shape
    i, j = np.indices((N, M))
    diag_sum = i + j

    # We want to sort first by diag_sum (ascending),
    # and then by i (descending) for elements with the same diag_sum.
    # Sorting by i descending is equivalent to sorting by -i ascending.
    order = np.lexsort((-i.ravel(), diag_sum.ravel()))

    return arr.ravel()[order]


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


@partial(jit, static_argnames=["d", "u"])
def group_by_y_id_limit(
    M_id: UInt32[Array, "x"],
    M: UInt32[Array, "x y"],
    d: int,
    # This is provided in case you know for sure that there are less than x * y maximum possible unique M IDs.
    u: int,
) -> Tuple[UInt32[Array, "u"], UInt32[Array, "u d"]]:
    """
    Same as group_by_y_id, but more memory efficient by limiting max. group size per distinct ID to `d`
    and not using a full u * x * y mask.
    """
    x, y = M.shape
    M_flat = M.reshape(-1)

    V, inv, counts = np.unique(
        M_flat, return_inverse=True, return_counts=True, size=u, fill_value=NULL_ID
    )
    # V: shape (u,), unique values sorted
    # inv: shape (x*y,), maps each M_flat entry to an index in [0, u-1]
    # counts: shape (u,), number of occurrences for each unique value

    # Sort inv to group identical values together
    # sorted_idx is the permutation that sorts inv, so elements with the same inv value are consecutive
    sorted_idx = np.argsort(inv)

    # Compute the starting offset for each unique value block
    # offsets[i] = sum(counts[:i])
    offsets = np.concatenate(
        [np.array([0], dtype=counts.dtype), np.cumsum(counts[:-1])]
    )

    # We want R of shape (u, d)
    # For each unique value i:
    #   block = sorted_idx[offsets[i] : offsets[i]+counts[i]]
    # We take up to d occurrences from this block.

    J = arange(d)  # (d,)
    # Ensure we never index out-of-bounds by clipping J to counts[i]-1
    # This ensures we always pick a valid index for each value block, even if counts[i]<d.
    # We'll mask out the invalid ones anyway.
    clipped_J = np.minimum(J[None, :], counts[:, None] - 1)

    # Mask to indicate which positions are valid (have occurrences)
    mask = J[None, :] < counts[:, None]

    # Compute the indices in sorted_idx for these occurrences
    idx_block = offsets[:, None] + clipped_J
    # Gather the row indices
    row_indices = sorted_idx[idx_block] // y

    # Fill invalid entries with NULL_ID where mask is False
    R = np.where(mask, row_indices, NULL_ID)

    return V, M_id.at[R].get(mode="fill", fill_value=NULL_ID)


# Handles NULL_ID indices.
@jit
def select_nodes(
    graph: UInt32[Array, "n d"],
    indices: np.ndarray,
):
    return graph.at[indices].get(mode="fill", fill_value=NULL_ID)


# Helper function for indexing into the `vecs` 2D array.
# Handles NULL_ID indices.
@jit
def select_vecs(
    vecs: BFloat16[Array, "n d"],
    indices: np.ndarray,
):
    return vecs.at[indices].get(mode="fill", fill_value=np.nan)
