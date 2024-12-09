from functools import partial
from jax import jit
from jaxtyping import Array
from jaxtyping import Float16
from jaxtyping import UInt32
from typing import Tuple
import jax.numpy as np

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
def pair_matrix_vector(
    M: UInt32[Array, "b m"], V: UInt32[Array, "b"]
) -> UInt32[Array, "b*m 2"]:
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
    # Padded with NULL_ID up to r_max, which should be <= k.
    unique_rows: UInt32[Array, "r_max"],
    values: UInt32[Array, "k"],
):
    n, m = A.shape
    (r_max,) = unique_rows.shape

    # Extract subset; since we pad with NULL_ID, make sure to drop out-of-bounds rows
    A_subset = A.at[unique_rows].get(mode="drop")  # shape (r, m)

    # Find filler positions only in the relevant subset
    # nonzero returns int32, not unsigned, so make sure NULL_ID is int32.max, not uint32.max which overflows to -1 which is a legal index.
    sub_fill_rows, sub_fill_cols = np.nonzero(
        A_subset == NULL_ID, size=r_max * m, fill_value=NULL_ID
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
