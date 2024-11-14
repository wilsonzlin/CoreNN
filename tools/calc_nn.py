from tqdm import tqdm
import argparse
import numpy as np
import torch

"""
We use PyTorch instead of cupy because it supports the latest versions of both ROCm and CUDA.
"""

parser = argparse.ArgumentParser(
    description="Perform similarity search on vectors using the GPU and save the results."
)
parser.add_argument(
    "--ids",
    type=str,
    help="Path to packed array of u32 IDs; if omitted, will use [0, N-1]",
)
parser.add_argument("--vectors", type=str, help="Path to packed array of f32 vectors")
parser.add_argument("--out-queries", type=str, help="Path to queries output")
parser.add_argument("--out-results", type=str, help="Path to results output")
parser.add_argument("--dim", type=int, help="Dimension of vectors")
parser.add_argument("--n-samples", type=int, help="Number of vectors to sample")
parser.add_argument(
    "--batch-size",
    type=int,
    help="Batch size for processing (higher values use more GPU VRAM)",
)
parser.add_argument(
    "--k", type=int, help="Number of nearest neighbors to find per vector"
)
args = parser.parse_args()

# Set device to GPU if available (works with both CUDA and ROCm).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vectors from disk.
print("Loading data from disk...")
with open(args.vectors, "rb") as f:
    vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(
        (-1, args.dim)
    )  # Shape: (N, D)
N, D = vectors.shape
print(f"Loaded {N} vectors of dimension {D}")

if args.ids is not None:
    with open(args.ids, "rb") as f:
        ids = np.frombuffer(f.read(), dtype=np.uint32)
else:
    ids = np.arange(N, dtype=np.uint32)  # Shape: (N,)
assert (N,) == ids.shape

# Sample random indices.
sample_indices = np.random.choice(N, size=args.n_samples, replace=False)
with open(args.out_queries, "wb") as f:
    f.write(vectors[sample_indices].tobytes())

# Transfer data to GPU.
print("Transferring data to GPU...")
vectors_gpu = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
out_knn = open(args.out_results, "wb")

pb = tqdm(total=args.n_samples)
for i in range(0, args.n_samples, args.batch_size):
    batch_indices = sample_indices[i : i + args.batch_size]
    # The last batch may be smaller than batch_size.
    batch_n = batch_indices.shape[0]
    # Avoid using CPU `vectors` (excessive copying) when vectors already exist in GPU.
    query_vectors_gpu = vectors_gpu[batch_indices, :]
    assert query_vectors_gpu.shape == (batch_n, D)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Calculate Euclidean distance (works like cosine distance for normalized vectors too).
        distances = torch.cdist(
            query_vectors_gpu, vectors_gpu
        )  # Shape: (batch_size, N)

        # Get top k nearest neighbors
        _, top_k_indices = torch.topk(distances, k=args.k, dim=1, largest=False)

    # Convert indices back to IDs
    result_ids = ids[top_k_indices.numpy(force=True)]
    assert result_ids.shape == (batch_n, args.k)
    assert result_ids.dtype == np.uint32
    out_knn.write(result_ids.tobytes())
    pb.update(batch_n)
pb.close()
print("All done!")
