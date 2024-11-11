from tqdm import tqdm
import argparse
import torch
import numpy as np

"""
We use PyTorch instead of cupy because it supports the latest versions of both ROCm and CUDA.
"""

parser = argparse.ArgumentParser(
    description="Perform similarity search on vectors using the GPU and save the results."
)
parser.add_argument("--ids", type=str, help="Path to packed array of u64 IDs")
parser.add_argument("--vectors", type=str, help="Path to packed array of f32 vectors")
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

# Load IDs and vectors from disk.
print("Loading data from disk...")
with open(args.ids, "rb") as f:
    ids = np.frombuffer(f.read(), dtype=np.uint64)  # Shape: (N,)
with open(args.vectors, "rb") as f:
    vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(
        (-1, args.dim)
    )  # Shape: (N, D)
N, D = vectors.shape
assert N == ids.shape[0]
print(f"Loaded {N} vectors of dimension {D}")

# Sample random indices.
sample_indices = np.random.choice(N, size=args.n_samples, replace=False)
with open("query_ids.bin", "wb") as f:
    f.write(ids[sample_indices].tobytes())

# Transfer data to GPU.
print("Transferring data to GPU...")
# Transpose before GPU to avoid doubling GPU memory usage.
vectors_t_gpu = torch.from_numpy(vectors.T).to(device=device, dtype=torch.float32)
out_knn = open("knn_ids.bin", "wb")

pb = tqdm(total=args.n_samples)
for i in range(0, args.n_samples, args.batch_size):
    batch_indices = sample_indices[i : i + args.batch_size]
    # The last batch may be smaller than batch_size.
    batch_n = batch_indices.shape[0]
    # Avoid using CPU `vectors` (excessive copying) when vectors already exist in GPU.
    query_vectors_gpu = vectors_t_gpu[:, batch_indices].T
    assert query_vectors_gpu.shape == (batch_n, D)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Calculate cosine similarity
        similarities = torch.mm(query_vectors_gpu, vectors_t_gpu)  # Shape: (batch_size, N)

        # Get top k nearest neighbors
        _, top_k_indices = torch.topk(similarities, k=args.k, dim=1)

    # Convert indices back to IDs
    result_ids = ids[top_k_indices.numpy(force=True)]
    assert result_ids.shape == (batch_n, args.k)
    assert result_ids.dtype == np.uint64
    out_knn.write(result_ids.tobytes())
    pb.update(batch_n)
pb.close()
print("All done!")
