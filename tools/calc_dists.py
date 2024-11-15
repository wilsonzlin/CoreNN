from tqdm import tqdm
import argparse
import numpy as np
import torch

"""
We use PyTorch instead of cupy because it supports the latest versions of both ROCm and CUDA.
"""

parser = argparse.ArgumentParser(
    description="Calculate pairwise Euclidean distances between all vectors."
)
parser.add_argument("--vectors", type=str, help="Path to packed array of f32 vectors")
parser.add_argument("--out", type=str, help="Path to output")
parser.add_argument("--dim", type=int, help="Dimension of vectors")
parser.add_argument(
    "--batch-size",
    type=int,
    help="Batch size for processing (higher values use more GPU VRAM)",
)
args = parser.parse_args()

# Set device to GPU if available (works with both CUDA and ROCm).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vectors from disk.
print("Loading data from disk...")
with open(args.vectors, "rb") as f:
    vectors = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, args.dim))
N, D = vectors.shape
print(f"Loaded {N} vectors of dimension {D}")

# Transfer data to GPU.
print("Transferring data to GPU...")
vectors_gpu = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
# We can fit more dists in memory with f16, and f16 is not that worse accuracy than f32 for dists.
out_dists = open(args.out, "wb")

pb = tqdm(total=N)
for i in range(0, N, args.batch_size):
    # Avoid using CPU `vectors` (excessive copying) when vectors already exist in GPU.
    query_vectors_gpu = vectors_gpu[i : i + args.batch_size]
    # The last batch may be smaller than batch_size.
    batch_n = query_vectors_gpu.shape[0]
    assert query_vectors_gpu.shape == (batch_n, D)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Calculate Euclidean distance (works like cosine distance for normalized vectors too).
        dists = torch.cdist(query_vectors_gpu, vectors_gpu)  # Shape: (batch_n, N)

    out_dists.write(dists.numpy(force=True).astype(np.float16).tobytes())
    pb.update(batch_n)
pb.close()
print("All done!")
