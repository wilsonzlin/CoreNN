from tqdm import tqdm
import argparse
import torch
import numpy as np
import numpy.typing as npt

"""
We use PyTorch instead of cupy because it supports the latest versions of both ROCm and CUDA.
"""

parser = argparse.ArgumentParser(
    description="Perform similarity search on vectors using the GPU and save the results."
)
parser.add_argument("--vectors", type=str, help="Path to packed array of f32 vectors")
parser.add_argument(
    "--batch-size",
    type=int,
    help="Batch size for processing (higher values use more GPU VRAM)",
)
args = parser.parse_args()

# Set device to GPU if available (works with both CUDA and ROCm).
device = torch.device("cuda")

def read_vectors(path: str, dtype: npt.DTypeLike) -> np.ndarray:
    elem_bytes = np.dtype(dtype).itemsize
    with open(path, "rb") as f:
        raw = f.read()
    dim = int.from_bytes(raw[:4], byteorder="little")
    raw_vec_len = 4 + dim * elem_bytes
    n = len(raw) // raw_vec_len
    return np.vstack(
        [
            # Add 4 to skip past leading dim. uint32.
            np.frombuffer(raw, dtype=dtype, count=dim, offset=raw_vec_len * i + 4)
            for i in range(n)
        ]
    )

# Load IDs and vectors from disk.
print("Loading data from disk...")
vectors = read_vectors(args.vectors, np.float32) # Shape: (N, D)
N, D = vectors.shape
print(f"Loaded {N} vectors of dimension {D}")
vectors = vectors[np.random.choice(N, 250_000, replace=False)]
N, D = vectors.shape
with open(f"vectors.f32-{D}d.bin", "wb") as f:
    f.write(vectors.tobytes())
print(f"Sampled {N} vectors of dimension {D}")

# Transfer data to GPU.
print("Transferring data to GPU...")
# Transpose before GPU to avoid doubling GPU memory usage.
vectors_t_gpu = torch.from_numpy(vectors.T).to(device=device, dtype=torch.float32)
# We can fit more dists in memory with f16, and f16 is not that worse accuracy than f32 for dists.
out_dists = open("dists.f16.bin", "wb")

pb = tqdm(total=N)
for i in range(0, N, args.batch_size):
    # Avoid using CPU `vectors` (excessive copying) when vectors already exist in GPU.
    query_vectors_gpu = vectors_t_gpu[:, i : i + args.batch_size].T
    # The last batch may be smaller than batch_size.
    batch_n = query_vectors_gpu.shape[0]
    assert query_vectors_gpu.shape == (batch_n, D)

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Calculate cosine similarity
        similarities = torch.mm(query_vectors_gpu, vectors_t_gpu)  # Shape: (batch_size, N)
        dists = 1 - similarities

    out_dists.write(dists.numpy(force=True).astype(np.float16).tobytes())
    pb.update(batch_n)
pb.close()
print("All done!")
