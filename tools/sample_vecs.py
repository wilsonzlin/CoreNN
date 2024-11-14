import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Sample a subset of vectors.")
parser.add_argument("input", type=str, help="Path to vectors")
parser.add_argument("--dim", type=int, help="Dimension of vectors")
parser.add_argument("--n", type=int, help="Sample size")
parser.add_argument("--out", type=str, help="Path to output")
args = parser.parse_args()


print("Loading data from disk")
with open(args.input, "rb") as f:
    vectors = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, args.dim))
N, _ = vectors.shape
print("Sampling")
sample_indices = np.random.choice(N, size=args.n, replace=False)
print("Writing")
with open(args.out, "wb") as f:
    f.write(vectors[sample_indices].tobytes())
