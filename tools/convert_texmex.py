import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Convert corpus-texmex.irisa.fr format vectors into packed arrays."
)
parser.add_argument("input", type=str, help="Path to corpus-texmex.irisa.fr vectors")
parser.add_argument(
    "--dtype", type=str, help="One of {float{16,32,64},{u,}int{8,16,32,64}}"
)
parser.add_argument("--out", type=str, help="Path to output")
args = parser.parse_args()

dtype = np.dtype(args.dtype)
elem_bytes = dtype.itemsize
with open(args.input, "rb") as f:
    raw = f.read()
dim = int.from_bytes(raw[:4], byteorder="little")
raw_vec_len = 4 + dim * elem_bytes
n = len(raw) // raw_vec_len
mat = np.vstack(
    [
        # Add 4 to skip past leading dim. uint32.
        np.frombuffer(raw, dtype=dtype, count=dim, offset=raw_vec_len * i + 4)
        for i in range(n)
    ]
)
with open(args.out, "wb") as f:
    f.write(mat.tobytes())
