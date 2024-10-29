import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns

def read_vector(path: str, dtype: npt.DTypeLike) -> np.ndarray:
    elem_bytes = np.dtype(dtype).itemsize
    with open(path, "rb") as f:
        raw = f.read()
    dim = int.from_bytes(raw[:4], byteorder="little")
    raw_vec_len = (4 + dim * elem_bytes)
    n = len(raw) // raw_vec_len
    return np.vstack([
        # Add 4 to skip past leading dim. uint32.
        np.frombuffer(raw, dtype=np.float32, count=dim, offset=raw_vec_len * i + 4)
        for i in range(n)
    ])


def plot_distribution(
  data,
  output_path,
  title='Distribution plot',
  bins=150,
  kde=False,
  color='blue',
  figsize=(15, 10),
  dpi=300,
):
    """
    Create a distribution plot of float values and save it as a WebP image.

    Parameters:
    -----------
    data : array-like
        The float values to plot
    output_path : str
        Path where the WebP file will be saved
    title : str
        Title of the plot
    bins : int
        Number of histogram bins
    kde : bool
        Whether to show the kernel density estimate
    color : str
        Color of the plot
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution of the output image
    """

    # Create a new figure with the specified size
    plt.figure(figsize=figsize)

    # Create the distribution plot using seaborn
    sns.histplot(data=data, bins=bins, kde=kde, color=color)

    # Customize the plot
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Count')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save as WebP with high quality
    plt.savefig(output_path, format='webp', dpi=dpi)
    plt.close()
