from matplotlib.colors import to_rgba
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns


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
            np.frombuffer(raw, dtype=np.float32, count=dim, offset=raw_vec_len * i + 4)
            for i in range(n)
        ]
    )


def plot_distribution(
    data,
    output_path,
    title="Distribution plot",
    xlabel="Value",
    ylabel="Count",
    bins=150,
    kde=False,
    color="blue",
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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save as WebP with high quality
    plt.savefig(output_path, format="webp", dpi=dpi)
    plt.close()


def plot_distributions(
    data_arrays,
    output_path,
    title,
    bins=150,
    alpha=0.3,
    xlabel="Value",
    ylabel="Frequency",
    figsize=(12, 8),
    dpi=300,
):
    """
    Plot multiple overlapping histograms with automatically generated colors and transparency.

    Parameters:
    -----------
    data_arrays : list of numpy arrays
        List containing the data arrays to plot as histograms
    bins : int or array-like, optional (default=50)
        Number of bins or bin edges for the histograms
    alpha : float, optional (default=0.3)
        Transparency level for each histogram
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    output_path : str, optional
        Output file path (must end in .webp)
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        Resolution of the output image
    """

    plt.figure(figsize=figsize)

    # Generate evenly spaced hues for distinct colors
    n_arrays = len(data_arrays)
    hues = np.linspace(0, 1, n_arrays, endpoint=False)

    # Find global range for consistent bins
    all_data = np.concatenate(data_arrays)
    global_min, global_max = np.min(all_data), np.max(all_data)
    if isinstance(bins, int):
        bins = np.linspace(global_min, global_max, bins)

    # Plot each histogram
    for idx, data in enumerate(data_arrays):
        # Generate color with consistent saturation and value
        rgb = colorsys.hsv_to_rgb(hues[idx], 0.8, 0.9)
        color = to_rgba(rgb, alpha)

        plt.hist(
            data,
            bins=bins,
            alpha=alpha,
            color=color,
            label=f"Dataset {idx+1}",
            density=True,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add legend if there aren't too many datasets
    if n_arrays <= 20:  # Only show legend for 20 or fewer datasets
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save as WebP with high quality
    plt.savefig(output_path, format="webp", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_time_series(
    arrays,
    output_path,
    title="Time Series Plot",
    dpi=300,
    xlabel="Time",
    ylabel="Value",
):
    # Convert all arrays to numpy arrays if they aren't already
    arrays = [np.array(arr) for arr in arrays]

    # Create time points (x-axis)
    max_length = max(len(arr) for arr in arrays)
    time_points = np.arange(max_length)

    # Set up the plot style
    plt.figure(figsize=(12, 6))

    # Plot individual series with reduced opacity
    for i, arr in enumerate(arrays):
        plt.plot(time_points[: len(arr)], arr, alpha=0.3)

    # Calculate and plot the average
    # First, pad shorter arrays with NaN to handle different lengths
    padded_arrays = [
        np.pad(
            arr.astype(np.float64), (0, max_length - len(arr)), constant_values=np.nan
        )
        for arr in arrays
    ]
    stacked_arrays = np.stack(padded_arrays)
    average = np.nanmean(stacked_arrays, axis=0)

    # Plot average line
    plt.plot(time_points, average, "r-", linewidth=2, label="Average")

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save as WebP with high quality
    plt.savefig(output_path, format="webp", dpi=dpi, bbox_inches="tight")
    plt.close()
