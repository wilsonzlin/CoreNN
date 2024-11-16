from matplotlib.colors import to_rgba
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_colors(n: int):
    """
    Generate n distinct colors by evenly spacing them around the HSV color wheel.
    Returns RGB colors with full saturation and value.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly space hues between 0 and 1
        # Convert HSV to RGB (using full saturation and value)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)
    return colors


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


def plot_distributions_as_lines(
    datasets,
    output_path,
    labels=None,
    colors=None,
    bins=200,
    title="Distribution Comparison",
    xlabel="Value",
    ylabel="Density",
):
    plt.figure(figsize=(10, 6))

    # Default labels if none provided
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]

    # Default colors if none provided
    if colors is None:
        colors = generate_colors(len(datasets))

    # Find global min and max for consistent x-range
    all_data = np.concatenate(datasets)
    global_min = np.min(all_data)
    global_max = np.max(all_data)

    # Create consistent bin edges for all datasets
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot each distribution
    for data, label, color in zip(datasets, labels, colors):
        # Calculate histogram
        hist, _ = np.histogram(data, bins=bin_edges)

        # Plot the histogram as a smooth line
        plt.plot(
            bin_centers,
            hist,
            label=label,
            color=color,
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_series(
    arrays,
    output_path,
    average=True,
    labels=None,
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
        plt.plot(
            time_points[: len(arr)], arr, alpha=0.3, label=labels[i] if labels else None
        )

    if average:
        # Calculate and plot the average
        # First, pad shorter arrays with NaN to handle different lengths
        padded_arrays = [
            np.pad(
                arr.astype(np.float64),
                (0, max_length - len(arr)),
                constant_values=np.nan,
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
