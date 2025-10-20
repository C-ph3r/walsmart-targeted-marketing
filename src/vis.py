# Functions for visualization
import os, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric_distributions(df, out_path="outputs/numeric_distributions.png", cols=4, bins=30, kde=True, figsize_per_plot=(4, 3), save=False, show=True):
    """
    Plot histograms for all numeric columns in the input dataframe.
    Returns the path to the saved image if save=True, otherwise returns the matplotlib Figure.
    """

    sns.set_style("whitegrid")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found.")

    n = len(numeric_cols)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
    # flatten axes to a list for easy indexing
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    for i, col in enumerate(numeric_cols):
        ax = axes_list[i]
        series = df[col].dropna()
        if series.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(col)
            continue
        try:
            sns.histplot(series, kde=kde, ax=ax, bins=bins, color="C0")
        except Exception:
            ax.hist(series, bins=bins, density=True, color="C0")
        ax.set_title(col)
        ax.set_xlabel("")

    # hide any unused axes
    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return out_path if save else fig