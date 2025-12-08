# Functions for visualization
import os, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

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

def plot_categorical_value_counts(df, out_path="outputs/categorical_value_counts.png", cols=4, top_n=10, figsize_per_plot=(4, 3), save=False, show=True):
    """
    Plot bar charts of value counts for all non-numeric (categorical/object) columns in the input dataframe.
    Only the top_n most frequent values are shown for each column.
    Returns the path to the saved image if save=True, otherwise returns the matplotlib Figure.
    """
    import os
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        raise ValueError("No categorical columns found.")

    n = len(cat_cols)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
    if hasattr(axes, "flatten"):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    for i, col in enumerate(cat_cols):
        ax = axes_list[i]
        vc = df[col].value_counts(dropna=False).head(top_n)
        if vc.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(col)
            continue
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h", color="C1")
        ax.set_title(col)
        ax.set_xlabel("Count")
        ax.set_ylabel("")

    # hide unused axes
    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_world_map(df, lat_col='Latitude', lon_col='Longitude', zoom_start=2):
    '''
    Plots latitude and longitude points on a folium world map

    Input:
    - df: dataframe with latitude and longitude columns
    - lat_col: name of the latitude column
    - lon_col: name of the longitude column
    - zoom_start: initial zoom level (2=global view)

    Output:
    - folium map object
    '''
    # Center map on mean coordinates
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()

    world_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(world_map)

    return world_map

def plot_elbow_graph(dispersion:list):
    '''
    Function to plot elbow graphs given a dispersion list
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 20), dispersion, marker='o')
    plt.xticks(range(1, 21, 1))
    plt.xlabel('Number of clusters')
    plt.ylabel('Dispersion')
    plt.show()