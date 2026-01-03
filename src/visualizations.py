import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np


def plot_heatmaps(df, columns, cluster_column, title="Cluster Heatmaps"):
    '''
    Plot heatmap and cluster size distribution for specific clusters
    
    Inputs:
      - df: dataframe with cluster labels and feature columns
      - columns: list of column names to include in heatmap
      - cluster_column: name of the cluster labels column
      - title: title for the plot
    '''
    # Compute mean profile for each cluster
    cluster_means = df.groupby(cluster_column)[columns].mean()
    cluster_counts = df[cluster_column].value_counts().sort_index()
    
    # Dynamically size figure based on number of columns
    n_cols = len(columns)
    figsize_width = max(16, n_cols * 0.8)  # Scale width with number of columns
    figsize_height = max(6, len(cluster_means) * 0.5)  # Scale height with number of clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_width, figsize_height))
    
    # Heatmap of cluster means
    sns.heatmap(
        cluster_means, 
        fmt='.2f', 
        cmap='RdYlBu', 
        ax=ax1
    )
    ax1.set_title(f'{title} - Cluster Means', fontsize=12)
    ax1.set_ylabel('Cluster')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Bar plot of cluster sizes
    cluster_counts.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title(f'{title} - Cluster Sizes', fontsize=12)
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Cluster')
    
    plt.tight_layout()
    plt.show()


def plot_tsne(df, feats, label,
              cmap='tab10',
              title="t-SNE Visualization of Clustering Solution"):

  two_dim = TSNE(random_state=42).fit_transform(df[feats])
  two_dim_df = pd.DataFrame(two_dim, index=df.index)
  two_dim_df[label] = df[label]


  fig, ax= plt.subplots(figsize=(10,10))
  scatter = ax.scatter(x = two_dim_df[0],
                      y=two_dim_df[1],
                      c=two_dim_df[label],
                      s=5,
                      cmap=cmap
                      )
  ax.set_xlabel("")
  ax.set_ylabel("")
  ax.set_xticks([])
  ax.set_yticks([])

  legend1 = ax.legend(*scatter.legend_elements(),
                      loc="best", title="Cluster Labels")
  ax.add_artist(legend1)

  plt.title(title)
  plt.show()

def plot_radar(df, columns, cluster_column, title="Radar Charts"):
    '''
    Plot radar charts for specific clusters
    
    Inputs:
      - df: source dataframe with cluster labels and feature columns
      - columns: list of column names to include in radar chart
      - cluster_column: name of the cluster labels column
      - title: title for the plot
    '''
    
    # Compute mean profile for each cluster
    cluster_means = df.groupby(cluster_column)[columns].mean()
    
    labels = columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    
    n_clusters = len(cluster_means)
    cols = 2
    rows = int(np.ceil(n_clusters / cols))
    fig = plt.figure(figsize=(cols * 6, rows * 5))
    
    for idx, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        ax = plt.subplot(rows, cols, idx + 1, projection='polar')
        values = row.values
        values = np.concatenate([values, values[:1]])
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=7)
        ax.set_title(f'Cluster {cluster_id}', fontsize=12, pad=12)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()