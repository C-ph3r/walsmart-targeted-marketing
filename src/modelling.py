import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------------------
# Perspective: Cluster merging via Hierarchical Clustering
# --------------------------------------------------------------------------------

# Create merged cluster using Hierarchical Clustering on centroids
def hc_merge_clusters(df, label1, label2, feats, merged_label, n_clusters=7):
    '''
    Merge two clustering perspectives using Hierarchical Clustering.
    
    Parameters:
    - df: DataFrame with label1 and label2 columns
    - label1: Name of first clustering label column
    - label2: Name of second clustering label column
    - feats: List of features to compute centroids
    - merged_label: Name of new merged label column
    - n_clusters: Number of clusters for HC
    '''
    df_ = df.copy()
    
    # Compute centroids for each combination of label1 and label2
    df_centroids = df_.groupby([label1, label2])[feats].mean()
    
    # Apply Hierarchical Clustering to the centroids
    hclust = AgglomerativeClustering(
        linkage='ward',
        metric='euclidean',
        n_clusters=n_clusters
    )
    hclust_labels = hclust.fit_predict(df_centroids)
    df_centroids[merged_label] = hclust_labels
    
    # Create mapping from (label1, label2) pairs to merged cluster labels
    cluster_mapper = df_centroids[merged_label].to_dict()
    
    # Apply mapping to original dataframe
    df_[merged_label] = df_.apply(
        lambda row: cluster_mapper[(row[label1], row[label2])], axis=1
    )
    
    return df_, df_centroids


# --------------------------------------------------------------------------------
# Modular functions for optimized HDBSCAN clustering pipeline
# --------------------------------------------------------------------------------
# 0. Separate ID column and numeric data
def prepare_data(scaled_data, id_column="ID_Client"):
    ids = scaled_data[id_column].values
    X = scaled_data.drop(columns=[id_column]).values
    return ids, X

# 1. PCA dimensionality reduction
def apply_pca(X, n_components=0.95):
    '''
    PCA with variance retention (e.g., 95%) or fixed number of components.
    '''
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced dimensionality from {X.shape[1]} → {X_pca.shape[1]}")
    return X_pca, pca

# 2. t-SNE visualization
def plot_tsne(X_pca, labels=None, perplexity=30, n_components=2, figsize=(10, 8)):
    '''
    Apply t-SNE dimensionality reduction and produce a scatter plot for cluster visualization.
    
    Inputs:
      - X_pca: Input data (PCA-reduced or original)
      - labels: Optional cluster labels for coloring points
      - perplexity: t-SNE perplexity parameter (default 30)
      - n_components: Target dimensionality (default 2)
      - figsize: Figure size for plot
    '''
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_pca)
    
    # Always create visualization for 2D
    if n_components == 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # Filter out noise points (-1) for better visualization
            mask = labels != -1
            scatter = plt.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1], 
                c=labels[mask], cmap='tab10', s=20, alpha=0.7, label='Clusters'
            )
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=20, alpha=0.7)
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Cluster Visualization')
        plt.tight_layout()
        plt.show()
    
    return X_tsne


# 3. HDBSCAN clustering
def apply_hdbscan(X_reduced, min_cluster_size=30, min_samples=None):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_reduced)
    probs = clusterer.probabilities_
    print(f"HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    return labels, probs, clusterer


# --------------------------------------------------------------------------------
# Optimizable PCA → Silhouette → HDBSCAN pipeline
# --------------------------------------------------------------------------------

def silhouette_on_pca(X_pca, labels, max_clusters=20, penalty_weight=0.02):
    '''
    Compute silhouette in PCA space, with penalties for:
      - too many clusters
      - too few clusters
    '''
    mask = labels != -1
    unique = np.unique(labels[mask])

    # Not enough clusters to compute silhouette
    if len(unique) < 2:
        return -1.0

    try:
        sil = silhouette_score(X_pca[mask], labels[mask])
    except Exception:
        return -1.0

    # Penalize excessive number of clusters
    k = len(unique)
    if k > max_clusters:
        sil -= penalty_weight * (k - max_clusters)

    return sil


def optimize_pca_hdbscan(
    scaled_data,
    id_column="ID_Client",
    pca_n_components_list=[0.90, 0.95],
    hdbscan_min_cluster_size_list=[500, 1000, 2000],
    hdbscan_min_samples_list=[50, 100],
    sample_size=5000,
    random_state=42
):
    '''
    Optimization over PCA and HDBSCAN parameters using silhouette on non-noise points.
    Evaluated on a sample for speed, then re-run on full data with best params.

    Parameters:
      - scaled_data: Preprocessed scaled data
      - id_column: Name of ID column
      - pca_n_components_list: List of PCA variance retention values (e.g., [0.90, 0.95])
      - hdbscan_min_cluster_size_list: List of HDBSCAN min_cluster_size values
      - hdbscan_min_samples_list: List of HDBSCAN min_samples values
      - sample_size: Size for sampling during optimization
      - random_state: Random seed
    '''

    ids, X_full = prepare_data(scaled_data, id_column=id_column)

    # Sample for speed
    rng = np.random.default_rng(random_state)
    if len(X_full) > sample_size:
        sample_idx = rng.choice(len(X_full), size=sample_size, replace=False)
        X_sample = X_full[sample_idx]
    else:
        sample_idx = np.arange(len(X_full))
        X_sample = X_full

    best_score = -1.0
    best_params = None

    # Grid search over parameter combinations
    for pca_comp in pca_n_components_list:
        for min_clust_size in hdbscan_min_cluster_size_list:
            for min_samp in hdbscan_min_samples_list:
                # PCA
                X_pca, pca_model = apply_pca(X_sample, n_components=pca_comp)

                # HDBSCAN
                labels, probs, hdbscan_model = apply_hdbscan(
                    X_pca,
                    min_cluster_size=min_clust_size,
                    min_samples=min_samp,
                )

                score = silhouette_on_pca(X_pca, labels)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "pca_n_components": pca_comp,
                        "hdbscan_min_cluster_size": min_clust_size,
                        "hdbscan_min_samples": min_samp,
                    }

    # Re-run best params on full data
    X_pca_full, pca_model_full = apply_pca(X_full, n_components=best_params["pca_n_components"])
    labels_full, probs_full, hdbscan_model_full = apply_hdbscan(
        X_pca_full,
        min_cluster_size=best_params["hdbscan_min_cluster_size"],
        min_samples=best_params["hdbscan_min_samples"],
    )

    # Compute t-SNE for visualization
    X_tsne_full = plot_tsne(
        X_pca_full,
        labels=labels_full,
        perplexity=30,
        n_components=2,
    )

    result_df = pd.DataFrame({
        id_column: ids,
        "Cluster_HDBSCAN": labels_full,
        "Cluster_Probability": probs_full,
    })

    return {
        "result_df": result_df,
        "best_params": best_params,
        "best_score": best_score,
        "X_pca": X_pca_full,
        "X_tsne": X_tsne_full,
        "pca_model": pca_model_full,
        "hdbscan_model": hdbscan_model_full,
    }

