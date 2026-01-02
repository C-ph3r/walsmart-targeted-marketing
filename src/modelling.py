import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import hdbscan






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
    """
    PCA with variance retention (e.g., 95%) or fixed number of components.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced dimensionality from {X.shape[1]} → {X_pca.shape[1]}")
    return X_pca, pca

# 2. UMAP nonlinear embedding
def apply_umap(X_pca, n_neighbors=30, min_dist=0.1, n_components=2):
    """
    UMAP projection to low-dimensional manifold.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"UMAP reduced dimensionality to {n_components} components")
    return X_umap, reducer


# 3. HDBSCAN clustering
def apply_hdbscan(X_umap, min_cluster_size=30, min_samples=None):
    """
    Density-based clustering on UMAP space.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_umap)
    probs = clusterer.probabilities_
    print(f"HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    return labels, probs, clusterer


# 4. Full pipeline: PCA → UMAP → HDBSCAN

def pca_umap_hdbscan_pipeline(
    scaled_data,
    id_column="ID_Client",
    pca_components=0.95,
    umap_neighbors = 100,
    umap_min_dist = 0.5,
    umap_components = 2,
    hdbscan_min_cluster_size = 300,
    hdbscan_min_samples = 50
):
    '''
    Full pipeline: PCA → UMAP → HDBSCAN
    Manually optimized by changing parameters
    '''
    # Prepare data
    ids, X = prepare_data(scaled_data, id_column)

    # PCA
    X_pca, pca_model = apply_pca(X, n_components=pca_components)

    # UMAP
    X_umap, umap_model = apply_umap(
        X_pca,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=umap_components
    )

    # HDBSCAN
    labels, probs, hdbscan_model = apply_hdbscan(
        X_umap,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples
    )

    # Build result dataframe
    result = pd.DataFrame({
        id_column: ids,
        "Cluster_HDBSCAN": labels,
        "Cluster_Probability": probs
    })

    return result, X_pca, X_umap, pca_model, umap_model, hdbscan_model