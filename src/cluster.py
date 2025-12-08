# Functions to apply clustering algs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import itertools


def apply_kmeans(data, n_clusters=3):
    '''
    Function to apply KMeans clustering
    Input:
    - data: dataframe of features
    - n_clusters: number of clusters to form

    Output:
    - labels: cluster labels for each data point
    - model: fitted KMeans model
    '''
    model = KMeans(n_clusters=n_clusters, random_state=1)
    labels = model.fit_predict(data)
    return labels, model

def apply_dbscan(data, eps=0.5, min_samples=5):
    '''
    Function to apply DBSCAN clustering
    Input:
    - data: dataframe of features
    - eps: distance threshold
    - min_samples: sample count threshold

    Output:
    - labels: cluster labels for each data point
    - model: fitted DBSCAN model
    '''
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels, model

def apply_agglomerative(data, n_clusters=3, linkage='ward'):
    '''
    Function to apply Agglomerative Clustering
    Input:
    - data: dataframe of features
    - n_clusters: number of clusters to form
    - linkage: linkage criterion

    Output:
    - labels: cluster labels for each data point
    - model: fitted AgglomerativeClustering model
    '''
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(data)
    return labels, model

def evaluate_clustering(data, labels):
    '''
    Function to evaluate clustering using silhouette score
    Input:
    - data: dataframe of features
    - labels: cluster labels for each data point
    Output:
    - score: silhouette score (None if invalid)
    '''
    if len(set(labels)) > 1 and -1 not in set(labels):  # Avoid invalid silhouette cases
        score = silhouette_score(data, labels)
        return score
    return None

def optimize_demographic(X, ids,
                            k_values=range(2,11),
                            linkage_options=('ward','average','complete','single'),
                            random_state=1,
                            verbose=False):
    '''
    Function to optimize parameters for the demographic clustering task

    Input:
    - X: dataframe of features
    - ids: list of client IDs

    Output:
    - result_df: dataframe with columns ['id', 'label'] for the best configuration
    - best_description: best model and hyperparameters
    - best_score: silhouette score
    - best_model: fitted sklearn-like model object
    '''
    X_arr = np.asarray(X)
    best_score = None
    best_labels = None
    best_model = None
    best_description = None

    # KMeans grid
    for k in k_values:
        labels, model = apply_kmeans(X_arr, n_clusters=int(k))
        score = evaluate_clustering(X_arr, labels)
        if score is None:
            if verbose:
                print(f"KMeans k={k} -> invalid (score None)")
            continue
        if (best_score is None) or (score > best_score):
            best_score = score
            best_labels = labels
            best_model = model
            best_description = f"KMeans n_clusters={k}"

    # Agglomerative grid
    for k, linkage in itertools.product(k_values, linkage_options):
        labels, model = apply_agglomerative(X_arr, n_clusters=int(k), linkage=linkage)
        score = evaluate_clustering(X_arr, labels)
        if score is None:
            if verbose:
                print(f"Agglomerative k={k} linkage={linkage} -> invalid (score None)")
            continue
        if (best_score is None) or (score > best_score):
            best_score = score
            best_labels = labels
            best_model = model
            best_description = f"Agglomerative n_clusters={k} linkage={linkage}"

    if best_labels is None:
        best_labels = np.full(X_arr.shape[0], -1, dtype=int)
        best_model = None
        best_description = None

    result_df = pd.DataFrame({'id': np.asarray(ids), 'label': np.asarray(best_labels)})
    return result_df, best_description, best_score, best_model

def optimize_purchase(X, ids,
                        eps_values=(0.1,0.2,0.3,0.5,0.8,1.0),
                        min_samples_values=(3,5,7,10),
                        metric='euclidean',
                        verbose=False):
    """
    Function to optimize parameters for the demographic clustering task

    Input:
    - X: dataframe of features
    - ids: list of client IDs

    Output:
    - result_df: dataframe with columns ['id', 'label'] for the best configuration
    - best_description: best DBSCAN config
    - best_score: silhouette score
    - best_model: fitted DBSCAN instance
    """
    X_arr = np.asarray(X)
    best_score = None
    best_labels = None
    best_model = None
    best_description = None

    for eps, min_s in itertools.product(eps_values, min_samples_values):
        labels, model = apply_dbscan(X_arr, eps=float(eps), min_samples=int(min_s))
        score = evaluate_clustering(X_arr, labels)
        if score is None:
            if verbose:
                print(f"DBSCAN eps={eps} min_samples={min_s} -> invalid (score None)")
            continue
        if (best_score is None) or (score > best_score):
            best_score = score
            best_labels = labels
            best_model = model
            best_description = f"DBSCAN eps={eps} min_samples={min_s} metric={metric}"

    if best_labels is None:
        best_labels = np.full(X_arr.shape[0], -1, dtype=int)
        best_model = None
        best_description = None

    result_df = pd.DataFrame({'id': np.asarray(ids), 'label': np.asarray(best_labels)})
    return result_df, best_description, best_score, best_model