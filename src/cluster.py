# Functions to apply clustering algs
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


def apply_kmeans(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels, model

def apply_dbscan(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels, model

def apply_agglomerative(data, n_clusters=3, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(data)
    return labels, model

def apply_gmm(data, n_components=3):
    model = GaussianMixture(n_components=n_components, random_state=42)
    labels = model.fit_predict(data)
    return labels, model

def evaluate_clustering(data, labels):
    if len(set(labels)) > 1 and -1 not in set(labels):  # Avoid invalid silhouette cases
        score = silhouette_score(data, labels)
        return score
    return None

def compare_clustering_algorithms(data, algorithms, plot=True):
    '''
    Compare clustering algorithms on the same dataset.
    
    Parameters:
    - data: ndarray, preprocessed input data
    - algorithms: dict, keys are names, values are functions returning (labels, model)
    - plot: bool, whether to show 2D scatter plots
    
    Returns:
    - results: dict of {algorithm_name: silhouette_score}
    '''
    results = {}
    for name, cluster_func in algorithms.items():
        try:
            labels, model = cluster_func(data)
            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(data, labels)
            else:
                score = None
            results[name] = score

            if plot:
                plt.figure(figsize=(5, 4))
                plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=40)
                plt.title(f'{name} (Silhouette: {score:.2f})' if score else f'{name} (Invalid Score)')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = None
    return results

# Optimization ----------------------------------------------------------------------------------

def genetic_clustering_optimizer(df, method='kmeans', generations=10, population_size=6,
                                  cluster_range=(2, 10), dbscan_grid=None,
                                  pca_range=(None, 2, 3), visualize=False):
    '''
    Genetic algorithm to optimize clustering with multiple parameters.

    Inputs:
    - df: df with scaled features; first column must be ID_Client
    - method: clustering method ('kmeans', 'gmm', 'agglomerative', 'dbscan')
    - generations: number of iterations of the gen alg loop
    - population_size: number of individuals per generation
    - cluster_range: tuple (min_k, max_k) for k-based methods
    - dbscan_grid: list of (eps, min_samples) tuples for DBSCAN
    - pca_range: list of PCA component options (None = no PCA)
    - visualize: whether to show final cluster plot

    Output:
    - best_result: dict with metadata and result_df (ID_Client + label)
    '''
    # Separate out the id column
    feature_data = df.iloc[:, 1:].copy()
    id_col = df.iloc[:, 0].reset_index(drop=True)

    # Initialize population
    if method == 'dbscan':
        if dbscan_grid is None:
            dbscan_grid = [(round(random.uniform(0.2, 1.0), 2), random.randint(3, 15)) for _ in range(population_size)]
        population = dbscan_grid
    else:
        population = [{
            'n_clusters': random.randint(*cluster_range),
            'init': random.choice(['k-means++', 'random']),
            'pca': random.choice(pca_range)
        } for i in range(population_size)]

    best_result = {'score': -1}

    # Main gen alg loop
    for gen in range(generations):
        scores, models, labels_list = [], [], []

        for individual in population:
            try:
                # Apply PCA if indicated
                if method != 'dbscan':
                    pca_components = individual['pca']
                else:
                    pca_components = random.choice(pca_range)

                if pca_components is not None:
                    pca = PCA(n_components=pca_components, random_state=42)
                    data = pca.fit_transform(feature_data.values)
                else:
                    pca = None
                    data = feature_data.values

                # Apply clustering
                if method == 'kmeans':
                    labels, model = apply_kmeans(data, n_clusters=individual['n_clusters'], init=individual['init'])
                elif method == 'gmm':
                    labels, model = apply_gmm(data, n_components=individual['n_clusters'])
                elif method == 'agglomerative':
                    labels, model = apply_agglomerative(data, n_clusters=individual['n_clusters'])
                elif method == 'dbscan':
                    eps, min_samples = individual
                    labels, model = apply_dbscan(data, eps=eps, min_samples=min_samples)
                else:
                    continue

                # Evaluate
                if len(set(labels)) > 1 and -1 not in set(labels):
                    score = silhouette_score(data, labels)
                    scores.append(score)
                    models.append(model)
                    labels_list.append(labels)

                    if score > best_result['score']:
                        best_result = {
                            'method': method,
                            'params': individual,
                            'score': score,
                            'labels': labels,
                            'model': model,
                            'pca': pca
                        }
                else:
                    scores.append(-1)
                    models.append(None)
                    labels_list.append(None)
            except:
                scores.append(-1)
                models.append(None)
                labels_list.append(None)

        # Select top 2 parents
        top_indices = np.argsort(scores)[-2:]
        parent1 = population[top_indices[0]]
        parent2 = population[top_indices[1]]

        # Crossover and mutation
        if method == 'dbscan':
            child1 = (parent2[0], parent1[1])
            mutated_eps = round(min(1.5, max(0.1, parent1[0] + random.uniform(-0.1, 0.1))), 2)
            child2 = (mutated_eps, parent1[1])
        else:
            # Crossover: blend n_clusters, swap init, average pca
            child1 = {
                'n_clusters': int((parent1['n_clusters'] + parent2['n_clusters']) / 2),
                'init': random.choice([parent1['init'], parent2['init']]),
                'pca': random.choice([parent1['pca'], parent2['pca']])
            }
            # Mutation: tweak n_clusters ±1, random init, random pca
            mutation_k = parent1['n_clusters'] + random.choice([-1, 1])
            mutation_k = max(cluster_range[0], min(cluster_range[1], mutation_k))
            child2 = {
                'n_clusters': mutation_k,
                'init': random.choice(['k-means++', 'random']),
                'pca': random.choice(pca_range)
            }

        # New population
        population = [parent1, parent2, child1, child2] + [
            random.choice(dbscan_grid) if method == 'dbscan' else {
                'n_clusters': random.randint(*cluster_range),
                'init': random.choice(['k-means++', 'random']),
                'pca': random.choice(pca_range)
            }
            for _ in range(population_size - 4)
        ]

    # Final result df
    result_df = pd.DataFrame({
        'ID_Client': id_col,
        'label': best_result['labels']
    })
    best_result['result_df'] = result_df

    # Optional visualization with scatterplots
    if visualize and best_result['labels'] is not None:
        plot_data = best_result['pca'].transform(feature_data.values) if best_result['pca'] else feature_data.values
        plt.figure(figsize=(6, 5))
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=best_result['labels'], cmap='tab10', s=40)
        plt.title(f"Genetic Optimized: {best_result['method']} ({best_result['params']})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_result

def describe_clusters(data_clean, cluster_labels, show_boxplots=False):
    '''
    Summarizes each cluster by the mean of original variables

    Input:
    - data_clean: dataframe of original (unscaled) features
    - cluster_labels: cluster assignments (output of genetic_clustering_optimizer)
    - show_boxplots: whether to display boxplots per variable per cluster
    
    Output:
    - summary_df: dataframe with mean values per cluster
    '''
    df = data_clean.copy()
    df['Cluster'] = cluster_labels
    summary_df = df.groupby('Cluster').mean().round(2)

    if show_boxplots:
            num_vars = df.shape[1] - 1  # exclude 'Cluster'
            fig, axes = plt.subplots(nrows=(num_vars + 1) // 2, ncols=2, figsize=(12, 4 * ((num_vars + 1) // 2)))
            axes = axes.flatten()

            for i, col in enumerate(df.columns[:-1]):  # skip 'Cluster'
                sns.boxplot(x='Cluster', y=col, data=df, ax=axes[i], palette='Set2')
                axes[i].set_title(f'{col} by Cluster')
                axes[i].grid(True)

            plt.tight_layout()
            plt.show()

    return summary_df