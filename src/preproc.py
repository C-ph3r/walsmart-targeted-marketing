import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def walsmart_preproc(raw_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the WalSmart raw data according to the defined steps
    
    Input: raw_data - dataframe of raw WalSmart data
    Output: preproc_data - dataframe of preprocessed WalSmart data
    '''
    preproc_data = raw_data.copy()

    # 1. Remove rows with missing and duplicate ID_Client
    preproc_data.dropna(subset=['ID_Client'], inplace=True)
    preproc_data.drop_duplicates(subset=['ID_Client'], inplace=True)

    # 2. Remove meaningless and unusable columns
    preproc_data.drop(columns=['Potencial_Score', 'Latitude', 'Longitude', 
                            'Credit_factor', 'Store_state', 'ID_Store_last',
                            'Gender', 'Checked_ok'], 
                        axis=1, inplace=True)

    # 3. Treat simple negative values (replace with 0)
    preproc_data['Longevity_months'] = preproc_data['Longevity_months'].apply(lambda x: max(x, 0))
    preproc_data['Recency_in_weeks'] = preproc_data['Recency_in_weeks'].apply(lambda x: max(x, 0))

    # 4. Process ZIP_Code into Missing, Top 3 encoded, Others
    zip_s = preproc_data['ZIP_Code'].astype(str).str.strip()
    is_missing = preproc_data['ZIP_Code'].isna() | (zip_s == '')
    top3 = ['8', '0', '4']

    preproc_data['ZIP_Missing'] = is_missing.astype(int)
    for z in top3:
        preproc_data[f'ZIP_{z}'] = ((~is_missing) & (zip_s == z)).astype(int)
    preproc_data['ZIP_Others'] = ((~is_missing) & (~zip_s.isin(top3))).astype(int)
    preproc_data.drop(columns=['ZIP_Code'], axis=1, inplace=True)

    # 5. Process Relevance_criteria into Relevance_Priority
    preproc_data['Relevance_Priority'] = (preproc_data['Relevance_criteria'] == 'Priority').astype(int)
    preproc_data.drop(columns=['Relevance_criteria'], axis=1, inplace=True)

    # 6. Process Returns into Has_Returns
    preproc_data['Has_Returns'] = (preproc_data['Returns'] > 0).astype(int)
    preproc_data.drop(columns=['Returns'], axis=1, inplace=True)

    # 7. Process Flaged (convert to boolean: replace 2 with 1)
    preproc_data['Flaged'] = preproc_data['Flaged'].replace(2, 1).astype(int)

    # 8. Process Promotional_percentage (trim to [0, 100] by clipping)
    preproc_data['Promotional_percentage'] = preproc_data['Promotional_percentage'].clip(lower=0, upper=100)

    # 9. Process Education into Education_Years
    preproc_data['Education'] = preproc_data['Education'].fillna('').astype(str).str.strip()

    education_mapping = {
        '': 12,
        'High School': 12,
        'Degree': 15,
        'Bachelor Degree': 15,
        'MSc Degree': 17
    }
    preproc_data['Education_Years'] = preproc_data['Education'].map(education_mapping).fillna(12).astype(int)
    preproc_data.drop(columns=['Education'], axis=1, inplace=True)

    # 10. Correct the datatype for Dairy
    preproc_data["Dairy"] = pd.to_numeric(
        preproc_data["Dairy"].astype(str).str.replace(",000", "", regex=False).str.strip())

    # 11. Impute missing values for Frozen_Foods using KNN based on the product columns
    product_cols = ['Beer', 'Bottled_Water', 'Bread', 'Meat', 'Dairy', 'Fresh_Foods',
                'Fruit_Beverages', 'Pastry', 'Sodas', 'Toiletries', 'Veggies', 'Wines', 'Frozen_Foods']

    imputer = KNNImputer(n_neighbors=5)
    preproc_data[product_cols] = imputer.fit_transform(preproc_data[product_cols])

    # 12. Add the variable Total_Profit
    preproc_data['Total_Profit'] = preproc_data[product_cols].sum(axis=1)
    
    return preproc_data



def walsmart_scaling(preproc_data:pd.DataFrame) -> pd.DataFrame:
    '''
    Applies scaling and outlier treatment to the preprocessed WalSmart data

    Input: preproc_data - output of walsmart_preproc
    Output: scaled_data - dataframe with treated columns and normalized for clustering
    '''
    scaled_data = preproc_data.copy()
    
    # Define product columns
    product_cols = ['Beer', 'Bottled_Water', 'Bread', 'Meat', 'Dairy', 'Fresh_Foods',
                    'Fruit_Beverages', 'Pastry', 'Sodas', 'Toiletries', 'Veggies', 
                    'Wines', 'Frozen_Foods', 'Total_Profit']

    # 1. Product columns: Replace negative values with -1
    # Keeps the "loss" indicator without affecting the distribution too much
    for col in product_cols:
        scaled_data[col] = scaled_data[col].where(scaled_data[col] >= 0, -1)

    # 2. Product columns outliers: Windsorization at 99th percentile
    for col in product_cols:
        p99 = scaled_data[col].quantile(0.99)
        scaled_data[col] = scaled_data[col].clip(upper=p99)

    # 3. Scale all numeric columns for distance-based clustering
    scaler = StandardScaler()
    numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns.tolist()
    scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])

    return scaled_data


def pca_dbscan_noise_removal(
        scaled_data, preproc_data,
        n_components=20, 
        min_samples=20,
        k_for_eps=10
    ):
    '''
    Removes noise from the dataset using PCA for dimensionality reduction
    followed by DBSCAN for noise detection
    
    Inputs:
    - scaled_data: scaled dataset
    - preproc_data: preprocessed dataset (before scaling)
    - n_components: int/float, number of PCA components or variance ratio to keep
    - eps: float, DBSCAN eps parameter
    - min_samples: int, DBSCAN min_samples parameter

    Outputs:
    - scaled_data_clean: cleaned scaled dataset (noise removed)
    - preproc_data_clean: cleaned preprocessed dataset (noise removed)

    '''

    # Remove ID column
    X = scaled_data.drop(columns=['ID_Client'], errors='ignore')

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(f"Original dims: {X.shape[1]}")
    print(f"PCA dims: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Estimate eps using k-distance
    nbrs = NearestNeighbors(n_neighbors=k_for_eps).fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    distances = np.sort(distances[:, -1])

    # Knee point
    eps = np.percentile(distances, 95)
    print(f"Estimated eps (95th percentile): {eps:.4f}")

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)

    noise_mask = labels == -1
    n_noise = noise_mask.sum()
    n_total = len(labels)

    print(f"Noise detected: {n_noise} ({n_noise/n_total:.2%})")

    # Clean data
    scaled_clean = scaled_data[~noise_mask].reset_index(drop=True)
    preproc_clean = preproc_data[~noise_mask].reset_index(drop=True)

    return scaled_clean, preproc_clean