import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

def negative_values_report(df, cols=None, only_numeric=True):
    '''
    Return a df with count and percentage of negative values per column.
    - cols: list of columns to check (default: all numeric columns if only_numeric=True,
            otherwise all columns).
    - only_numeric: if T, only numeric dtypes are checked when cols is None.
    '''
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist() if only_numeric else df.columns.tolist()

    neg_counts = (df[cols] < 0).sum()
    neg_pct = neg_counts / len(df) * 100
    report = pd.concat([neg_counts, neg_pct], axis=1, keys=['neg_count', 'neg_pct'])
    report = report.loc[report['neg_count'] > 0].sort_values('neg_pct', ascending=False)
    return report

def clean_outliers(df, threshold=3, count_limit=1000, 
                   numeric_only=True, imputer_max_iter=10):

    '''
    For each numeric column:
      - if outlier_count > count_limit: set those values to nan
      - else: mark those rows for removal
    After processing all columns:
      - drop marked rows
      - impute remaining NaNs in numeric columns using random forest

    Returns cleaned_df and a small report dict.
    '''

    dfc = df.copy()
    numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist() if numeric_only else dfc.columns.tolist()

    # set to track rows that should be dropped
    to_drop_idx = set()
    # dictionary to record columns where outliers were taken out
    cols_set_to_nan = {}

    # for each column to detect outliers from
    for col in numeric_cols:
        col_series = dfc[col]

        # skip column if it's empty'
        if col_series.dropna().empty:
            continue

        # compute mean and std deviation for outlier detection
        mean = col_series.mean()
        std = col_series.std()

        # skip column if std is undefined or zero (no variation)
        if pd.isna(std) or std == 0:
            continue

        # identify outliers using the threshold (by default 3 standard deviations from mean)
        mask = (col_series > mean + threshold * std) | (col_series < mean - threshold * std)
        out_idx = dfc.index[mask].tolist()
        cnt = len(out_idx)

        # skip column if no outliers are found
        if cnt == 0:
            continue

        if cnt > count_limit:
            # too many outliers: set them to nan for later imputation
            dfc.loc[out_idx, col] = pd.NA
            cols_set_to_nan[col] = cnt
        else:
            # few outliers: mark rows for removal
            to_drop_idx.update(out_idx)

    # drop small-number outlier rows
    if to_drop_idx:
        dfc = dfc.drop(index=list(to_drop_idx)).reset_index(drop=True)

    # impute numeric columns if there are NaNs
    numeric_cols_after = dfc.select_dtypes(include=[np.number]).columns.tolist()
    if dfc[numeric_cols_after].isna().any().any():
        estimator = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        imp = IterativeImputer(estimator=estimator, random_state=0, max_iter=imputer_max_iter, initial_strategy='median')
        imputed = imp.fit_transform(dfc[numeric_cols_after])
        dfc[numeric_cols_after] = imputed

    report = {
        'dropped_rows': len(to_drop_idx),
        'cols_set_to_nan': cols_set_to_nan,
        'final_shape': dfc.shape
    }
    return dfc, report

def show_outliers(data, threshold=2):
    '''
    Function to find outliers in each column of a df and print the number of outliers per column

    Input:
    - data: DataFrame containing the data.
    - threshold: Threshold for determining outliers, default is 2 standard deviations

    Output:
    - A dictionary containing the number of outliers for each column
    '''
    outlier_counts = {}

    for column in data.columns:
        datacopy = data.copy()
        mean = datacopy[column].mean()
        std = datacopy[column].std()
        outliers = datacopy[(datacopy[column] > mean + threshold * std) | (datacopy[column] < mean - threshold * std)]
        outlier_counts[column] = len(outliers)

    for column, count in outlier_counts.items():
        print(f"Column '{column}' has {count} outliers.")

    return outlier_counts