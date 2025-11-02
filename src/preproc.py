import pandas as pd
import numpy as np

def negative_values_report(df, cols=None, only_numeric=True):
    """
    Return a DataFrame with count and percentage of negative values per column.
    - cols: list of columns to check (default: all numeric columns if only_numeric=True,
            otherwise all columns).
    - only_numeric: if True (default) only numeric dtypes are checked when cols is None.
    """
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist() if only_numeric else df.columns.tolist()

    neg_counts = (df[cols] < 0).sum()
    neg_pct = neg_counts / len(df) * 100
    report = pd.concat([neg_counts, neg_pct], axis=1, keys=['neg_count', 'neg_pct'])
    report = report.loc[report['neg_count'] > 0].sort_values('neg_pct', ascending=False)
    return report