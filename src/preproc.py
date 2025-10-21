import pandas as pd
import numpy as np

def percent_rows_not_meeting_conditions(df, conditions):
    """
    Takes a DataFrame and a list of conditions (as strings, e.g. "col >= 0"),
    checks each row for each condition, and outputs the percentage of rows that do NOT meet all conditions.

    Args:
        df: pandas DataFrame
        conditions: list of strings, each a valid pandas query condition (e.g. ["A >= 0", "B <= 100"])

    Returns:
        float: percentage of rows not meeting all conditions
    """
    # Combine all conditions with '&' for pandas query
    combined = " & ".join(conditions)
    # Rows that meet all conditions
    meets = df.query(combined)
    pct_not_meeting = 100 * (1 - len(meets) / len(df))
    print(f"Percentage of rows NOT meeting all conditions: {pct_not_meeting:.2f}%")
    return pct_not_meeting