from typing import Any

import numpy as np
import pandas as pd


def analyze_data_quality(dataframe: pd.DataFrame, high_missing_threshold: float = 0.4) -> dict[str, Any]:
    """Return transparent quality diagnostics; never mutate or silently drop source rows."""
    row_count, column_count = dataframe.shape
    missing = dataframe.isna().sum()
    duplicate_rows = int(dataframe.duplicated().sum())
    constant_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]
    high_missing_columns = [
        {
            "column": str(column),
            "missing_ratio": round(float(missing[column] / row_count), 4) if row_count else 0.0,
        }
        for column in dataframe.columns
        if row_count and float(missing[column] / row_count) >= high_missing_threshold
    ]

    outliers = []
    for column in dataframe.select_dtypes(include="number").columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            outliers.append(
                {
                    "column": str(column),
                    "count": count,
                    "ratio": round(float(count / len(values)), 4),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }
            )

    return {
        "rows": int(row_count),
        "columns": int(column_count),
        "missing_cells": int(missing.sum()),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
        "numeric_iqr_outlier_warnings": outliers,
        "policy": "Diagnostics only: no source rows or outliers are automatically deleted.",
    }
