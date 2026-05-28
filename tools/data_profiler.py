from typing import Any

import pandas as pd


def profile_data(dataframe: pd.DataFrame, top_n: int = 5) -> dict:
    row_count = len(dataframe)
    column_profiles: list[dict[str, Any]] = []
    numeric_summary: dict[str, dict] = {}
    categorical_summary: dict[str, dict] = {}

    for column in dataframe.columns:
        series = dataframe[column]
        missing_count = int(series.isna().sum())
        missing_ratio = float(missing_count / row_count) if row_count else 0.0
        unique_count = int(series.nunique(dropna=True))
        profile = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 4),
            "unique_count": unique_count,
        }
        column_profiles.append(profile)

        if pd.api.types.is_numeric_dtype(series):
            numeric_summary[column] = {
                "mean": _safe_float(series.mean()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "median": _safe_float(series.median()),
            }
        else:
            top_values = (
                series.astype("object")
                .fillna("<missing>")
                .value_counts(dropna=False)
                .head(top_n)
                .to_dict()
            )
            categorical_summary[column] = {str(key): int(value) for key, value in top_values.items()}

    return {
        "row_count": row_count,
        "column_count": dataframe.shape[1],
        "column_profiles": column_profiles,
        "numeric_columns": list(dataframe.select_dtypes(include="number").columns),
        "categorical_columns": list(dataframe.select_dtypes(exclude="number").columns),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
