import pandas as pd


def detect_task_type(dataframe: pd.DataFrame, target_column: str) -> dict:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column `{target_column}` does not exist.")

    target = dataframe[target_column].dropna()
    if target.empty:
        raise ValueError("Target column contains only missing values.")

    unique_count = int(target.nunique())
    row_count = int(len(target))
    dtype = str(target.dtype)

    if pd.api.types.is_numeric_dtype(target):
        discrete_threshold = max(10, int(row_count * 0.05))
        if unique_count <= discrete_threshold:
            return {
                "task_type": "classification",
                "reason": (
                    f"Target is numeric but has only {unique_count} unique values, "
                    "so it is treated as a discrete label."
                ),
                "target_dtype": dtype,
                "unique_count": unique_count,
            }
        return {
            "task_type": "regression",
            "reason": "Target is numeric with enough unique values to be treated as continuous.",
            "target_dtype": dtype,
            "unique_count": unique_count,
        }

    return {
        "task_type": "classification",
        "reason": "Target is non-numeric and treated as categorical labels.",
        "target_dtype": dtype,
        "unique_count": unique_count,
    }
