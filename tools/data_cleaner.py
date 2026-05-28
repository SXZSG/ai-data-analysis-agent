from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CleanedData:
    X: pd.DataFrame
    y: pd.Series
    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_rows: int


def prepare_features(dataframe: pd.DataFrame, target_column: str, task_type: str) -> CleanedData:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column `{target_column}` does not exist.")

    before_rows = len(dataframe)
    working = dataframe.dropna(subset=[target_column]).copy()
    if working.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    X = working.drop(columns=[target_column]).copy()
    if X.empty:
        raise ValueError("No feature columns are available after removing the target column.")

    y = working[target_column].copy()
    if task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()
        if y.empty:
            raise ValueError("Regression target could not be converted to numeric values.")

    numeric_features = list(X.select_dtypes(include="number").columns)
    categorical_features = [column for column in X.columns if column not in numeric_features]

    for column in numeric_features:
        if X[column].isna().all():
            X[column] = X[column].fillna(0)

    for column in categorical_features:
        if X[column].isna().all():
            X[column] = X[column].fillna("missing")

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return CleanedData(
        X=X,
        y=y,
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_rows=before_rows - len(X),
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
