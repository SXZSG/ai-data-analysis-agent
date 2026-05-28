from dataclasses import dataclass

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tools.data_cleaner import CleanedData


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    X_test: object
    y_test: object
    split_info: dict


def train_baseline_models(
    cleaned_data: CleanedData,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    X = cleaned_data.X
    y = cleaned_data.y

    if len(X) < 5:
        raise ValueError("At least 5 rows are required to train baseline models.")

    stratify = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Classification requires at least two target classes.")
        if class_counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    estimators = _get_estimators(task_type, random_state)
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(cleaned_data.preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

    return TrainingResult(
        models=fitted_models,
        X_test=X_test,
        y_test=y_test,
        split_info={
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": test_size,
            "numeric_features": cleaned_data.numeric_features,
            "categorical_features": cleaned_data.categorical_features,
            "dropped_rows": cleaned_data.dropped_rows,
        },
    )


def _get_estimators(task_type: str, random_state: int) -> dict:
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=120,
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
