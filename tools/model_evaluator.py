import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from tools.model_trainer import TrainingResult


def evaluate_models(training_result: TrainingResult, task_type: str) -> dict:
    results = []

    for model_name, model in training_result.models.items():
        predictions = model.predict(training_result.X_test)
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(training_result.y_test, predictions)),
                "precision": float(
                    precision_score(training_result.y_test, predictions, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(training_result.y_test, predictions, average="weighted", zero_division=0)
                ),
                "f1": float(f1_score(training_result.y_test, predictions, average="weighted", zero_division=0)),
            }
            score = metrics["f1"]
        else:
            mae = mean_absolute_error(training_result.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(training_result.y_test, predictions))
            r2 = r2_score(training_result.y_test, predictions)
            metrics = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
            score = metrics["R2"]

        results.append(
            {
                "model_name": model_name,
                "metrics": metrics,
                "selection_score": float(score),
            }
        )

    best_model = max(results, key=lambda item: item["selection_score"])
    return {
        "results": results,
        "best_model": best_model,
        "split_info": training_result.split_info,
        "selection_metric": "f1" if task_type == "classification" else "R2",
    }
