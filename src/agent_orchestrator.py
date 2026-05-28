from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import OUTPUT_DIR
from src.llm_client import LLMClient
from src.utils import ensure_output_dir, to_serializable
from tools.data_cleaner import prepare_features
from tools.data_loader import load_dataset
from tools.data_profiler import profile_data
from tools.model_evaluator import evaluate_models
from tools.model_trainer import train_baseline_models
from tools.report_exporter import export_markdown_report, export_word_report
from tools.task_detector import detect_task_type
from tools.visualizer import generate_visualizations


LogCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class AgentStep:
    number: int
    name: str
    handler_name: str


class DataAnalysisAgent:
    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = ensure_output_dir(output_dir)
        self.llm_client = LLMClient()
        self.steps = [
            AgentStep(1, "Load Dataset", "_step_load_dataset"),
            AgentStep(2, "Profile Data", "_step_profile_data"),
            AgentStep(3, "Detect Task Type", "_step_detect_task_type"),
            AgentStep(4, "Clean and Encode Data", "_step_clean_and_encode_data"),
            AgentStep(5, "Generate Visualizations", "_step_generate_visualizations"),
            AgentStep(6, "Train Baseline Models", "_step_train_baseline_models"),
            AgentStep(7, "Evaluate Models", "_step_evaluate_models"),
            AgentStep(8, "Generate AI Summary", "_step_generate_ai_summary"),
            AgentStep(9, "Export Report", "_step_export_report"),
        ]

    def run(
        self,
        dataset_path: str | Path,
        target_column: str,
        log_callback: LogCallback = None,
    ) -> dict:
        state: dict = {
            "dataset_path": Path(dataset_path),
            "target_column": target_column,
            "output_dir": self.output_dir,
            "logs": [],
        }

        for step in self.steps:
            self._log(state, f"Step {step.number}: {step.name} - started", log_callback)
            handler = getattr(self, step.handler_name)
            handler(state)
            self._log(state, f"Step {step.number}: {step.name} - completed", log_callback)

        return self._build_result(state)

    def _log(self, state: dict, message: str, callback: LogCallback = None) -> None:
        state["logs"].append(message)
        if callback:
            callback(message)

    def _step_load_dataset(self, state: dict) -> None:
        load_result = load_dataset(state["dataset_path"])
        state["load_result"] = load_result
        state["dataframe"] = load_result.dataframe

    def _step_profile_data(self, state: dict) -> None:
        state["profile"] = profile_data(state["dataframe"])

    def _step_detect_task_type(self, state: dict) -> None:
        state["task_detection"] = detect_task_type(state["dataframe"], state["target_column"])

    def _step_clean_and_encode_data(self, state: dict) -> None:
        state["cleaned_data"] = prepare_features(
            state["dataframe"],
            target_column=state["target_column"],
            task_type=state["task_detection"]["task_type"],
        )

    def _step_generate_visualizations(self, state: dict) -> None:
        state["visualizations"] = generate_visualizations(
            state["dataframe"],
            target_column=state["target_column"],
            task_type=state["task_detection"]["task_type"],
            output_dir=state["output_dir"],
        )
        state["correlations"] = self._find_target_correlations(
            state["dataframe"],
            target_column=state["target_column"],
            task_type=state["task_detection"]["task_type"],
        )

    def _step_train_baseline_models(self, state: dict) -> None:
        state["training_result"] = train_baseline_models(
            state["cleaned_data"],
            task_type=state["task_detection"]["task_type"],
        )

    def _step_evaluate_models(self, state: dict) -> None:
        state["evaluation"] = evaluate_models(
            state["training_result"],
            task_type=state["task_detection"]["task_type"],
        )

    def _step_generate_ai_summary(self, state: dict) -> None:
        context = self._build_summary_context(state)
        state["summary_context"] = context
        state["ai_summary"] = self.llm_client.generate_summary(context)

    def _step_export_report(self, state: dict) -> None:
        context = self._build_report_context(state)
        markdown_path = export_markdown_report(context, state["output_dir"])
        docx_path = export_word_report(context, state["output_dir"])
        state["report_paths"] = {
            "markdown": markdown_path,
            "docx": docx_path,
        }

    def _build_result(self, state: dict) -> dict:
        return {
            "logs": state["logs"],
            "profile": state["profile"],
            "task_detection": state["task_detection"],
            "visualizations": state["visualizations"],
            "evaluation": state["evaluation"],
            "ai_summary": state["ai_summary"],
            "report_paths": state["report_paths"],
        }

    def _build_summary_context(self, state: dict) -> dict:
        dataframe = state["dataframe"]
        target_column = state["target_column"]
        return to_serializable(
            {
                "dataset": {
                    "source": str(state["dataset_path"]),
                    "rows": dataframe.shape[0],
                    "columns": dataframe.shape[1],
                    "column_names": list(dataframe.columns),
                },
                "profile": state["profile"],
                "task_detection": state["task_detection"],
                "target": {
                    "name": target_column,
                    "distribution": self._target_distribution(dataframe, target_column),
                },
                "correlations": state.get("correlations", []),
                "evaluation": state["evaluation"],
            }
        )

    def _build_report_context(self, state: dict) -> dict:
        context = self._build_summary_context(state)
        context["ai_summary"] = state["ai_summary"]
        context["logs"] = state["logs"]
        context["visualizations"] = state["visualizations"]
        return context

    def _target_distribution(self, dataframe: pd.DataFrame, target_column: str) -> dict:
        target = dataframe[target_column]
        if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
            return {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "median": float(target.median()),
            }
        return target.astype("object").fillna("<missing>").value_counts().head(20).to_dict()

    def _find_target_correlations(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> list[dict]:
        if task_type != "regression":
            return []
        numeric = dataframe.select_dtypes(include="number")
        if target_column not in numeric.columns or len(numeric.columns) < 2:
            return []
        corr = numeric.corr(numeric_only=True)[target_column].drop(labels=[target_column], errors="ignore")
        corr = corr.dropna().sort_values(key=lambda values: values.abs(), ascending=False)
        return [
            {"feature": feature, "correlation": float(value)}
            for feature, value in corr.head(10).items()
        ]
