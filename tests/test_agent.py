from pathlib import Path

import pandas as pd

from src.agent_orchestrator import DataAnalysisAgent
from tools.data_quality import analyze_data_quality


def test_quality_diagnostics_do_not_mutate_dataframe():
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0, 7.0],
            "mostly_missing": [None, None, None, 1.0, None, None, None, None, None],
            "constant": ["x"] * 9,
        }
    )
    original = frame.copy(deep=True)

    result = analyze_data_quality(frame)

    assert result["duplicate_rows"] == 1
    assert "constant" in result["constant_columns"]
    assert "mostly_missing" in [item["column"] for item in result["high_missing_columns"]]
    assert "value" in [item["column"] for item in result["numeric_iqr_outlier_warnings"]]
    pd.testing.assert_frame_equal(frame, original)


def test_agent_completes_classification_workflow_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    dataset = Path(__file__).resolve().parents[1] / "examples" / "sample_classification.csv"
    agent = DataAnalysisAgent(output_dir=tmp_path)

    result = agent.run(
        dataset_path=dataset,
        target_column="churn",
        user_goal="Check quality and compare baseline classification models.",
    )

    assert result["agent_mode"] == "deterministic_fallback"
    assert result["task_detection"]["task_type"] == "classification"
    assert len(result["evaluation"]["results"]) >= 2
    assert result["data_quality"]["missing_cells"] > 0
    assert Path(result["report_paths"]["markdown"]).exists()
    assert Path(result["report_paths"]["markdown"]).parent.parent == tmp_path
    assert len(result["tool_trace"]) >= 7
    assert all(item["status"] == "success" for item in result["tool_trace"])



def test_agent_completes_regression_workflow_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    dataset = Path(__file__).resolve().parents[1] / "examples" / "sample_regression.csv"
    agent = DataAnalysisAgent(output_dir=tmp_path)

    result = agent.run(
        dataset_path=dataset,
        target_column="price",
        user_goal="Inspect quality and compare baseline regression models.",
    )

    assert result["agent_mode"] == "deterministic_fallback"
    assert result["task_detection"]["task_type"] == "regression"
    assert len(result["evaluation"]["results"]) >= 2
    assert {"MAE", "RMSE", "R2"} <= set(result["evaluation"]["results"][0]["metrics"])
    assert Path(result["report_paths"]["markdown"]).exists()
