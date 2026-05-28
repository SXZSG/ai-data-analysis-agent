import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._")


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.head(20).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(20).to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def format_metric(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""

    columns = [str(column) for column in frame.columns]
    rows = frame.astype(object).where(pd.notna(frame), "").values.tolist()
    lines = [
        "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |")

    return "\n".join(lines)


def dict_to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    return dataframe_to_markdown_table(pd.DataFrame(rows))


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
