from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_output_dir


def generate_visualizations(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    output_dir: Path,
) -> dict[str, Path]:
    ensure_output_dir(output_dir)
    paths = {
        "missing_values": output_dir / "missing_values.png",
        "numeric_distribution": output_dir / "numeric_distribution.png",
        "correlation_heatmap": output_dir / "correlation_heatmap.png",
        "target_distribution": output_dir / "target_distribution.png",
    }

    _plot_missing_values(dataframe, paths["missing_values"])
    _plot_numeric_distribution(dataframe, paths["numeric_distribution"])
    _plot_correlation_heatmap(dataframe, paths["correlation_heatmap"])
    _plot_target_distribution(dataframe, target_column, task_type, paths["target_distribution"])
    return paths


def _plot_missing_values(dataframe: pd.DataFrame, path: Path) -> None:
    missing_ratio = dataframe.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing_ratio.index.astype(str), missing_ratio.values, color="#4C78A8")
    ax.set_title("Missing Value Ratio by Column")
    ax.set_ylabel("Missing Ratio")
    ax.set_ylim(0, max(0.05, float(missing_ratio.max()) * 1.2))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_numeric_distribution(dataframe: pd.DataFrame, path: Path) -> None:
    numeric = dataframe.select_dtypes(include="number")
    columns = list(numeric.columns[:6])
    if not columns:
        _save_text_plot(path, "No numeric columns available")
        return

    n_cols = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_array = np.array(axes).reshape(-1)

    for ax, column in zip(axes_array, columns):
        values = numeric[column].dropna()
        ax.hist(values, bins=20, color="#59A14F", edgecolor="white")
        ax.set_title(str(column))
        ax.set_ylabel("Count")

    for ax in axes_array[len(columns):]:
        ax.axis("off")

    fig.suptitle("Numeric Feature Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(dataframe: pd.DataFrame, path: Path) -> None:
    numeric = dataframe.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        _save_text_plot(path, "At least two numeric columns are required for correlation heatmap")
        return

    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")

    for row in range(len(corr.columns)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    path: Path,
) -> None:
    target = dataframe[target_column].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))

    if task_type == "regression":
        ax.hist(target, bins=25, color="#F28E2B", edgecolor="white")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")
    else:
        counts = target.astype("object").value_counts().head(30)
        ax.bar(counts.index.astype(str), counts.values, color="#E15759")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    ax.set_title("Target Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_text_plot(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
