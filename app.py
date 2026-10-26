from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.agent_orchestrator import DataAnalysisAgent
from src.config import OUTPUT_DIR, get_config
from src.utils import ensure_output_dir, safe_filename
from tools.data_loader import load_dataset


st.set_page_config(page_title="AI Data Analysis Agent", page_icon="📊", layout="wide")


def save_uploaded_file(uploaded_file) -> Path:
    ensure_output_dir(OUTPUT_DIR)
    suffix = Path(uploaded_file.name).suffix.lower()
    file_name = safe_filename(Path(uploaded_file.name).stem) or "uploaded_dataset"
    saved_path = OUTPUT_DIR / f"{file_name}_{uuid4().hex[:8]}{suffix}"
    saved_path.write_bytes(uploaded_file.getbuffer())
    return saved_path


def render_model_results(results: list[dict]) -> None:
    rows = []
    for item in results:
        row = {"model": item["model_name"]}
        row.update(item["metrics"])
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main() -> None:
    config = get_config()
    st.title("AI Data Analysis Agent")
    st.caption(
        "A tool-calling agent that inspects tabular data, checks quality, trains baselines, "
        "evaluates evidence, and creates a downloadable report."
    )

    with st.sidebar:
        st.header("Agent Configuration")
        st.write(f"LLM model: {config.llm_model}")
        if config.llm_api_key:
            st.success("LLM tool-calling enabled")
            st.caption("Tool execution is allowlisted and bounded; required outputs have a deterministic completion path.")
        else:
            st.info("No LLM API key. The deterministic tool plan will run, and the app remains usable.")
        st.write(f"Outputs: {OUTPUT_DIR}")

    uploaded_file = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx"])
    user_goal = st.text_area(
        "What do you want the agent to do?",
        value="Analyze data quality and important patterns, compare baseline models for the selected target, "
        "explain the evidence and limitations, and produce a report.",
        help="Describe the analysis goal in natural language. The selected target column is used for baseline modeling.",
        height=100,
    )

    if not uploaded_file:
        st.info("Upload a dataset to begin. Demo datasets are available in the examples folder.")
        return

    try:
        preview_result = load_dataset(uploaded_file, file_name=uploaded_file.name)
        uploaded_file.seek(0)
    except Exception as exc:
        st.error(f"Failed to read dataset: {exc}")
        return

    df = preview_result.dataframe
    st.subheader("Dataset Preview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", f"{df.shape[0]:,}")
    col_b.metric("Columns", f"{df.shape[1]:,}")
    col_c.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    col_d.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(10), use_container_width=True)

    target_column = st.selectbox("Target column for baseline modeling", options=list(df.columns))
    run_clicked = st.button("Run Agent", type="primary")
    if not run_clicked:
        return
    if not user_goal.strip():
        st.warning("Please describe the analysis goal.")
        return

    dataset_path = save_uploaded_file(uploaded_file)
    agent = DataAnalysisAgent(output_dir=OUTPUT_DIR)
    log_box = st.empty()
    logs: list[str] = []

    def log_callback(message: str) -> None:
        logs.append(message)
        log_box.markdown("\n".join(f"- {line}" for line in logs))

    with st.spinner("Agent is planning and executing tools..."):
        try:
            result = agent.run(
                dataset_path=dataset_path,
                target_column=target_column,
                user_goal=user_goal,
                log_callback=log_callback,
            )
        except Exception as exc:
            st.error(f"Agent run failed: {type(exc).__name__}: {exc}")
            st.caption("Check the dataset, target column, and console traceback for diagnostics.")
            return

    st.success("Analysis run completed.")
    st.caption(f"Run ID: {result['run_id']} · Mode: {result['agent_mode']}")

    overview_tab, profile_tab, quality_tab, model_tab, chart_tab, trace_tab, report_tab = st.tabs(
        ["Summary", "Data Profile", "Data Quality", "Models", "Visualizations", "Agent Trace", "Report"]
    )

    with overview_tab:
        st.subheader("Analysis Summary")
        st.markdown(result["ai_summary"])
        st.subheader("Detected Task")
        st.write(result["task_detection"]["task_type"])
        st.caption(result["task_detection"]["reason"])
        if result.get("agent_final_message"):
            with st.expander("Agent planning note"):
                st.write(result["agent_final_message"])

    with profile_tab:
        st.subheader("Column Profile")
        st.dataframe(pd.DataFrame(result["profile"]["column_profiles"]), use_container_width=True)
        if result["profile"]["numeric_summary"]:
            st.subheader("Numeric Summary")
            st.dataframe(pd.DataFrame(result["profile"]["numeric_summary"]).T, use_container_width=True)
        if result["profile"]["categorical_summary"]:
            st.subheader("Categorical Top Values")
            st.json(result["profile"]["categorical_summary"])

    with quality_tab:
        quality = result["data_quality"]
        quality_a, quality_b, quality_c = st.columns(3)
        quality_a.metric("Duplicate Rows", quality["duplicate_rows"])
        quality_b.metric("Constant Columns", len(quality["constant_columns"]))
        quality_c.metric("High-Missingness Columns", len(quality["high_missing_columns"]))
        st.caption(quality["policy"])
        if quality["high_missing_columns"]:
            st.write("Columns with high missingness")
            st.dataframe(pd.DataFrame(quality["high_missing_columns"]), use_container_width=True)
        if quality["constant_columns"]:
            st.write("Constant columns")
            st.write(", ".join(quality["constant_columns"]))
        if quality["numeric_iqr_outlier_warnings"]:
            st.write("Potential numeric outliers (IQR rule; review, do not auto-delete)")
            st.dataframe(pd.DataFrame(quality["numeric_iqr_outlier_warnings"]), use_container_width=True)
        else:
            st.info("No numeric columns were flagged by the IQR warning rule.")

    with model_tab:
        st.subheader("Baseline Model Results")
        render_model_results(result["evaluation"]["results"])
        st.info(
            f"Best baseline by held-out {result['evaluation']['selection_metric']}: "
            f"{result['evaluation']['best_model']['model_name']}"
        )
        st.caption("These are hold-out baseline metrics, not a production performance guarantee.")

    with chart_tab:
        st.subheader("Generated Visualizations")
        image_cols = st.columns(2)
        for idx, (name, path) in enumerate(result["visualizations"].items()):
            with image_cols[idx % 2]:
                st.caption(name.replace("_", " ").title())
                st.image(str(path), use_container_width=True)

    with trace_tab:
        st.subheader("Tool Execution Trace")
        trace_rows = [
            {
                "tool": item["tool_name"],
                "status": item["status"],
                "source": item["source"],
                "duration_ms": item["duration_ms"],
            }
            for item in result["tool_trace"]
        ]
        if trace_rows:
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)
        else:
            st.info("No tool calls were recorded.")
        with st.expander("Full run logs"):
            st.code("\n".join(result["logs"]), language="text")

    with report_tab:
        report_path = Path(result["report_paths"]["markdown"])
        docx_path = result["report_paths"].get("docx")
        st.subheader("Downloadable Report")
        st.download_button(
            "Download Markdown Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/markdown",
        )
        if docx_path and Path(docx_path).exists():
            st.download_button(
                "Download Word Report",
                data=Path(docx_path).read_bytes(),
                file_name=Path(docx_path).name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        st.caption(f"Artifacts saved under: {report_path.parent}")
        with st.expander("Preview Markdown"):
            st.code(report_path.read_text(encoding="utf-8"), language="markdown")


if __name__ == "__main__":
    main()
