from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_orchestrator import DataAnalysisAgent
from src.config import OUTPUT_DIR, get_config
from src.utils import ensure_output_dir, safe_filename
from tools.data_loader import load_dataset


st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)


def save_uploaded_file(uploaded_file) -> Path:
    ensure_output_dir(OUTPUT_DIR)
    suffix = Path(uploaded_file.name).suffix.lower()
    file_name = safe_filename(Path(uploaded_file.name).stem) or "uploaded_dataset"
    saved_path = OUTPUT_DIR / f"{file_name}{suffix}"
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
    st.caption("Upload a CSV or Excel dataset, choose a target column, and run an automated analysis workflow.")

    with st.sidebar:
        st.header("Configuration")
        st.write(f"LLM model: `{config.llm_model}`")
        if config.llm_api_key:
            st.success("LLM API key detected")
        else:
            st.info("No LLM API key found. Rule-based report generation will be used.")
        st.write(f"Outputs: `{OUTPUT_DIR}`")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("Upload a dataset to begin. Demo files are available in the `examples/` folder.")
        return

    try:
        preview_result = load_dataset(uploaded_file, file_name=uploaded_file.name)
        uploaded_file.seek(0)
    except Exception as exc:
        st.error(f"Failed to read dataset: {exc}")
        return

    df = preview_result.dataframe
    st.subheader("Dataset Preview")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{df.shape[0]:,}")
    col_b.metric("Columns", f"{df.shape[1]:,}")
    col_c.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    st.dataframe(df.head(10), use_container_width=True)

    target_column = st.selectbox("Target column", options=list(df.columns))
    run_clicked = st.button("Run Agent", type="primary")

    if not run_clicked:
        return

    dataset_path = save_uploaded_file(uploaded_file)
    agent = DataAnalysisAgent(output_dir=OUTPUT_DIR)
    log_box = st.empty()
    logs: list[str] = []

    def log_callback(message: str) -> None:
        logs.append(message)
        log_box.markdown("\n".join(f"- {line}" for line in logs))

    with st.spinner("Running agent workflow..."):
        try:
            result = agent.run(
                dataset_path=dataset_path,
                target_column=target_column,
                log_callback=log_callback,
            )
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")
            return

    st.success("Agent workflow completed.")

    overview_tab, profile_tab, model_tab, chart_tab, report_tab = st.tabs(
        ["Summary", "Data Profile", "Models", "Visualizations", "Report"]
    )

    with overview_tab:
        st.subheader("AI Summary")
        st.markdown(result["ai_summary"])
        st.subheader("Detected Task")
        st.write(result["task_detection"]["task_type"])
        st.caption(result["task_detection"]["reason"])

    with profile_tab:
        st.subheader("Column Profile")
        st.dataframe(pd.DataFrame(result["profile"]["column_profiles"]), use_container_width=True)

        if result["profile"]["numeric_summary"]:
            st.subheader("Numeric Summary")
            st.dataframe(pd.DataFrame(result["profile"]["numeric_summary"]).T, use_container_width=True)

        if result["profile"]["categorical_summary"]:
            st.subheader("Categorical Top Values")
            st.json(result["profile"]["categorical_summary"])

    with model_tab:
        st.subheader("Baseline Model Results")
        render_model_results(result["evaluation"]["results"])
        st.info(f"Best baseline model: `{result['evaluation']['best_model']['model_name']}`")

    with chart_tab:
        st.subheader("Generated Visualizations")
        image_cols = st.columns(2)
        for idx, (name, path) in enumerate(result["visualizations"].items()):
            with image_cols[idx % 2]:
                st.caption(name)
                st.image(str(path), use_container_width=True)

    with report_tab:
        report_path = Path(result["report_paths"]["markdown"])
        docx_path = result["report_paths"].get("docx")
        st.subheader("Markdown Report")
        st.code(report_path.read_text(encoding="utf-8"), language="markdown")
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


if __name__ == "__main__":
    main()
