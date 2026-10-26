# AI Data Analysis Agent

A portfolio-ready data-analysis agent that accepts a natural-language goal and a CSV/XLSX file, selects from a bounded set of analysis tools, checks tool results, completes required deliverables deterministically, and exports an evidence-grounded report.

## What makes this an agent?

This project is more than a fixed sequence of pandas and scikit-learn calls. When an OpenAI-compatible API is configured, an LLM selects allowlisted tools through function calling based on the user's goal. Tool calls are validated, bounded, recorded with status and duration, and returned to the model as observations. The application independently completes required analysis steps if the planner skips a step or the API is unavailable.

Without an API key, the application remains usable through a deterministic tool plan.

## Features

- CSV and XLSX upload, schema preview, missing-cell and duplicate-row counts.
- Natural-language analysis goal plus user-selected target column.
- Tool-calling agent with a fixed allowlist, bounded turns/tool calls, argument validation, and execution trace.
- Dataset profile: data types, missingness, unique counts, numeric summaries, and categorical frequencies.
- Data-quality diagnostics: duplicate rows, constant columns, high missingness, and potential numeric IQR outliers. Findings are warnings; source rows are never silently deleted.
- Classification/regression task detection and baseline model comparison.
- Train-fitted preprocessing pipeline to avoid fitting imputation/scaling/encoding on the held-out test data.
- Hold-out metrics, plots, rule-based fallback summary, optional LLM summary, Markdown report, and optional Word report.
- Run-specific output directories to reduce artifact collisions between analysis runs.
- GitHub Actions checks for tests and Python compilation.

## Agent workflow

1. Load and validate the uploaded dataset.
2. Let the LLM select useful tools for the stated goal, if an API key is configured.
3. Execute only registered tools; validate arguments and record outcomes.
4. Ensure required deliverables exist using a deterministic completion policy.
5. Summarize measured evidence and limitations.
6. Export a Markdown report and, when available, a Word report.

Available tools: \`inspect_dataset\`, \`profile_dataset\`, \`analyze_data_quality\`, \`detect_task_type\`, \`train_baselines\`, \`evaluate_models\`, and \`generate_visualizations\`.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, matplotlib, OpenAI-compatible Chat Completions function calling, pytest, GitHub Actions.

## Quick start

\`\`\`bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
\`\`\`

Upload either \`examples/sample_classification.csv\` (target: \`churn\`) or \`examples/sample_regression.csv\` (target: \`price\`), describe your goal, select the target, and click **Run Agent**.

## Optional LLM configuration

Copy \`.env.example\` to \`.env\` and fill in a provider's OpenAI-compatible settings:

\`\`\`env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
\`\`\`

The default values are examples. Use a model/provider that supports Chat Completions tool calling. Never commit \`.env\` or expose API keys in screenshots.

## Tests

\`\`\`bash
pip install -r requirements.txt pytest
pytest -q
python -m compileall -q app.py src tools tests
\`\`\`

The integration test runs the classification workflow without an API key and verifies that a report and tool trace are produced.

## Safety and limitations

- The LLM can select only registered tools; this project does not execute arbitrary model-generated Python code.
- Tool calls have bounded turns and total call count. Each tool accepts no arguments in this version.
- Dataset values are treated as untrusted data, not instructions, when building prompts.
- Uploaded datasets are processed locally by the app. If an LLM API is configured, the current summary/planning context sends column names, aggregate profile information, and selected-target metadata to that provider; raw dataset rows are not intentionally included in the planning prompt. Review your provider's data handling terms before using sensitive data.
- Outlier flags use a simple IQR heuristic and require domain review. They do not imply data errors.
- Automatic task detection is heuristic; users should verify the selected target and task type.
- The models are baselines evaluated on one hold-out split, not a robust production model selection process. Cross-validation, leakage checks for time/grouped data, explainability, and hyperparameter search are future work.
- LLM-generated text may be wrong. Reported metrics are sourced from programmatic evaluation; interpretations should be checked.
- The agent does not currently support multi-file joins, SQL sources, arbitrary Python execution, or persistent multi-turn sessions.

## Project structure

\`\`\`text
ai-data-analysis-agent/
├── app.py
├── src/
│   ├── agent_orchestrator.py
│   ├── agent_runtime.py
│   ├── config.py
│   ├── llm_client.py
│   ├── prompts.py
│   └── utils.py
├── tools/
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_quality.py
│   ├── data_cleaner.py
│   ├── task_detector.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── visualizer.py
│   └── report_exporter.py
├── tests/
└── examples/
\`\`\`

## Resume-ready description

> Built a tool-calling AI Data Analysis Agent using Python and Streamlit. Implemented an allowlisted function-calling runtime with bounded execution, prerequisite-aware analysis tools, execution traces, deterministic fallback, and run-specific artifacts. Automated dataset profiling and quality diagnostics, classification/regression baseline evaluation with train-fitted preprocessing, visualizations, and Markdown/Word reporting; added pytest integration coverage and GitHub Actions CI.

Only claim tool-calling mode when tested with a configured compatible API. Record real test results before adding quantitative performance claims to a resume.
