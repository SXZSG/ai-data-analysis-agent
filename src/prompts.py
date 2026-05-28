import json

from src.utils import to_serializable


SYSTEM_PROMPT = (
    "You are a senior data analyst and machine learning engineer. "
    "Write concise, practical analysis reports for tabular datasets. "
    "Focus on data quality, target behavior, baseline model results, and next steps."
)


def build_analysis_prompt(context: dict) -> str:
    payload = json.dumps(to_serializable(context), ensure_ascii=False, indent=2)
    return f"""
Generate an AI data analysis summary in Markdown.

The summary must cover:
1. Dataset scale
2. Main fields
3. Data quality issues
4. Target distribution
5. Important correlated features
6. Baseline model performance
7. Practical next-step recommendations

Analysis context:
{payload}
""".strip()
