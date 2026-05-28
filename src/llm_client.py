from openai import OpenAI

from src.config import get_config
from src.prompts import SYSTEM_PROMPT, build_analysis_prompt


class LLMClient:
    def __init__(self) -> None:
        self.config = get_config()

    def generate_summary(self, context: dict) -> str:
        if not self.config.llm_api_key:
            return generate_rule_based_summary(context)

        try:
            client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_analysis_prompt(context)},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return content.strip() if content else generate_rule_based_summary(context)
        except Exception as exc:
            fallback = generate_rule_based_summary(context)
            return f"{fallback}\n\n> LLM generation failed, rule-based fallback was used. Error: {exc}"


def generate_rule_based_summary(context: dict) -> str:
    dataset = context.get("dataset", {})
    profile = context.get("profile", {})
    task_detection = context.get("task_detection", {})
    evaluation = context.get("evaluation", {})
    target = context.get("target", {})
    correlations = context.get("correlations", [])

    row_count = dataset.get("rows", 0)
    column_count = dataset.get("columns", 0)
    missing_columns = [
        item["column"]
        for item in profile.get("column_profiles", [])
        if item.get("missing_ratio", 0) > 0
    ]
    best_model = evaluation.get("best_model", {})

    lines = [
        "## AI Summary",
        "",
        f"- Dataset scale: {row_count} rows and {column_count} columns.",
        f"- Main fields: {', '.join(dataset.get('column_names', [])[:12])}.",
        f"- Task type: {task_detection.get('task_type', 'unknown')}.",
        f"- Target column: `{target.get('name', 'unknown')}`.",
    ]

    if missing_columns:
        lines.append(f"- Data quality issues: missing values were found in {', '.join(missing_columns[:10])}.")
    else:
        lines.append("- Data quality issues: no missing values were detected.")

    if target.get("distribution"):
        lines.append(f"- Target distribution: {target['distribution']}.")

    if correlations:
        formatted = ", ".join(
            f"{item['feature']} ({item['correlation']:.3f})" for item in correlations[:5]
        )
        lines.append(f"- Important correlated features: {formatted}.")
    else:
        lines.append("- Important correlated features: no strong numeric correlations were available.")

    if best_model:
        lines.append(f"- Best baseline model: {best_model.get('model_name')} with metrics {best_model.get('metrics')}.")

    lines.extend(
        [
            "- Next steps: validate data definitions, add domain-specific feature engineering, tune model hyperparameters, and compare cross-validation results before production use.",
            "",
            "_This summary was generated with the rule-based fallback because no LLM API key was configured or the LLM request failed._",
        ]
    )
    return "\n".join(lines)
