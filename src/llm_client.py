from openai import OpenAI

from src.config import get_config
from src.prompts import SYSTEM_PROMPT, build_analysis_prompt


class LLMClient:
    def __init__(self) -> None:
        self.config = get_config()
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.config.llm_api_key)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.config.llm_api_key:
                raise RuntimeError("LLM_API_KEY is not configured.")
            self._client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                timeout=45.0,
                max_retries=1,
            )
        return self._client

    def create_agent_completion(self, messages: list[dict], tools: list[dict]):
        """Ask an OpenAI-compatible chat-completions endpoint to select allowlisted tools."""
        if not self.available:
            raise RuntimeError("LLM_API_KEY is not configured.")
        return self._get_client().chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )

    def generate_summary(self, context: dict) -> str:
        if not self.available:
            return generate_rule_based_summary(context)

        try:
            response = self._get_client().chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_analysis_prompt(context)},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return content.strip() if content else generate_rule_based_summary(context)
        except Exception:
            return (
                generate_rule_based_summary(context)
                + "\n\n_LLM summary generation failed; a deterministic summary was used instead._"
            )


def generate_rule_based_summary(context: dict) -> str:
    dataset = context.get("dataset", {})
    profile = context.get("profile", {})
    task_detection = context.get("task_detection", {})
    evaluation = context.get("evaluation", {})
    target = context.get("target", {})
    correlations = context.get("correlations", [])
    quality = context.get("data_quality", {})

    row_count = dataset.get("rows", 0)
    column_count = dataset.get("columns", 0)
    missing_columns = [
        item["column"]
        for item in profile.get("column_profiles", [])
        if item.get("missing_ratio", 0) > 0
    ]
    best_model = evaluation.get("best_model", {})

    lines = [
        "## Analysis Summary",
        "",
        f"- Dataset scale: {row_count} rows and {column_count} columns.",
        f"- Main fields: {', '.join(dataset.get('column_names', [])[:12])}.",
        f"- Task type: {task_detection.get('task_type', 'unknown')}.",
        f"- Target column: {target.get('name', 'unknown')}.",
    ]

    if missing_columns:
        lines.append(f"- Columns containing missing values: {', '.join(missing_columns[:10])}.")
    else:
        lines.append("- No missing values were detected.")

    if quality:
        lines.append(
            f"- Quality diagnostics: {quality.get('duplicate_rows', 0)} duplicate rows; "
            f"{len(quality.get('constant_columns', []))} constant columns; "
            f"{len(quality.get('numeric_iqr_outlier_warnings', []))} numeric columns flagged for IQR outlier review."
        )
    if target.get("distribution"):
        lines.append(f"- Target distribution/statistics: {target['distribution']}.")
    if correlations:
        formatted = ", ".join(
            f"{item['feature']} (r={item['correlation']:.3f})" for item in correlations[:5]
        )
        lines.append(f"- Largest absolute numeric correlations with the regression target: {formatted}.")
    if best_model:
        lines.append(
            f"- Best baseline by held-out {evaluation.get('selection_metric', 'selection metric')}: "
            f"{best_model.get('model_name')} with {best_model.get('metrics')}."
        )
    lines.extend(
        [
            "- Interpretation caution: correlations are not causal effects; hold-out metrics are a baseline, not a production guarantee.",
            "- Suggested next steps: validate domain definitions, inspect flagged quality issues, establish a cross-validation baseline, and compare against a simple domain-appropriate reference model.",
        ]
    )
    return "\n".join(lines)
