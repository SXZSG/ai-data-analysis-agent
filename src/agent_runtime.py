import json
import time
from typing import Any, Callable

from src.utils import to_serializable


MAX_AGENT_TURNS = 10
MAX_TOOL_CALLS = 14
MAX_OBSERVATION_CHARS = 7000

_TOOL_DEFINITIONS = [
    ("inspect_dataset", "Inspect schema, row count, column names, types, missing counts, and unique counts. Never returns raw rows."),
    ("profile_dataset", "Build column statistics, numeric summaries, and categorical value counts."),
    ("analyze_data_quality", "Check missingness, duplicate rows, constant columns, high-missingness columns, and potential numeric IQR outliers. Never automatically deletes data."),
    ("detect_task_type", "Determine whether the user-selected target should be treated as classification or regression."),
    ("train_baselines", "Train baseline models using a preprocessing pipeline fitted only on the training split. Requires task detection."),
    ("evaluate_models", "Evaluate trained baselines on the held-out test split. Requires training."),
    ("generate_visualizations", "Generate dataset quality, distribution, correlation, and target plots."),
]

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }
    for name, description in _TOOL_DEFINITIONS
]


def run_tool_calling_loop(
    llm_client: Any,
    messages: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[[], Any]],
    state: dict[str, Any],
    log_callback: Callable[[str], None] | None = None,
) -> str:
    """Run a bounded allowlisted function-calling loop and return the final assistant message."""
    conversation = list(messages)
    completed_tools: set[str] = set()
    total_tool_calls = 0
    final_text = ""

    for _ in range(MAX_AGENT_TURNS):
        response = llm_client.create_agent_completion(conversation, AGENT_TOOLS)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            final_text = (message.content or "").strip()
            break

        conversation.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            if total_tool_calls >= MAX_TOOL_CALLS:
                break
            total_tool_calls += 1
            tool_name = call.function.name
            started = time.perf_counter()
            status = "error"
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if arguments:
                    raise ValueError("This tool accepts no arguments.")
                if tool_name not in tool_handlers:
                    raise ValueError(f"Unknown tool: {tool_name}")
                if tool_name in completed_tools:
                    result = {"status": "already_completed"}
                else:
                    result = tool_handlers[tool_name]()
                    completed_tools.add(tool_name)
                payload = {"ok": True, "result": to_serializable(result)}
                status = "success"
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": str(exc)[:500],
                    "hint": "Check prerequisite tools, then retry only if the prerequisite can be completed.",
                }

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            observation = json.dumps(payload, ensure_ascii=False, default=str)
            if len(observation) > MAX_OBSERVATION_CHARS:
                observation = observation[:MAX_OBSERVATION_CHARS] + " ... [observation truncated]"
            state.setdefault("tool_trace", []).append(
                {
                    "tool_name": tool_name,
                    "status": status,
                    "duration_ms": elapsed_ms,
                    "source": "llm",
                    "observation": observation,
                }
            )
            if log_callback:
                log_callback(f"Tool {tool_name}: {status} ({elapsed_ms} ms)")
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                }
            )

        if total_tool_calls >= MAX_TOOL_CALLS:
            break

    return final_text
