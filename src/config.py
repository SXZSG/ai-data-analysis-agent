import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


@dataclass(frozen=True)
class AppConfig:
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str


def get_config() -> AppConfig:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("LLM_API_KEY") or None
    base_url = os.getenv("LLM_BASE_URL") or None
    model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    return AppConfig(
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=model,
    )
