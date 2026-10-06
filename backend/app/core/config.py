"""Application settings: one source of truth, read from the environment and `.env`.

Every module reads configuration through `get_settings()` at call time (never at
import time), so tests can override values with environment variables.
"""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repository root is three levels above `core/`.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Provider keys (MISTRAL_API_KEY, ANTHROPIC_API_KEY, ...) are read by LiteLLM from
# os.environ, so `.env` must be exported there too. Real env vars keep precedence.
load_dotenv(REPO_ROOT / ".env", override=False)


# What a call is made of. LLM_EXTRA_PARAMS adds to it and never replaces any of it.
_REQUEST_PARAMETERS = frozenset({"model", "messages", "tools", "tool_choice", "max_tokens"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Paths -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    evals_dir: Path = REPO_ROOT / "evals"

    # --- API -------------------------------------------------------------
    api_key: str = ""  # empty = authentication disabled (local development only)
    cors_origins: str = ""  # comma-separated list, empty = no CORS headers
    rate_limit_per_minute: int = 60  # per API key / client IP, 0 = disabled
    auto_ingest: bool = True  # build the index at startup when it is missing

    # --- Storage ---------------------------------------------------------
    database_url: str = ""  # empty = SQLite file under data_dir

    # --- Retrieval -------------------------------------------------------
    retrieval_mode: Literal["hybrid", "bm25"] = "hybrid"
    embedding_backend: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qdrant_url: str = ""  # empty = local NumPy vector file, set = Qdrant server
    qdrant_api_key: str = ""
    qdrant_collection: str = "dossierops_fr"
    top_k: int = 4
    candidate_k: int = 20  # candidates per retriever before fusion
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_confidence: float = 0.35  # below this, the RAG route escalates to a human

    # --- LLM -------------------------------------------------------------
    # auto = use the LLM when the provider key of the task's model is present.
    llm_enabled: Literal["auto", "on", "off"] = "auto"
    rag_model: str = "mistral/mistral-small-latest"
    agent_model: str = "anthropic/claude-sonnet-5-5"
    judge_model: str = "anthropic/claude-haiku-4-5"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    # Provider-specific parameters added to every call, as a JSON object. Hosted models
    # need none. A local model served by Ollama takes its context size, its sampling
    # temperature and its reasoning switch here:
    # {"num_ctx": 16384, "temperature": 0, "reasoning_effort": "none"}
    llm_extra_params: dict[str, Any] = {}
    rag_max_tokens: int = 600
    agent_max_tokens: int = 4000
    agent_max_steps: int = 6
    max_cost_eur_per_request: float = 0.02  # RG-08
    usd_to_eur: float = 0.90  # indicative rate used to convert provider prices

    # --- Business rules --------------------------------------------------
    rg03_max_amount: float = 500.0  # RG-03: disputes strictly below are automated
    ticket_dedupe_days: int = 7

    # --- Logging ---------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("llm_extra_params")
    @classmethod
    def _extra_params_leave_the_request_alone(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(_REQUEST_PARAMETERS & set(value))
        if reserved:
            raise ValueError(f"set by the application, not by LLM_EXTRA_PARAMS: {reserved}")
        return value

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "sample_docs"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'dossierops.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Cached singletons built from the settings (DB engine, retriever, client repository...)
# register a reset function here so that one call rebuilds everything consistently.
_RESET_HOOKS: list[Callable[[], None]] = []


def on_reset(hook: Callable[[], None]) -> Callable[[], None]:
    """Register `hook` to run whenever settings are reset. Usable as a decorator."""
    _RESET_HOOKS.append(hook)
    return hook


def reset_settings() -> None:
    """Drop the cached settings and every singleton derived from them."""
    get_settings.cache_clear()
    for hook in _RESET_HOOKS:
        hook()
