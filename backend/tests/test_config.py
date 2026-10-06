"""Settings: environment overrides, derived paths, cache and reset hooks."""

import pytest
from pydantic import ValidationError

from app.core import config
from app.core.config import REPO_ROOT, get_settings, on_reset, reset_settings


def test_repo_root_is_the_repository_root():
    assert REPO_ROOT.is_absolute()
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "backend" / "app" / "core" / "config.py").is_file()


def test_environment_overrides_are_applied(env, tmp_path):
    # Values set by the `env` fixture of conftest.py.
    assert env.data_dir == tmp_path / "data"
    assert env.evals_dir == tmp_path / "evals"
    assert env.llm_enabled == "off"
    assert env.embedding_backend == "hash"
    assert env.api_key == ""
    assert env.rate_limit_per_minute == 0
    assert env.auto_ingest is False
    assert env.log_json is False
    assert env.min_confidence == 0.35


def test_derived_paths_follow_data_dir(env):
    assert env.docs_dir == env.data_dir / "sample_docs"
    assert env.reference_dir == env.data_dir / "reference"
    assert env.index_dir == env.data_dir / "index"
    assert env.model_cache_dir == env.data_dir / "models"
    assert env.docs_dir.is_absolute()


def test_database_url_defaults_to_a_sqlite_file_under_data_dir(env):
    url = env.resolved_database_url
    assert url == f"sqlite:///{(env.data_dir / 'dossierops.db').as_posix()}"
    assert "\\" not in url


def test_explicit_database_url_is_used_as_is(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://app@db:5432/dossierops")
    reset_settings()
    assert get_settings().resolved_database_url == "postgresql+psycopg://app@db:5432/dossierops"


def test_extra_llm_parameters_are_a_json_object(monkeypatch):
    assert get_settings().llm_extra_params == {}
    monkeypatch.setenv("LLM_EXTRA_PARAMS", '{"num_ctx": 16384, "temperature": 0}')
    reset_settings()
    assert get_settings().llm_extra_params == {"num_ctx": 16384, "temperature": 0}


@pytest.mark.parametrize("value", ['{"model": "another/model"}', '{"messages": []}', "[1, 2]"])
def test_extra_llm_parameters_cannot_replace_the_request(monkeypatch, value):
    monkeypatch.setenv("LLM_EXTRA_PARAMS", value)
    reset_settings()
    with pytest.raises(ValidationError):
        get_settings()


def test_settings_are_cached_until_reset(monkeypatch):
    first = get_settings()
    monkeypatch.setenv("TOP_K", "7")
    assert get_settings() is first  # still the cached object

    reset_settings()
    assert get_settings() is not first
    assert get_settings().top_k == 7


def test_values_are_parsed_to_their_types(monkeypatch):
    monkeypatch.setenv("RG03_MAX_AMOUNT", "750.5")
    monkeypatch.setenv("AUTO_INGEST", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "30")
    reset_settings()
    settings = get_settings()
    assert settings.rg03_max_amount == 750.5
    assert settings.auto_ingest is True
    assert settings.rate_limit_per_minute == 30


def test_invalid_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "semantic")
    reset_settings()
    with pytest.raises(ValidationError):
        get_settings()


def test_unknown_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("NOT_A_SETTING", "x")
    reset_settings()
    assert not hasattr(get_settings(), "not_a_setting")


def test_reset_hooks_run_on_every_reset(monkeypatch):
    # Work on a copy so the hook registered here does not outlive the test.
    monkeypatch.setattr(config, "_RESET_HOOKS", list(config._RESET_HOOKS))
    calls = []

    @on_reset
    def hook() -> None:
        calls.append("reset")

    assert callable(hook)  # usable as a decorator: the function is returned
    reset_settings()
    reset_settings()
    assert calls == ["reset", "reset"]
