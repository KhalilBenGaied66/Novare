"""Logging setup: JSON and text formats, request id, extras, idempotence."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from app.core import logging as app_logging
from app.core.config import reset_settings
from app.core.logging import get_logger, request_id_var, setup_logging


@pytest.fixture(autouse=True)
def clean_logging():
    """Leave the process-wide logging state as it was found."""
    root = logging.getLogger()
    level = root.level
    app_logging._remove_handlers()
    yield
    app_logging._remove_handlers()
    root.setLevel(level)
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = True


@pytest.fixture
def configure(monkeypatch):
    """Set LOG_JSON / LOG_LEVEL, rebuild the settings and set logging up."""

    def _configure(log_json: bool = True, level: str = "INFO") -> None:
        monkeypatch.setenv("LOG_JSON", "true" if log_json else "false")
        monkeypatch.setenv("LOG_LEVEL", level)
        reset_settings()
        setup_logging()

    return _configure


def json_lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()]


def test_json_line_has_the_standard_fields(configure, capsys):
    configure(log_json=True)
    get_logger("app.test").info("request handled")

    (record,) = json_lines(capsys)
    assert set(record) == {"ts", "level", "logger", "msg", "request_id"}
    assert record["level"] == "INFO"
    assert record["logger"] == "app.test"
    assert record["msg"] == "request handled"
    assert record["request_id"] == ""
    assert datetime.fromisoformat(record["ts"]).utcoffset().total_seconds() == 0


def test_json_line_carries_the_request_id_of_the_context(configure, capsys):
    configure(log_json=True)
    token = request_id_var.set("req-42")
    try:
        get_logger("app.test").info("inside")
    finally:
        request_id_var.reset(token)
    get_logger("app.test").info("outside")

    inside, outside = json_lines(capsys)
    assert inside["request_id"] == "req-42"
    assert outside["request_id"] == ""


def test_extra_keys_are_added_to_the_json_line(configure, capsys):
    configure(log_json=True)
    get_logger("app.test").info(
        "ask", extra={"route": "rag", "latency_ms": 12, "pii_types": ["TEL"], "path": Path("a/b")}
    )

    (record,) = json_lines(capsys)
    assert record["route"] == "rag"
    assert record["latency_ms"] == 12
    assert record["pii_types"] == ["TEL"]
    assert record["path"] == str(Path("a/b"))  # not JSON-serialisable: logged as text


def test_extra_keys_cannot_replace_standard_fields(configure, capsys):
    configure(log_json=True)
    get_logger("app.test").warning("real", extra={"level": "DEBUG", "request_id": "forged"})

    (record,) = json_lines(capsys)
    assert record["level"] == "WARNING"
    assert record["request_id"] == ""


def test_message_arguments_and_french_text_are_kept_readable(configure, capsys):
    configure(log_json=True)
    get_logger("app.test").info("route %s en %d ms", "détection", 7)

    line = capsys.readouterr().err
    assert json.loads(line)["msg"] == "route détection en 7 ms"
    assert "détection" in line  # not escaped as é


def test_exception_is_logged_by_class_and_stack_without_its_message(configure, capsys):
    configure(log_json=True)
    # Built at run time so that the text is not in the source line shown by the stack.
    user_text = " ".join(["texte", "brut", "du", "client"])
    try:
        raise ValueError(user_text)
    except ValueError:
        get_logger("app.test").exception("unexpected failure")

    err = capsys.readouterr().err
    assert user_text not in err
    record = json.loads(err)
    assert record["exc_type"] == "ValueError"
    assert "test_exception_is_logged_by_class_and_stack_without_its_message" in record["traceback"]
    assert record["msg"] == "unexpected failure"


def test_setup_is_idempotent(configure, capsys):
    configure(log_json=True)
    setup_logging()
    setup_logging()
    get_logger("app.test").info("once")

    assert len(json_lines(capsys)) == 1
    assert len(app_logging._own_handlers(logging.getLogger())) == 1


def test_level_comes_from_the_settings(configure, capsys):
    configure(log_json=True, level="warning")
    logger = get_logger("app.test")
    logger.info("hidden")
    logger.warning("shown")

    assert [record["msg"] for record in json_lines(capsys)] == ["shown"]


def test_unknown_level_falls_back_to_info_and_says_so(configure, capsys):
    configure(log_json=True, level="verbose")
    get_logger("app.test").debug("hidden")
    get_logger("app.test").info("shown")

    warning, shown = json_lines(capsys)
    assert warning["level"] == "WARNING"
    assert warning["log_level"] == "verbose"
    assert shown["msg"] == "shown"
    assert logging.getLogger().level == logging.INFO


def test_text_format_shows_request_id_and_extras(configure, capsys):
    configure(log_json=False)
    get_logger("app.test").info("no request")
    token = request_id_var.set("req-42")
    try:
        get_logger("app.test").warning("ask", extra={"route": "rag"})
    finally:
        request_id_var.reset(token)

    first, second = capsys.readouterr().err.splitlines()
    assert first.endswith("INFO app.test [-] no request")
    assert second.endswith("WARNING app.test [req-42] ask route=rag")


def test_text_format_logs_exception_class_without_its_message(configure, capsys):
    configure(log_json=False)
    user_text = " ".join(["texte", "brut", "du", "client"])
    try:
        raise KeyError(user_text)
    except KeyError:
        get_logger("app.test").exception("lookup failed")

    err = capsys.readouterr().err
    lines = err.splitlines()
    assert user_text not in err
    assert lines[0].endswith("ERROR app.test [-] lookup failed")
    assert "test_text_format_logs_exception_class_without_its_message" in err
    assert lines[-1] == "KeyError"


def test_settings_reset_removes_the_handler(configure, capsys):
    configure(log_json=True)
    reset_settings()
    get_logger("app.test").info("after reset")

    assert app_logging._own_handlers(logging.getLogger()) == []
    assert capsys.readouterr().err == ""


def test_setup_after_a_reset_applies_the_new_settings(configure, capsys):
    configure(log_json=True)
    configure(log_json=False)
    get_logger("app.test").info("now in text")

    assert capsys.readouterr().err.rstrip().endswith("INFO app.test [-] now in text")
    assert len(app_logging._own_handlers(logging.getLogger())) == 1


def test_uvicorn_messages_use_the_same_format_and_its_access_log_is_dropped(configure, capsys):
    logging.getLogger("uvicorn").addHandler(logging.StreamHandler())
    configure(log_json=True)
    # Uvicorn attaches an ANSI-coloured copy of its messages as an extra.
    logging.getLogger("uvicorn.error").info(
        "Application startup complete.",
        extra={"color_message": "\x1b[1mApplication startup complete.\x1b[0m"},
    )
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /x?q=secret HTTP/1.1" 200')

    (record,) = json_lines(capsys)
    assert record["logger"] == "uvicorn.error"
    assert record["msg"] == "Application startup complete."
    assert "color_message" not in record
    assert logging.getLogger("uvicorn").handlers == []
