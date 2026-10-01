"""Logging setup: one handler on the root logger, JSON lines or plain text.

Every line carries the current request id (set by the API middleware through
`request_id_var`), so all the lines of one request can be found together.

Callers must never put raw user text in a message or in `extra`: only masked text,
lengths, ids, routes and PII types.
"""

import json
import logging
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.config import get_settings, on_reset

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Attributes present on every LogRecord; anything else was passed through `extra=`.
# Also skipped: `message` and `asctime` (added by other formatters handling the same
# record), `taskName` (standard from Python 3.12) and `color_message` (the ANSI-coloured
# copy of the message that uvicorn attaches to its own records).
_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


def _extras(record: logging.LogRecord) -> dict:
    return {key: value for key, value in vars(record).items() if key not in _RECORD_ATTRS}


def _exception_summary(record: logging.LogRecord) -> tuple[str, str] | None:
    """Exception class name and stack of the record, without the exception message.

    The message is left out on purpose: it can quote the input that caused the error
    (a validation error repeats the offending value), and raw user text must not reach
    the logs. The class and the stack are enough to locate the failure.
    """
    if not record.exc_info or record.exc_info[0] is None:
        return None
    exc_type, _, tb = record.exc_info
    return exc_type.__name__, "".join(traceback.format_tb(tb))


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, request_id, then `extra` keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in _extras(record).items():
            payload.setdefault(key, value)  # an extra key never replaces a standard field
        summary = _exception_summary(record)
        if summary is not None:
            payload["exc_type"], payload["traceback"] = summary
        # default=str: an extra that is not JSON-serialisable (Path, datetime) must not
        # make the log call fail.
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Readable single line for local development; same content as the JSON form."""

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            self.formatTime(record, "%H:%M:%S"),
            record.levelname,
            record.name,
            f"[{request_id_var.get() or '-'}]",
            record.getMessage(),
        ]
        parts.extend(f"{key}={value}" for key, value in _extras(record).items())
        line = " ".join(parts)
        summary = _exception_summary(record)
        if summary is not None:
            exc_type, stack = summary
            line = f"{line}\n{stack}{exc_type}"
        return line


def _own_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if isinstance(h.formatter, JsonFormatter | TextFormatter)]


@on_reset
def _remove_handlers() -> None:
    """Remove the handler installed by `setup_logging`.

    Registered as a reset hook: the handler is built from the settings (format), so it
    is dropped with them and `setup_logging()` must be called again after a reset.
    """
    root = logging.getLogger()
    for handler in _own_handlers(root):
        root.removeHandler(handler)


def setup_logging() -> None:
    """Configure the root logger from the settings. Safe to call several times."""
    settings = get_settings()
    _remove_handlers()
    root = logging.getLogger()

    # stderr (the StreamHandler default) keeps stdout free for command-line output
    # such as the evaluation report.
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())
    root.addHandler(handler)

    level = logging.getLevelNamesMapping().get(settings.log_level.upper())
    root.setLevel(logging.INFO if level is None else level)

    # Uvicorn installs its own plain-text handlers. Its messages go through the root
    # handler so that the process emits a single format. Its access log is dropped: the
    # API middleware writes its own access line, without the query string.
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.propagate = True
    error_logger = logging.getLogger("uvicorn.error")
    error_logger.handlers.clear()
    error_logger.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False

    if level is None:
        get_logger(__name__).warning(
            "unknown LOG_LEVEL, using INFO", extra={"log_level": settings.log_level}
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
