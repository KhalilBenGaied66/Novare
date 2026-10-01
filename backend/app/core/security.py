"""API authentication and rate limiting, as FastAPI dependencies.

Limits of the rate limiter (documented, accepted for a single-process deployment):
- counters live in the memory of the process: with N workers the effective limit is
  N times higher, and a restart clears them. A shared store (Redis) would be needed to
  enforce one limit across processes;
- behind a reverse proxy the client host is the proxy's address unless uvicorn is
  started with `--proxy-headers`.
"""

import hmac
import math
import threading
import time

from fastapi import Header, HTTPException, Request, status

from app.core.config import get_settings, on_reset

WINDOW_S = 60.0
# Above this number of tracked callers, expired windows are purged so that the table
# cannot grow without bound when requests come from many hosts.
_MAX_TRACKED_CALLERS = 10_000

# Sync dependencies run in FastAPI's thread pool, so the table is shared between threads.
_lock = threading.Lock()
# caller id -> (start of its current window, requests counted in that window)
_windows: dict[str, tuple[float, int]] = {}

# Counters are dropped with the settings so that every test starts from an empty table.
on_reset(_windows.clear)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject the request unless the `X-API-Key` header matches `settings.api_key`.

    An empty `api_key` setting disables authentication (local development only).
    """
    expected = get_settings().api_key
    if not expected:
        return
    if not _same_key(x_api_key or "", expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou absente",
            headers={"WWW-Authenticate": "APIKey"},
        )


def rate_limit(request: Request) -> None:
    """Allow `settings.rate_limit_per_minute` requests per caller and 60 s window.

    Fixed window: it opens with the caller's first request and the counter restarts
    60 s later. 0 disables the limit. A refused request gets a 429 with `Retry-After`.
    """
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    caller = _caller_id(request)
    now = _now()
    with _lock:
        if len(_windows) > _MAX_TRACKED_CALLERS:
            _drop_expired(now)
        start, count = _windows.get(caller, (now, 0))
        if now - start >= WINDOW_S:
            start, count = now, 0
        if count < limit:
            _windows[caller] = (start, count + 1)
            return
        retry_after = max(1, math.ceil(start + WINDOW_S - now))
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Trop de requêtes : réessayez dans {retry_after} s",
        headers={"Retry-After": str(retry_after)},
    )


def _same_key(provided: str, expected: str) -> bool:
    # Constant-time comparison, so the response time does not reveal how many leading
    # characters are right. On bytes because compare_digest rejects non-ASCII str.
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _caller_id(request: Request) -> str:
    """Whose counter to use: the API key when a valid one is sent, else the client host.

    The header is checked before it is trusted: otherwise a caller could get a fresh
    counter for every request just by sending a different made-up key.
    """
    expected = get_settings().api_key
    if expected and _same_key(request.headers.get("x-api-key", ""), expected):
        return "api-key"
    host = request.client.host if request.client else "unknown"
    return f"host:{host}"


def _drop_expired(now: float) -> None:
    expired = [caller for caller, (start, _) in _windows.items() if now - start >= WINDOW_S]
    for caller in expired:
        del _windows[caller]


def _now() -> float:
    # Monotonic clock: not affected by system clock adjustments. Kept in one function
    # so that tests can move time forward.
    return time.monotonic()
