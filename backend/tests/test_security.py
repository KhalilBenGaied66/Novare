"""API key and rate limit dependencies, exercised through a minimal FastAPI app."""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core import security
from app.core.config import reset_settings


def make_client(host: str = "10.0.0.1", *, auth: bool = True) -> TestClient:
    dependencies = [Depends(security.rate_limit)]
    if auth:
        dependencies.insert(0, Depends(security.require_api_key))
    app = FastAPI()

    @app.get("/ping", dependencies=dependencies)
    def ping() -> dict:
        return {"pong": True}

    return TestClient(app, client=(host, 50000))


@pytest.fixture
def configure(monkeypatch):
    """Set API_KEY / RATE_LIMIT_PER_MINUTE for the test and rebuild the settings."""

    def _configure(api_key: str = "", limit: int = 0) -> None:
        monkeypatch.setenv("API_KEY", api_key)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", str(limit))
        reset_settings()

    return _configure


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(security, "_now", fake)
    return fake


# --- API key ---------------------------------------------------------------------


def test_no_key_configured_allows_everyone(configure):
    configure(api_key="")
    client = make_client()
    assert client.get("/ping").status_code == 200
    assert client.get("/ping", headers={"X-API-Key": "anything"}).status_code == 200


def test_missing_key_is_rejected(configure):
    configure(api_key="s3cret-key")
    response = make_client().get("/ping")
    assert response.status_code == 401
    assert response.json() == {"detail": "Clé API invalide ou absente"}
    assert response.headers["WWW-Authenticate"] == "APIKey"


@pytest.mark.parametrize("wrong", ["other", "s3cret-ke", "s3cret-key ", "S3CRET-KEY", ""])
def test_wrong_key_is_rejected(configure, wrong):
    configure(api_key="s3cret-key")
    response = make_client().get("/ping", headers={"X-API-Key": wrong})
    assert response.status_code == 401
    assert response.json() == {"detail": "Clé API invalide ou absente"}


def test_right_key_is_accepted(configure):
    configure(api_key="s3cret-key")
    response = make_client().get("/ping", headers={"X-API-Key": "s3cret-key"})
    assert response.status_code == 200
    assert response.json() == {"pong": True}


def test_non_ascii_key_is_a_401_not_a_server_error(configure):
    # hmac.compare_digest raises TypeError on non-ASCII str; the comparison is on bytes.
    configure(api_key="s3cret-key")
    response = make_client().get("/ping", headers={"X-API-Key": "clé".encode()})
    assert response.status_code == 401


# --- Rate limit ------------------------------------------------------------------


def test_limit_zero_disables_rate_limiting(configure, clock):
    configure(limit=0)
    client = make_client()
    assert [client.get("/ping").status_code for _ in range(20)] == [200] * 20


def test_requests_over_the_limit_get_429_with_retry_after(configure, clock):
    configure(limit=2)
    client = make_client()
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200

    clock.now += 20
    response = client.get("/ping")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "40"
    assert response.json() == {"detail": "Trop de requêtes : réessayez dans 40 s"}


def test_window_restarts_after_sixty_seconds(configure, clock):
    configure(limit=2)
    client = make_client()
    assert [client.get("/ping").status_code for _ in range(3)] == [200, 200, 429]

    clock.now += 59.5
    response = client.get("/ping")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"

    clock.now += 0.5
    assert [client.get("/ping").status_code for _ in range(3)] == [200, 200, 429]


def test_each_client_host_has_its_own_counter(configure, clock):
    configure(limit=1)
    first, second = make_client("10.0.0.1"), make_client("10.0.0.2")
    assert first.get("/ping").status_code == 200
    assert first.get("/ping").status_code == 429
    assert second.get("/ping").status_code == 200


def test_valid_api_key_shares_one_counter_across_hosts(configure, clock):
    configure(api_key="s3cret-key", limit=2)
    headers = {"X-API-Key": "s3cret-key"}
    assert make_client("10.0.0.1").get("/ping", headers=headers).status_code == 200
    assert make_client("10.0.0.2").get("/ping", headers=headers).status_code == 200
    assert make_client("10.0.0.3").get("/ping", headers=headers).status_code == 429


def test_made_up_keys_do_not_open_new_counters(configure, clock):
    # Rate limit alone, as if it ran before authentication: the header is not trusted.
    configure(api_key="s3cret-key", limit=2)
    client = make_client(auth=False)
    statuses = [
        client.get("/ping", headers={"X-API-Key": f"guess-{i}"}).status_code for i in range(3)
    ]
    assert statuses == [200, 200, 429]


def test_rejected_keys_do_not_use_up_the_quota(configure, clock):
    configure(api_key="s3cret-key", limit=1)
    client = make_client()
    assert [client.get("/ping").status_code for _ in range(3)] == [401, 401, 401]
    assert client.get("/ping", headers={"X-API-Key": "s3cret-key"}).status_code == 200


def test_expired_windows_are_purged_when_the_table_is_full(configure, clock, monkeypatch):
    monkeypatch.setattr(security, "_MAX_TRACKED_CALLERS", 2)
    configure(limit=5)
    for n in range(1, 4):
        assert make_client(f"10.0.0.{n}").get("/ping").status_code == 200
    assert len(security._windows) == 3

    clock.now += 61
    assert make_client("10.0.0.9").get("/ping").status_code == 200
    assert list(security._windows) == ["host:10.0.0.9"]


def test_counters_are_cleared_with_the_settings(configure, clock):
    configure(limit=1)
    client = make_client()
    assert [client.get("/ping").status_code for _ in range(2)] == [200, 429]
    reset_settings()
    assert client.get("/ping").status_code == 200
