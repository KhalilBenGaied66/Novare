"""`api_client`: what is sent to the API and how failures become French messages."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import api_client
import pytest
import requests


class FakeResponse:
    def __init__(self, status_code: int = 200, body=None, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


class FakeTransport:
    """Replaces `requests.request`: records the call, then answers or raises."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.response = FakeResponse(200, {})
        self.error: Exception | None = None

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def transport(monkeypatch) -> FakeTransport:
    # The developer's shell or `.env` may define these: start from a known state.
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    fake = FakeTransport()
    monkeypatch.setattr(api_client.requests, "request", fake)
    return fake


def error_message(call) -> str:
    with pytest.raises(api_client.ApiError) as raised:
        call()
    return str(raised.value)


# --- Requests ------------------------------------------------------------------------


def test_ask_posts_the_three_fields_and_keeps_none_as_null(transport):
    transport.response = FakeResponse(200, {"request_id": "req-1", "route": "rag"})

    result = api_client.ask("Quel est le tarif du déplacement ?", None, None)

    assert result == {"request_id": "req-1", "route": "rag"}
    assert transport.calls == [
        {
            "method": "POST",
            "url": "http://localhost:8000/api/v1/ask",
            "json": {"q": "Quel est le tarif du déplacement ?", "client_id": None, "montant": None},
            "headers": {},
            "timeout": (api_client.CONNECT_TIMEOUT_S, api_client.ASK_TIMEOUT_S),
        }
    ]


def test_ask_sends_client_and_amount_when_given(transport):
    api_client.ask("Je conteste ma facture.", "C-12", 120.5)

    assert transport.calls[0]["json"] == {
        "q": "Je conteste ma facture.",
        "client_id": "C-12",
        "montant": 120.5,
    }


def test_api_url_and_api_key_are_read_from_the_environment_at_call_time(transport, monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8000/")
    monkeypatch.setenv("API_KEY", "clef-de-test")

    api_client.metrics()

    assert transport.calls[0]["url"] == "http://api:8000/api/v1/metrics"
    assert transport.calls[0]["headers"] == {"X-API-Key": "clef-de-test"}


def test_no_api_key_header_when_the_key_is_empty(transport, monkeypatch):
    monkeypatch.setenv("API_KEY", "")

    api_client.metrics()

    assert transport.calls[0]["headers"] == {}


def test_approve_and_reject_post_the_decision_on_the_action(transport):
    api_client.approve("9f1c2ab4", "Marie Durand")
    api_client.reject("9f1c2ab4", "Marie Durand", "Doublon")

    approve, reject = transport.calls
    assert (approve["method"], approve["url"]) == (
        "POST",
        "http://localhost:8000/api/v1/actions/9f1c2ab4/approve",
    )
    assert approve["json"] == {"validator": "Marie Durand"}
    assert (reject["method"], reject["url"]) == (
        "POST",
        "http://localhost:8000/api/v1/actions/9f1c2ab4/reject",
    )
    assert reject["json"] == {"validator": "Marie Durand", "reason": "Doublon"}


def test_action_id_cannot_change_the_route(transport):
    api_client.approve("../tickets?x=1", "Marie Durand")

    assert transport.calls[0]["url"] == (
        "http://localhost:8000/api/v1/actions/..%2Ftickets%3Fx%3D1/approve"
    )


def test_feedback_posts_request_id_verdict_and_comment(transport):
    api_client.feedback("req-1", False, "Mauvais délai cité")

    call = transport.calls[0]
    assert (call["method"], call["url"]) == ("POST", "http://localhost:8000/api/v1/feedback")
    assert call["json"] == {"request_id": "req-1", "ok": False, "comment": "Mauvais délai cité"}


def test_metrics_is_a_get_without_body(transport):
    transport.response = FakeResponse(200, {"requests": 3})

    assert api_client.metrics() == {"requests": 3}
    call = transport.calls[0]
    assert (call["method"], call["url"]) == ("GET", "http://localhost:8000/api/v1/metrics")
    assert call["json"] is None
    assert call["timeout"] == (api_client.CONNECT_TIMEOUT_S, api_client.STATUS_TIMEOUT_S)


# --- Readiness -----------------------------------------------------------------------


def test_ready_calls_the_root_route(transport):
    transport.response = FakeResponse(200, {"ready": True, "checks": {"database": "ok"}})

    assert api_client.ready() == {"ready": True, "checks": {"database": "ok"}}
    assert transport.calls[0]["url"] == "http://localhost:8000/ready"


def test_ready_returns_the_report_of_an_api_that_is_not_ready(transport):
    report = {"ready": False, "checks": {"database": "ok", "index": "absent"}}
    transport.response = FakeResponse(503, report)

    assert api_client.ready() == report


def test_ready_raises_on_a_503_without_report(transport):
    transport.response = FakeResponse(503, None)

    assert error_message(api_client.ready) == "L'API a renvoyé une erreur (code 503)."


# --- Failures ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("refused"), requests.ConnectTimeout("no route")]
)
def test_unreachable_api(transport, monkeypatch, error):
    monkeypatch.setenv("API_URL", "http://api:8000")
    transport.error = error

    assert error_message(lambda: api_client.ask("Bonjour", None, None)) == (
        "API injoignable (http://api:8000). Vérifiez que le service est démarré "
        "et que la variable API_URL est correcte."
    )


def test_slow_api(transport):
    transport.error = requests.ReadTimeout("read timed out")

    assert error_message(api_client.metrics) == (
        "L'API n'a pas répondu dans le délai imparti. Réessayez dans un instant."
    )


def test_other_transport_failure_does_not_leak_technical_details(transport):
    transport.error = requests.TooManyRedirects("Exceeded 30 redirects at http://internal")

    assert error_message(api_client.metrics) == (
        "La requête vers l'API a échoué. Vérifiez la variable API_URL (http://localhost:8000)."
    )


def test_invalid_api_key(transport):
    transport.response = FakeResponse(401, {"detail": "Clé API invalide ou absente"})

    assert error_message(api_client.metrics) == (
        "Clé API invalide ou absente. Vérifiez la variable API_KEY de l'interface."
    )


def test_validation_error_names_the_fields_in_french(transport):
    detail = [
        {"type": "string_pattern_mismatch", "loc": ["body", "client_id"], "msg": "String should"},
        {"type": "greater_than", "loc": ["body", "montant"], "msg": "Input should be greater"},
        {"type": "string_pattern_mismatch", "loc": ["body", "client_id"], "msg": "duplicate"},
    ]
    transport.response = FakeResponse(422, {"detail": detail})

    assert error_message(lambda: api_client.ask("Bonjour", "12", -1.0)) == (
        "Demande refusée par l'API. Champ(s) à corriger : "
        "identifiant client (format attendu : C-12), montant (nombre supérieur à 0)."
    )


@pytest.mark.parametrize("body", [{"detail": "texte inattendu"}, {"detail": [{"loc": []}]}, None])
def test_validation_error_without_usable_field_names(transport, body):
    transport.response = FakeResponse(422, body)

    assert error_message(api_client.metrics) == "Demande refusée par l'API : contenu invalide."


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "42"}, "Trop de requêtes. Réessayez dans 42 secondes."),
        ({}, "Trop de requêtes. Réessayez dans un instant."),
    ],
)
def test_rate_limit(transport, headers, expected):
    transport.response = FakeResponse(429, {"detail": "Trop de requêtes"}, headers)

    assert error_message(api_client.metrics) == expected


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (404, "Action introuvable"),
        (409, "Cette action a déjà été refusée"),
        (503, "Index documentaire absent : lancez POST /api/v1/ingest"),
    ],
)
def test_french_detail_written_by_the_api_is_shown_as_is(transport, status, detail):
    transport.response = FakeResponse(status, {"detail": detail})

    assert error_message(lambda: api_client.approve("9f1c2ab4", "Marie Durand")) == detail


@pytest.mark.parametrize("body", [None, {}, {"detail": {"code": 12}}, ["inattendu"]])
def test_error_without_readable_detail_gives_a_generic_message(transport, body):
    transport.response = FakeResponse(500, body)

    assert error_message(api_client.metrics) == "L'API a renvoyé une erreur (code 500)."


@pytest.mark.parametrize("body", [None, ["liste"], "texte"])
def test_success_with_an_unexpected_body(transport, body):
    transport.response = FakeResponse(200, body)

    assert error_message(api_client.metrics) == "Réponse illisible reçue de l'API."


# --- On the wire ---------------------------------------------------------------------


class RecordingHandler(BaseHTTPRequestHandler):
    """Local HTTP endpoint that records what really arrives, then answers 200."""

    received: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        RecordingHandler.received.append(
            {
                "path": self.path,
                "content_type": self.headers["Content-Type"],
                "api_key": self.headers["X-API-Key"],
                "body": self.rfile.read(length).decode("utf-8"),
            }
        )
        payload = json.dumps({"request_id": "req-1", "answer": "Reçu, délai de 4 h ouvrées."})
        encoded = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args) -> None:
        """Silence the access log that the base class prints on stderr."""


def test_ask_over_real_http_sends_json_null_and_reads_utf8(monkeypatch):
    RecordingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), RecordingHandler)
    # A short poll interval lets `shutdown()` return at once instead of after 0.5 s.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_KEY", "clef-de-test")
    try:
        result = api_client.ask("La chaudière est en panne.", None, None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result == {"request_id": "req-1", "answer": "Reçu, délai de 4 h ouvrées."}
    (received,) = RecordingHandler.received
    assert received["path"] == "/api/v1/ask"
    assert received["content_type"] == "application/json"
    assert received["api_key"] == "clef-de-test"
    assert json.loads(received["body"]) == {
        "q": "La chaudière est en panne.",
        "client_id": None,
        "montant": None,
    }
    assert '"client_id": null' in received["body"]
    assert '"montant": null' in received["body"]
