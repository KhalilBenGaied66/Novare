"""HTTP client for the DossierOps API, used by the Streamlit interface.

Configuration comes from two environment variables read at call time: `API_URL`
(default `http://localhost:8000`) and `API_KEY` (optional, sent as `X-API-Key`).

Every failure is raised as `ApiError` carrying a French message that the interface
shows as is. The interface never sees a `requests` exception or a stack trace.
"""

import os
from urllib.parse import quote

import requests

DEFAULT_API_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Time allowed to open the connection: an API that is down must be reported quickly,
# whatever the time then allowed for the answer.
CONNECT_TIMEOUT_S = 5
# An agent request can chain several LLM calls, each with its own timeout and retries.
ASK_TIMEOUT_S = 120
ACTION_TIMEOUT_S = 30
# Readiness and metrics are fetched on every page refresh: fail fast.
STATUS_TIMEOUT_S = 5

# Field names used by the API schemas, as shown to the user when the API rejects a value.
FIELD_LABELS = {
    "q": "demande (3 à 2000 caractères)",
    "client_id": "identifiant client (format attendu : C-12)",
    "montant": "montant (nombre supérieur à 0)",
    "validator": "nom du valideur",
    "reason": "motif",
    "comment": "commentaire",
    "request_id": "identifiant de la demande",
}


class ApiError(Exception):
    """A failed API call. `str(error)` is a French message meant for the user."""


def ask(q: str, client_id: str | None, montant: float | None) -> dict:
    """Submit a request. `None` is sent as JSON `null`: the field is not provided."""
    payload = {"q": q, "client_id": client_id, "montant": montant}
    return _call("POST", f"{API_PREFIX}/ask", payload, ASK_TIMEOUT_S)


def approve(action_id: str, validator: str, priority: str | None = None) -> dict:
    """Approve a proposed ticket; `priority` replaces the proposed priority when given."""
    payload = {"validator": validator}
    if priority is not None:
        payload["priority"] = priority
    return _call("POST", _action_path(action_id, "approve"), payload, ACTION_TIMEOUT_S)


def reject(action_id: str, validator: str, reason: str) -> dict:
    payload = {"validator": validator, "reason": reason}
    return _call("POST", _action_path(action_id, "reject"), payload, ACTION_TIMEOUT_S)


def feedback(request_id: str, ok: bool, comment: str) -> dict:
    payload = {"request_id": request_id, "ok": ok, "comment": comment}
    return _call("POST", f"{API_PREFIX}/feedback", payload, ACTION_TIMEOUT_S)


def metrics() -> dict:
    return _call("GET", f"{API_PREFIX}/metrics", None, STATUS_TIMEOUT_S)


def ready() -> dict:
    """Readiness report `{"ready": bool, "checks": {...}}`.

    `/ready` answers 503 with the same report when a dependency is down. That is an
    answer to the question asked, not a failure, so it is returned instead of raised.
    """
    response = _send("GET", "/ready", None, STATUS_TIMEOUT_S)
    if response.status_code == 503:
        body = _json_body(response)
        if isinstance(body, dict) and "ready" in body:
            return body
    return _parse(response)


def _action_path(action_id: str, decision: str) -> str:
    # The id is quoted so that an unexpected value can never change the route.
    return f"{API_PREFIX}/actions/{quote(action_id, safe='')}/{decision}"


def _call(method: str, path: str, payload: dict | None, timeout: float) -> dict:
    return _parse(_send(method, path, payload, timeout))


def _send(method: str, path: str, payload: dict | None, timeout: float) -> requests.Response:
    base_url = os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        return requests.request(
            method,
            f"{base_url}{path}",
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT_S, timeout),
        )
    # ConnectTimeout is both a ConnectionError and a Timeout: an unreachable host must
    # read as "unreachable", so ConnectionError is tested first.
    except requests.ConnectionError as exc:
        raise ApiError(
            f"API injoignable ({base_url}). Vérifiez que le service est démarré "
            "et que la variable API_URL est correcte."
        ) from exc
    except requests.Timeout as exc:
        raise ApiError(
            "L'API n'a pas répondu dans le délai imparti. Réessayez dans un instant."
        ) from exc
    except requests.RequestException as exc:
        raise ApiError(
            f"La requête vers l'API a échoué. Vérifiez la variable API_URL ({base_url})."
        ) from exc


def _parse(response: requests.Response) -> dict:
    if response.status_code >= 400:
        raise ApiError(_error_message(response))
    body = _json_body(response)
    if not isinstance(body, dict):
        raise ApiError("Réponse illisible reçue de l'API.")
    return body


def _json_body(response: requests.Response) -> object:
    """Decoded JSON body, or None when the body is not JSON (proxy error page...)."""
    try:
        return response.json()
    except ValueError:
        return None


def _error_message(response: requests.Response) -> str:
    status = response.status_code
    body = _json_body(response)
    detail = body.get("detail") if isinstance(body, dict) else None

    if status == 401:
        return "Clé API invalide ou absente. Vérifiez la variable API_KEY de l'interface."
    if status == 422:
        fields = _invalid_fields(detail)
        if fields:
            return "Demande refusée par l'API. Champ(s) à corriger : " + ", ".join(fields) + "."
        return "Demande refusée par l'API : contenu invalide."
    if status == 429:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return f"Trop de requêtes. Réessayez dans {retry_after} secondes."
        return "Trop de requêtes. Réessayez dans un instant."
    # For the other statuses the API writes `detail` in French for the user
    # (unknown action, decision conflict, index not built...).
    if isinstance(detail, str) and detail:
        return detail
    return f"L'API a renvoyé une erreur (code {status})."


def _invalid_fields(detail: object) -> list[str]:
    """French labels of the fields named in a FastAPI validation error (422)."""
    if not isinstance(detail, list):
        return []
    labels: list[str] = []
    for error in detail:
        location = error.get("loc") if isinstance(error, dict) else None
        if not isinstance(location, list) or not location:
            continue
        name = str(location[-1])
        label = FIELD_LABELS.get(name, name)
        if label not in labels:
            labels.append(label)
    return labels
