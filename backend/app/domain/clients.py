"""Client reference data: contracts and SLA hours, read from `reference/clients.json`.

File format:
    {"clients": [{"client_id": "C-12", "nom": ..., "formule": "Confort",
                  "date_debut": "2025-01-01", "date_fin": "2028-12-31",
                  "astreinte_weekend": false, "sites": ["..."]}],
     "sla_heures": {"Confort": {"P1": 4, "P2": 8, "P3": 48}}}
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.core.config import get_settings, on_reset
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Client:
    client_id: str
    nom: str
    formule: str  # "Essentiel" | "Confort" | "Premium"
    date_debut: date
    date_fin: date  # last day of validity, included
    astreinte_weekend: bool  # 24/7 on-call included in the contract
    sites: tuple[str, ...]

    def is_active(self, today: date | None = None) -> bool:
        """True when the contract is in force on `today` (default: the current date)."""
        today = today or date.today()
        return self.date_debut <= today <= self.date_fin


class ClientRepository:
    def __init__(self, clients: list[Client], sla_hours: dict[str, dict[str, int]]) -> None:
        self._clients = {_key(client.client_id): client for client in clients}
        self._sla_hours = {
            _key(formule): {_key(priority): hours for priority, hours in by_priority.items()}
            for formule, by_priority in sla_hours.items()
        }

    def get(self, client_id: str | None) -> Client | None:
        if not client_id:
            return None
        return self._clients.get(_key(client_id))

    def all(self) -> list[Client]:
        return list(self._clients.values())

    def sla_hours(self, formule: str, priority: str) -> int | None:
        """Contractual intervention delay in hours; None when the pair is not defined."""
        return self._sla_hours.get(_key(formule), {}).get(_key(priority))


def _key(value: str) -> str:
    """Lookups ignore case and surrounding spaces ("c-12 " finds "C-12")."""
    return value.strip().casefold()


def _parse_client(item: dict) -> Client:
    return Client(
        client_id=item["client_id"],
        nom=item["nom"],
        formule=item["formule"],
        date_debut=date.fromisoformat(item["date_debut"]),
        date_fin=date.fromisoformat(item["date_fin"]),
        astreinte_weekend=bool(item["astreinte_weekend"]),
        sites=tuple(item.get("sites", [])),
    )


def load_clients(path: Path) -> ClientRepository:
    """Read the reference file; a missing file gives an empty repository.

    A malformed file raises: silently losing the clients would change how requests
    are routed (no dispute could be automated any more).
    """
    if not path.exists():
        logger.warning("clients_file_missing", extra={"path": str(path)})
        return ClientRepository([], {})
    data = json.loads(path.read_text(encoding="utf-8"))
    clients = [_parse_client(item) for item in data.get("clients", [])]
    return ClientRepository(clients, data.get("sla_heures", {}))


_repository: ClientRepository | None = None


def get_client_repository() -> ClientRepository:
    """Repository for `settings.reference_dir / "clients.json"`, loaded once."""
    global _repository
    if _repository is None:
        _repository = load_clients(get_settings().reference_dir / "clients.json")
    return _repository


@on_reset
def _reset_repository() -> None:
    global _repository
    _repository = None
