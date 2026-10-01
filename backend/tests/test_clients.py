"""Client reference data: lookup, contract validity, SLA hours, missing or broken file."""

import dataclasses
import json
import logging
from datetime import date

import pytest

from app.core.config import reset_settings
from app.domain import clients
from app.domain.clients import Client, ClientRepository, get_client_repository


@pytest.fixture
def repo(mini_corpus) -> ClientRepository:
    return get_client_repository()


def test_get_returns_the_contract_as_written_in_the_file(repo):
    assert repo.get("C-12") == Client(
        client_id="C-12",
        nom="Syndic Lumière",
        formule="Confort",
        date_debut=date(2025, 1, 1),
        date_fin=date(2028, 12, 31),
        astreinte_weekend=False,
        sites=("Chaufferie immeuble Lumière, Lyon 3e",),
    )
    assert repo.get("C-27").astreinte_weekend is True
    assert repo.get("C-27").formule == "Premium"


@pytest.mark.parametrize("client_id", ["c-12", "C-12", " C-12 ", "c-12\n"])
def test_get_ignores_case_and_surrounding_spaces(repo, client_id):
    assert repo.get(client_id).client_id == "C-12"


@pytest.mark.parametrize(
    "client_id", [None, "", "   ", "C-99", "C-1", "C-123", "12", "Syndic Lumière"]
)
def test_get_unknown_client_returns_none(repo, client_id):
    # "C-1" and "C-123" would match "C-12" with a prefix or substring comparison.
    assert repo.get(client_id) is None


def test_all_lists_every_client_in_file_order(repo):
    assert [client.client_id for client in repo.all()] == ["C-12", "C-27", "C-34"]


@pytest.mark.parametrize(
    ("client_id", "today", "expected"),
    [
        ("C-34", date(2024, 6, 30), False),  # the day before the contract starts
        ("C-34", date(2024, 7, 1), True),  # first day
        ("C-34", date(2026, 6, 30), True),  # last day is included
        ("C-34", date(2026, 7, 1), False),  # expired
        ("C-12", date(2026, 10, 1), True),
        ("C-12", date(2029, 1, 1), False),
    ],
)
def test_is_active_on_a_given_day(repo, client_id, today, expected):
    assert repo.get(client_id).is_active(today) is expected


def test_is_active_defaults_to_the_current_date(repo, monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 1)

    monkeypatch.setattr(clients, "date", FrozenDate)
    assert repo.get("C-12").is_active() is True
    assert repo.get("C-34").is_active() is False  # ended on 2026-06-30


@pytest.mark.parametrize(
    ("formule", "p1", "p2", "p3"),
    [("Premium", 2, 4, 24), ("Confort", 4, 8, 48), ("Essentiel", 8, 24, 72)],
)
def test_sla_hours_for_every_formule_and_priority(repo, formule, p1, p2, p3):
    assert repo.sla_hours(formule, "P1") == p1
    assert repo.sla_hours(formule, "P2") == p2
    assert repo.sla_hours(formule, "P3") == p3


@pytest.mark.parametrize(
    ("formule", "priority", "expected"),
    [
        ("confort", "p1", 4),  # case is ignored
        (" Confort ", "P1 ", 4),
        ("Confort", "P4", None),
        ("Platine", "P1", None),
        ("", "", None),
    ],
)
def test_sla_hours_lookup_is_tolerant_but_never_guesses(repo, formule, priority, expected):
    assert repo.sla_hours(formule, priority) == expected


def test_client_is_immutable(repo):
    with pytest.raises(dataclasses.FrozenInstanceError):
        repo.get("C-12").formule = "Premium"


def test_missing_file_gives_an_empty_repository_and_a_warning(env, caplog):
    assert not (env.reference_dir / "clients.json").exists()
    with caplog.at_level(logging.WARNING, logger="app.domain.clients"):
        repo = get_client_repository()
    assert repo.all() == []
    assert repo.get("C-12") is None
    assert repo.sla_hours("Confort", "P1") is None
    assert [record.getMessage() for record in caplog.records] == ["clients_file_missing"]


def test_malformed_file_raises_instead_of_hiding_the_clients(env):
    (env.reference_dir / "clients.json").write_text('{"clients": [', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        get_client_repository()


def test_client_without_a_required_field_raises(env):
    broken = {"clients": [{"client_id": "C-77", "nom": "Sans dates", "formule": "Confort"}]}
    (env.reference_dir / "clients.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(KeyError, match="date_debut"):
        get_client_repository()


def test_repository_is_cached_until_the_settings_are_reset(mini_corpus, env):
    first = get_client_repository()
    assert get_client_repository() is first

    path = env.reference_dir / "clients.json"
    updated = json.loads(path.read_text(encoding="utf-8"))
    updated["clients"].append(
        {
            "client_id": "C-45",
            "nom": "Résidence des Érables",
            "formule": "Confort",
            "date_debut": "2025-09-01",
            "date_fin": "2029-08-31",
            "astreinte_weekend": False,
            "sites": ["Sous-station, Bron"],
        }
    )
    path.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
    assert get_client_repository().get("C-45") is None  # still the cached repository

    reset_settings()
    assert get_client_repository().get("C-45").nom == "Résidence des Érables"
