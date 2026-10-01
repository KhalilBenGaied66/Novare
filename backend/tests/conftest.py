"""Shared fixtures: every test runs against a temporary data directory, offline.

- no LLM (LLM_ENABLED=off), deterministic hash embeddings, SQLite file under tmp_path;
- a small corpus that mirrors the structure of the real one (front matter, client-scoped
  contracts) so tests do not depend on the demo corpus under data/.
"""

import json

import pytest

from app.core.config import Settings, get_settings, reset_settings
from app.core.logging import request_id_var

MINI_DOCS = {
    "procedure_sav.md": """---
titre: Procédure SAV
type: procedure
---
# Procédure SAV

## Priorités
P1 (critique) : arrêt total d'un équipement ou risque pour la sécurité des personnes.
P2 (majeure) : fonctionnement dégradé d'un équipement.
P3 (mineure) : gêne sans impact sur l'exploitation.

## Délais d'intervention
Les délais dépendent de la formule du contrat et de la priorité.

| Formule | P1 | P2 | P3 |
|---|---|---|---|
| Premium | 2 h | 4 h | 24 h |
| Confort | 4 h ouvrées | 8 h ouvrées | 48 h ouvrées |
| Essentiel | 8 h ouvrées | 24 h ouvrées | 72 h ouvrées |

## Escalade
En cas de doute sur la priorité, le dossier est transmis à un gestionnaire.
""",
    "grille_tarifs.md": """---
titre: Grille tarifaire 2026
type: tarif
---
# Grille tarifaire 2026

## Déplacements
Déplacement en journée : 89 € HT.
Majoration week-end : +35 % sur le déplacement et la main-d'œuvre.
Majoration de nuit (22 h - 6 h) : +50 %.

## Forfaits
Forfait diagnostic : 120 € HT, déduit si le devis est accepté.
""",
    "contrat_C-12.md": """---
titre: Contrat de maintenance — client C-12
type: contrat
client_id: C-12
---
# Contrat de maintenance — client C-12

Formule Confort, valable du 1er janvier 2025 au 31 décembre 2028.
Site couvert : chaufferie de l'immeuble Lumière, Lyon 3e.
Pièces et main-d'œuvre incluses en jours ouvrés. Astreinte week-end non incluse.
""",
    "contrat_C-34.md": """---
titre: Contrat de maintenance — client C-34
type: contrat
client_id: C-34
---
# Contrat de maintenance — client C-34

Formule Essentiel, échue le 30 juin 2026. Clause particulière : franchise de 250 € par sinistre.
""",
}

MINI_CLIENTS = {
    "clients": [
        {
            "client_id": "C-12",
            "nom": "Syndic Lumière",
            "formule": "Confort",
            "date_debut": "2025-01-01",
            "date_fin": "2028-12-31",
            "astreinte_weekend": False,
            "sites": ["Chaufferie immeuble Lumière, Lyon 3e"],
        },
        {
            "client_id": "C-27",
            "nom": "Clinique des Tilleuls",
            "formule": "Premium",
            "date_debut": "2024-03-01",
            "date_fin": "2029-02-28",
            "astreinte_weekend": True,
            "sites": ["Bloc technique, Villeurbanne"],
        },
        {
            "client_id": "C-34",
            "nom": "Garage Morel",
            "formule": "Essentiel",
            "date_debut": "2024-07-01",
            "date_fin": "2026-06-30",
            "astreinte_weekend": False,
            "sites": ["Atelier, Vénissieux"],
        },
    ],
    "sla_heures": {
        "Premium": {"P1": 2, "P2": 4, "P3": 24},
        "Confort": {"P1": 4, "P2": 8, "P3": 48},
        "Essentiel": {"P1": 8, "P2": 24, "P3": 72},
    },
}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Isolated settings for every test; returns the Settings object."""
    data_dir = tmp_path / "data"
    evals_dir = tmp_path / "evals"
    for directory in (data_dir / "sample_docs", data_dir / "reference", evals_dir):
        directory.mkdir(parents=True)
    overrides = {
        "DATA_DIR": str(data_dir),
        "EVALS_DIR": str(evals_dir),
        "DATABASE_URL": "",
        "RETRIEVAL_MODE": "hybrid",
        "EMBEDDING_BACKEND": "hash",
        "QDRANT_URL": "",
        "LLM_ENABLED": "off",
        "API_KEY": "",
        "RATE_LIMIT_PER_MINUTE": "0",
        "AUTO_INGEST": "false",
        "LOG_JSON": "false",
        "MIN_CONFIDENCE": "0.35",
        # LiteLLM otherwise downloads its price table from GitHub on first import.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    # A developer's .env must not change what the tests see: the file is not read, and
    # the values config.py already exported from it at import are removed.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    for name in ("MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield get_settings()
    reset_settings()
    # `handle_ask` called directly from a test sets the request id in the test's own
    # context; a server gives each request its own context.
    request_id_var.set("")


@pytest.fixture
def mini_corpus(env):
    """Write the small corpus and the client reference data; returns the docs directory."""
    for name, content in MINI_DOCS.items():
        (env.docs_dir / name).write_text(content, encoding="utf-8")
    (env.reference_dir / "clients.json").write_text(
        json.dumps(MINI_CLIENTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return env.docs_dir


@pytest.fixture
def db(env):
    from app.db.session import init_db

    init_db()
    return env


@pytest.fixture
def indexed(mini_corpus, db):
    """Corpus ingested (BM25 + hash vectors) and database initialised."""
    from app.ingestion.ingest import ingest_docs

    return ingest_docs()


@pytest.fixture
def api(indexed):
    """FastAPI test client with the lifespan started."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client
