"""Database layer on SQLite: tickets, proposed actions, request log, feedback, engine."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from app.core import schemas
from app.core.config import reset_settings
from app.db import repositories as repo
from app.db.models import Base, FeedbackRow, RequestLogRow, TicketRow
from app.db.session import get_engine, init_db, session_scope


def make_ticket(session, **overrides) -> TicketRow:
    fields = {
        "client_id": "C-12",
        "subject": "Litige facturation 120.00 € — C-12",
        "body": "Je conteste la facture de 120 € reçue en septembre.",
        "kind": "litige_standard",
        "source": "automation",
        "created_by": "system:RG-03",
    }
    fields.update(overrides)
    return repo.create_ticket(session, **fields)


def make_request(session, request_id: str, route: str = "rag", **overrides) -> RequestLogRow:
    fields = {"request_id": request_id, "route": route, "mode": "extractive", "rule": "DEFAULT"}
    fields.update(overrides)
    return repo.log_request(session, **fields)


# --- Tickets ---------------------------------------------------------------


def test_ticket_ids_are_sequential_and_zero_padded(db):
    with session_scope() as session:
        first = make_ticket(session)
        second = make_ticket(session)
        assert (first.ticket_id, second.ticket_id) == ("T-000001", "T-000002")


def test_ticket_id_keeps_all_digits_beyond_six():
    assert TicketRow(id=12).ticket_id == "T-000012"
    assert TicketRow(id=1234567).ticket_id == "T-1234567"


def test_ticket_defaults_and_stored_fields(db):
    before = datetime.now(UTC)
    with session_scope() as session:
        created = make_ticket(session, request_id="req-1", dedupe_key="abc", priority=None)
    with session_scope() as session:
        ticket = repo.get_ticket(session, created.ticket_id)
        assert ticket.status == "open"
        assert ticket.client_id == "C-12"
        assert ticket.subject == "Litige facturation 120.00 € — C-12"
        assert ticket.body == "Je conteste la facture de 120 € reçue en septembre."
        assert (ticket.kind, ticket.source, ticket.created_by) == (
            "litige_standard",
            "automation",
            "system:RG-03",
        )
        assert (ticket.request_id, ticket.dedupe_key, ticket.priority) == ("req-1", "abc", None)
        assert ticket.created_at.tzinfo is not None
        assert before <= ticket.created_at <= datetime.now(UTC)


@pytest.mark.parametrize("ticket_id", ["T-000001", "T-1", "t-000001", " T-000001 "])
def test_get_ticket_accepts_the_public_identifier(db, ticket_id):
    with session_scope() as session:
        make_ticket(session)
        assert repo.get_ticket(session, ticket_id).id == 1


@pytest.mark.parametrize(
    "ticket_id",
    [
        "T-000002",
        "T-999999",
        "1",
        "000001",
        "T-",
        "T-abc",
        "T-1 OR 1=1",
        "",
        "T-99999999999999999999",
    ],
)
def test_get_ticket_unknown_or_malformed_returns_none(db, ticket_id):
    with session_scope() as session:
        make_ticket(session)
        assert repo.get_ticket(session, ticket_id) is None


def test_list_tickets_newest_first_with_limit(db):
    with session_scope() as session:
        for number in range(1, 6):
            make_ticket(session, subject=f"Ticket {number}")
        assert [t.subject for t in repo.list_tickets(session)] == [
            "Ticket 5",
            "Ticket 4",
            "Ticket 3",
            "Ticket 2",
            "Ticket 1",
        ]
        assert [t.ticket_id for t in repo.list_tickets(session, limit=2)] == [
            "T-000005",
            "T-000004",
        ]


def test_ticket_number_is_never_reused_after_a_deletion(db):
    with session_scope() as session:
        session.delete(make_ticket(session))
    with session_scope() as session:
        assert make_ticket(session).ticket_id == "T-000002"


def test_to_ticket_maps_every_field(db):
    with session_scope() as session:
        row = make_ticket(
            session,
            kind="intervention",
            source="agent",
            priority="P1",
            subject="Chaudière à l'arrêt — immeuble Lumière",
            body="Arrêt total du chauffage signalé par le syndic.",
            created_by="Camille Durand",
            request_id="req-7",
        )
    ticket = repo.to_ticket(row)  # the row stays readable after the transaction
    assert isinstance(ticket, schemas.Ticket)
    assert ticket.model_dump(exclude={"created_at"}) == {
        "ticket_id": "T-000001",
        "client_id": "C-12",
        "subject": "Chaudière à l'arrêt — immeuble Lumière",
        "body": "Arrêt total du chauffage signalé par le syndic.",
        "kind": "intervention",
        "priority": "P1",
        "source": "agent",
        "status": "open",
        "created_by": "Camille Durand",
        "request_id": "req-7",
    }
    assert ticket.created_at == row.created_at
    assert ticket.model_dump(mode="json")["created_at"].endswith(("Z", "+00:00"))


# --- Duplicate lookup ------------------------------------------------------


def test_find_recent_ticket_inside_and_outside_the_window(db):
    now = datetime.now(UTC)
    with session_scope() as session:
        ticket = make_ticket(session, dedupe_key="k1")
        ticket.created_at = now - timedelta(days=3)

    with session_scope() as session:
        assert (
            repo.find_recent_ticket(session, "k1", now - timedelta(days=7)).ticket_id == "T-000001"
        )
        assert (
            repo.find_recent_ticket(session, "k1", now - timedelta(days=3)).ticket_id == "T-000001"
        )
        assert repo.find_recent_ticket(session, "k1", now - timedelta(days=2)) is None
        assert repo.find_recent_ticket(session, "autre-cle", now - timedelta(days=7)) is None


def test_find_recent_ticket_returns_the_latest_match(db):
    now = datetime.now(UTC)
    with session_scope() as session:
        make_ticket(session, dedupe_key="k1")
        make_ticket(session, dedupe_key="k2")
        make_ticket(session, dedupe_key="k1")
        make_ticket(session, dedupe_key=None)
        assert (
            repo.find_recent_ticket(session, "k1", now - timedelta(days=7)).ticket_id == "T-000003"
        )
        assert (
            repo.find_recent_ticket(session, "k2", now - timedelta(days=7)).ticket_id == "T-000002"
        )


def test_find_recent_ticket_accepts_any_timezone_and_naive_utc(db):
    now = datetime.now(UTC)
    paris_summer = timezone(timedelta(hours=2))
    with session_scope() as session:
        make_ticket(session, dedupe_key="k1")
        an_hour_ago = now - timedelta(hours=1)
        in_an_hour = now + timedelta(hours=1)
        assert repo.find_recent_ticket(session, "k1", an_hour_ago.astimezone(paris_summer))
        assert repo.find_recent_ticket(session, "k1", in_an_hour.astimezone(paris_summer)) is None
        assert repo.find_recent_ticket(session, "k1", an_hour_ago.replace(tzinfo=None))
        assert repo.find_recent_ticket(session, "k1", in_an_hour.replace(tzinfo=None)) is None


def test_datetimes_are_read_back_as_the_same_instant_in_utc(db):
    paris_noon = datetime(2026, 7, 14, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    with session_scope() as session:
        make_ticket(session).created_at = paris_noon
    with session_scope() as session:
        stored = repo.get_ticket(session, "T-000001").created_at
        assert stored == datetime(2026, 7, 14, 10, 0, tzinfo=UTC)
        assert stored.utcoffset() == timedelta(0)


# --- Transactions ----------------------------------------------------------


def test_session_scope_commits_on_success(db):
    with session_scope() as session:
        make_ticket(session)
    with session_scope() as session:
        assert len(repo.list_tickets(session)) == 1


def test_session_scope_rolls_back_on_error(db):
    with pytest.raises(RuntimeError, match="panne"), session_scope() as session:
        make_ticket(session)
        raise RuntimeError("panne simulée")
    with session_scope() as session:
        assert repo.list_tickets(session) == []


def test_init_db_is_idempotent_and_keeps_the_data(db):
    with session_scope() as session:
        make_ticket(session)
    init_db()
    init_db()
    with session_scope() as session:
        assert [t.ticket_id for t in repo.list_tickets(session)] == ["T-000001"]


# --- Proposed actions ------------------------------------------------------

PROPOSAL = {
    "subject": "Panne chaudière — immeuble Lumière",
    "summary": "Arrêt total du chauffage, intervention P1 à planifier.",
    "priority": "P1",
    "client_id": "C-12",
}


def test_action_is_created_pending(db):
    with session_scope() as session:
        action = repo.create_action(
            session, request_id="req-1", type="create_ticket", payload=PROPOSAL
        )
        action_id = action.id
    assert len(action_id) == 32
    assert int(action_id, 16) >= 0  # uuid4 hex

    with session_scope() as session:
        stored = repo.get_action(session, action_id)
        assert stored.status == "pending"
        assert stored.request_id == "req-1"
        assert stored.type == "create_ticket"
        assert stored.payload == PROPOSAL  # JSON round trip keeps accents and structure
        assert stored.created_at.tzinfo is not None
        assert (stored.decided_at, stored.decided_by, stored.ticket_id) == (None, None, None)
        assert stored.reason == ""
        assert repo.get_action(session, "inconnu") is None


def test_action_approval_lifecycle(db):
    with session_scope() as session:
        action_id = repo.create_action(
            session, request_id="req-1", type="create_ticket", payload=PROPOSAL
        ).id
        assert [a.id for a in repo.list_actions(session, status="pending")] == [action_id]

    with session_scope() as session:
        action = repo.get_action(session, action_id)
        ticket = make_ticket(session, kind="intervention", source="agent", created_by="Camille")
        decided = repo.decide_action(
            session, action, status="approved", decided_by="Camille", ticket_id=ticket.ticket_id
        )
        assert decided is action

    with session_scope() as session:
        stored = repo.get_action(session, action_id)
        assert stored.status == "approved"
        assert stored.decided_by == "Camille"
        assert stored.ticket_id == "T-000001"
        assert stored.reason == ""
        assert stored.created_at <= stored.decided_at <= datetime.now(UTC)
        assert repo.list_actions(session, status="pending") == []
        assert [a.id for a in repo.list_actions(session, status="approved")] == [action_id]
        assert repo.to_action(stored) == schemas.ProposedAction(
            action_id=action_id,
            type="create_ticket",
            status="approved",
            payload=PROPOSAL,
            ticket_id="T-000001",
        )


def test_action_rejection_keeps_the_reason_and_creates_no_ticket(db):
    with session_scope() as session:
        action = repo.create_action(
            session, request_id="req-2", type="create_ticket", payload=PROPOSAL
        )
        repo.decide_action(
            session, action, status="rejected", decided_by="Léa", reason="Doublon du T-000004"
        )
    with session_scope() as session:
        stored = repo.get_action(session, action.id)
        assert (stored.status, stored.decided_by, stored.reason, stored.ticket_id) == (
            "rejected",
            "Léa",
            "Doublon du T-000004",
            None,
        )
        assert repo.list_tickets(session) == []


@pytest.mark.parametrize("status", ["pending", "done", "", "APPROVED"])
def test_decide_action_refuses_an_unknown_status(db, status):
    with session_scope() as session:
        action = repo.create_action(
            session, request_id="req-3", type="create_ticket", payload=PROPOSAL
        )
        with pytest.raises(ValueError, match="status"):
            repo.decide_action(session, action, status=status, decided_by="Léa")
        assert action.status == "pending"
        assert action.decided_at is None


def test_list_actions_newest_first_filter_and_limit(db):
    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    with session_scope() as session:
        ids = []
        for offset in range(4):
            action = repo.create_action(
                session, request_id=f"req-{offset}", type="create_ticket", payload=PROPOSAL
            )
            action.created_at = start + timedelta(hours=offset)
            ids.append(action.id)
        repo.decide_action(
            session, repo.get_action(session, ids[3]), status="rejected", decided_by="Léa"
        )

    with session_scope() as session:
        assert [a.id for a in repo.list_actions(session)] == [ids[3], ids[2], ids[1], ids[0]]
        assert [a.id for a in repo.list_actions(session, status="pending")] == [
            ids[2],
            ids[1],
            ids[0],
        ]
        assert [a.id for a in repo.list_actions(session, status="pending", limit=2)] == [
            ids[2],
            ids[1],
        ]
        assert [a.id for a in repo.list_actions(session, status="rejected")] == [ids[3]]


# --- Request log -----------------------------------------------------------


def test_log_request_stores_every_field(db):
    fields = {
        "request_id": "req-42",
        "route": "rag",
        "rule": "DEFAULT",
        "mode": "llm",
        "q_masked": "Quel est le tarif week-end ? Contact : [EMAIL]",
        "client_id": "C-12",
        "confidence": 0.82,
        "latency_ms": 640,
        "llm_calls": 1,
        "prompt_tokens": 812,
        "completion_tokens": 96,
        "cost_eur": 0.000161,
        "model": "mistral/mistral-small-latest",
        "retrieval_mode": "hybrid",
        "pii_types": ["EMAIL"],
        "needs_validation": True,
        "error": None,
    }
    with session_scope() as session:
        repo.log_request(session, **fields)
    with session_scope() as session:
        row = repo.get_request(session, "req-42")
        assert {name: getattr(row, name) for name in fields} == fields
        assert row.ts.tzinfo is not None
        assert repo.get_request(session, "inconnu") is None


def test_log_request_defaults_for_an_escalation_or_an_error(db):
    with session_scope() as session:
        repo.log_request(
            session, request_id="req-err", route="human", mode="none", error="ZeroDivisionError"
        )
    with session_scope() as session:
        row = repo.get_request(session, "req-err")
        assert (row.rule, row.q_masked, row.retrieval_mode) == ("", "", "none")
        assert (row.llm_calls, row.prompt_tokens, row.completion_tokens, row.latency_ms) == (
            0,
            0,
            0,
            0,
        )
        assert row.cost_eur == 0.0
        assert row.pii_types == []
        assert (row.client_id, row.confidence, row.model) == (None, None, None)
        assert row.needs_validation is True
        assert row.error == "ZeroDivisionError"


def test_log_request_rejects_an_unknown_field(db):
    with session_scope() as session, pytest.raises(TypeError, match="question"):
        repo.log_request(session, request_id="r", route="rag", mode="llm", question="texte brut")


def test_request_id_is_unique(db):
    with pytest.raises(IntegrityError), session_scope() as session:
        make_request(session, "req-1")
        make_request(session, "req-1")


def test_list_requests_newest_first_with_limit(db):
    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    with session_scope() as session:
        for offset in (2, 0, 3, 1):
            make_request(session, f"req-{offset}", ts=start + timedelta(minutes=offset))
    with session_scope() as session:
        assert [r.request_id for r in repo.list_requests(session)] == [
            "req-3",
            "req-2",
            "req-1",
            "req-0",
        ]
        assert [r.request_id for r in repo.list_requests(session, limit=2)] == ["req-3", "req-2"]


# --- Feedback --------------------------------------------------------------


def test_feedback_stats_without_feedback(db):
    with session_scope() as session:
        make_request(session, "req-1")
        assert repo.feedback_stats(session) == {"n": 0, "ok_rate": None, "by_route": {}}


def test_feedback_stats_by_route(db):
    with session_scope() as session:
        make_request(session, "rag-1", route="rag")
        make_request(session, "rag-2", route="rag")
        make_request(session, "agent-1", route="agent")
        make_request(session, "auto-1", route="automation")  # no feedback: not counted
        repo.save_feedback(session, request_id="rag-1", ok=True, comment="")
        repo.save_feedback(session, request_id="rag-2", ok=False, comment="Délai erroné")
        repo.save_feedback(session, request_id="rag-2", ok=False, comment="Toujours faux")
        repo.save_feedback(session, request_id="agent-1", ok=True, comment="Brouillon utilisable")

    with session_scope() as session:
        stats = repo.feedback_stats(session)
    assert stats == {
        "n": 4,
        "ok_rate": 0.5,
        "by_route": {"agent": {"n": 1, "ok": 1}, "rag": {"n": 3, "ok": 1}},
    }
    assert schemas.FeedbackStats(**stats).by_route["rag"] == {"n": 3, "ok": 1}


def test_feedback_ok_rate_is_rounded(db):
    with session_scope() as session:
        make_request(session, "req-1")
        for ok in (True, True, False):
            repo.save_feedback(session, request_id="req-1", ok=ok, comment="")
        assert repo.feedback_stats(session)["ok_rate"] == 0.6667


def test_save_feedback_stores_the_comment(db):
    with session_scope() as session:
        make_request(session, "req-1")
        feedback = repo.save_feedback(
            session, request_id="req-1", ok=False, comment="La majoration citée est fausse"
        )
        assert feedback.id == 1
    with session_scope() as session:
        stored = session.scalars(select(FeedbackRow)).one()
        assert (stored.request_id, stored.ok, stored.comment) == (
            "req-1",
            False,
            "La majoration citée est fausse",
        )
        assert stored.ts.tzinfo is not None


def test_feedback_on_an_unknown_request_is_refused_by_the_database(db):
    with pytest.raises(IntegrityError), session_scope() as session:
        repo.save_feedback(session, request_id="inconnu", ok=True, comment="")
    with session_scope() as session:
        assert repo.feedback_stats(session)["n"] == 0


# --- Engine ----------------------------------------------------------------


def test_sqlite_file_lives_under_the_data_directory(db):
    assert get_engine().dialect.name == "sqlite"
    assert (db.data_dir / "dossierops.db").is_file()


def test_engine_is_cached_and_rebuilt_after_a_settings_reset(db, tmp_path, monkeypatch):
    first = get_engine()
    assert get_engine() is first
    with session_scope() as session:
        make_ticket(session)

    # A data directory that does not exist yet is created for the SQLite file.
    other = tmp_path / "autre" / "données"
    monkeypatch.setenv("DATA_DIR", str(other))
    reset_settings()
    assert get_engine() is not first
    init_db()
    assert (other / "dossierops.db").is_file()
    with session_scope() as session:
        assert repo.list_tickets(session) == []  # a different, empty database


def test_sessions_work_from_worker_threads(db):
    # API handlers run in a thread pool, not in the thread that created the engine.
    def create(number: int) -> str:
        with session_scope() as session:
            return make_ticket(session, subject=f"Ticket {number}").ticket_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ticket_ids = list(pool.map(create, range(8)))
    assert sorted(ticket_ids) == [f"T-{number:06d}" for number in range(1, 9)]


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://dossierops:secret@db:5432/dossierops",
        "postgres://dossierops:secret@db:5432/dossierops",
        "postgresql+psycopg://dossierops:secret@db:5432/dossierops",
    ],
)
def test_postgres_urls_use_the_installed_psycopg_driver(env, monkeypatch, url):
    monkeypatch.setenv("DATABASE_URL", url)
    reset_settings()
    engine = get_engine()  # no connection is opened until the first query
    assert (engine.dialect.name, engine.driver) == ("postgresql", "psycopg")
    assert engine.url.host == "db"
    assert engine.url.database == "dossierops"


def test_schema_compiles_for_postgresql():
    ddl = {
        table.name: str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in Base.metadata.sorted_tables
    }
    assert set(ddl) == {"tickets", "actions", "request_log", "feedback"}
    assert "id SERIAL NOT NULL" in ddl["tickets"]
    assert "created_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl["tickets"]
    assert "payload JSON NOT NULL" in ddl["actions"]
    assert "decided_at TIMESTAMP WITH TIME ZONE," in ddl["actions"]
    assert "pii_types JSON NOT NULL" in ddl["request_log"]
    assert "needs_validation BOOLEAN NOT NULL" in ddl["request_log"]
    assert "FOREIGN KEY(request_id) REFERENCES request_log (request_id)" in ddl["feedback"]
