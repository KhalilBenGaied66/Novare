"""Database tables (SQLAlchemy 2). SQLite by default, PostgreSQL through `DATABASE_URL`.

Only generic column types are used so the same models run on both databases. Text
columns carry no length: SQLite ignores lengths while PostgreSQL enforces them, so a
limit would only ever fail in production. Sizes are validated at the API boundary.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetime on every database.

    PostgreSQL keeps the offset, SQLite drops it. Values are converted to UTC before
    they are written and marked as UTC when they are read, so Python code always sees
    aware datetimes. A naive value is taken as UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class TicketRow(Base):
    __tablename__ = "tickets"
    # A ticket number is given to the client: SQLite must never reuse one, even if the
    # latest row were deleted (PostgreSQL sequences already guarantee this).
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    client_id: Mapped[str | None] = mapped_column(String)
    subject: Mapped[str] = mapped_column(String)
    body: Mapped[str] = mapped_column(Text)  # PII-masked request text or agent summary
    kind: Mapped[str] = mapped_column(String)  # "litige_standard" | "intervention"
    priority: Mapped[str | None] = mapped_column(String)  # "P1" | "P2" | "P3"
    source: Mapped[str] = mapped_column(String)  # "automation" | "agent"
    status: Mapped[str] = mapped_column(String, default="open")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    created_by: Mapped[str] = mapped_column(String)  # "system:RG-03" or the validator's name
    request_id: Mapped[str | None] = mapped_column(String)
    dedupe_key: Mapped[str | None] = mapped_column(String, index=True)

    @property
    def ticket_id(self) -> str:
        """Public identifier, e.g. `T-000012`."""
        return f"T-{self.id:06d}"


class ActionRow(Base):
    """An action proposed by the agent, waiting for a human decision."""

    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    # No foreign key to the request log: the action is stored while the request is
    # still being processed, before its log row exists.
    request_id: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)  # "create_ticket"
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    decided_by: Mapped[str | None] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text, default="")
    ticket_id: Mapped[str | None] = mapped_column(String)  # "T-000012" once approved


class RequestLogRow(Base):
    """One row per request handled by `/ask`. Never holds the raw user text."""

    __tablename__ = "request_log"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    route: Mapped[str] = mapped_column(String)
    rule: Mapped[str] = mapped_column(String, default="")
    mode: Mapped[str] = mapped_column(String)
    q_masked: Mapped[str] = mapped_column(Text, default="")
    client_id: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column()
    latency_ms: Mapped[int] = mapped_column(default=0)
    llm_calls: Mapped[int] = mapped_column(default=0)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    cost_eur: Mapped[float] = mapped_column(default=0.0)
    model: Mapped[str | None] = mapped_column(String)
    retrieval_mode: Mapped[str] = mapped_column(String, default="none")
    pii_types: Mapped[list] = mapped_column(JSON, default=list)
    needs_validation: Mapped[bool] = mapped_column(default=True)
    error: Mapped[str | None] = mapped_column(String)  # exception class name only


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("request_log.request_id"), index=True)
    ok: Mapped[bool] = mapped_column()
    comment: Mapped[str] = mapped_column(Text, default="")  # PII-masked by the caller
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
