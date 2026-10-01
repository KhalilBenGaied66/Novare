"""Database engine and unit-of-work helper."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings, on_reset, reset_settings
from app.db.models import Base

_engine: Engine | None = None


def get_engine() -> Engine:
    """Engine for the configured database, created on first use."""
    global _engine
    if _engine is None:
        _engine = _create_engine(_database_url())
    return _engine


@on_reset
def _dispose_engine() -> None:
    """Close the connections so that the next call follows the current settings."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def init_db() -> None:
    """Create the missing tables. Existing tables are left untouched (no migration)."""
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction: committed when the block succeeds, rolled back when it raises.

    Rows stay readable after the block (`expire_on_commit=False`), so a caller can
    return a row it has just created without keeping the session open.
    """
    with Session(get_engine(), expire_on_commit=False) as session, session.begin():
        yield session


@contextmanager
def temporary_database() -> Iterator[None]:
    """Run a block against an empty SQLite database that is deleted afterwards.

    Used by the evaluation and the LLM smoke script, which run real requests and must
    not leave their tickets and request log in the application database. Settings are
    reset on entry and on exit, so cached objects follow the switch.
    """
    previous = os.environ.get("DATABASE_URL")
    with tempfile.TemporaryDirectory(prefix="dossierops-") as workdir:
        os.environ["DATABASE_URL"] = f"sqlite:///{(Path(workdir) / 'temporary.db').as_posix()}"
        reset_settings()
        try:
            init_db()
            yield
        finally:
            if previous is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous
            # Closes the temporary database before its directory is removed.
            reset_settings()


def _database_url() -> URL:
    url = make_url(get_settings().resolved_database_url)
    # Only psycopg 3 is installed: with a bare "postgresql://" URL SQLAlchemy would
    # look for psycopg2.
    if url.drivername in ("postgresql", "postgres"):
        url = url.set(drivername="postgresql+psycopg")
    return url


def _create_engine(url: URL) -> Engine:
    if url.get_backend_name() != "sqlite":
        # pre_ping replaces connections closed by the server or a network device.
        return create_engine(url, pool_pre_ping=True)

    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    """SQLite ignores foreign keys unless asked; PostgreSQL always enforces them."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
