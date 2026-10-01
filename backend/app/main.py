"""FastAPI application: lifespan, request id and access log, error mapping, health checks."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, request_id_var, setup_logging
from app.core.schemas import ReadyResponse
from app.db import session
from app.ingestion import ingest
from app.retrieval import hybrid

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    session.init_db()
    settings = get_settings()
    status = hybrid.index_status()
    # An index built without vectors (by an offline evaluation, for instance) would
    # silently serve lexical search only.
    stale = settings.retrieval_mode == "hybrid" and status["mode"] != "hybrid"
    if settings.auto_ingest and (not status["ready"] or stale):
        try:
            report = ingest.ingest_docs()
            logger.info("auto_ingest", extra={"docs": report.docs, "chunks": report.chunks})
        except (OSError, ValueError) as exc:
            # The API still starts: /ready reports the missing index, /ask answers 503.
            logger.error("auto_ingest_failed", extra={"error": type(exc).__name__})
    yield


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    app = FastAPI(title="Novare DossierOps", version="1.0.0", lifespan=lifespan)

    if settings.cors_origins:
        origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # Always generated here: the id is the key of the request log, so a value chosen
        # by the caller could collide with an earlier request.
        request_id = uuid.uuid4().hex
        request_id_var.set(request_id)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # The path only: a query string could carry data that must not reach the logs.
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(hybrid.IndexNotReady)
    async def index_not_ready(_request: Request, _exc: hybrid.IndexNotReady) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "L'index documentaire n'est pas prêt : lancez POST /api/v1/ingest "
                "puis réessayez."
            },
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Where and why only: the default body also echoes the rejected value, which
        # can be the text of a request, and fails to serialise values such as NaN.
        errors = [{"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", extra={"error": type(exc).__name__}, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Erreur interne"})

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process answers."""
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> JSONResponse:
        """Readiness: the database answers and the document index is loaded."""
        checks = {"database": _database_check(), "index": _index_check()}
        is_ready = all(result.startswith("ok") for result in checks.values())
        report = ReadyResponse(ready=is_ready, checks=checks)
        return JSONResponse(status_code=200 if report.ready else 503, content=report.model_dump())

    app.include_router(router, prefix="/api/v1")
    return app


def _database_check() -> str:
    try:
        with session.session_scope() as db:
            db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("database_unreachable", extra={"error": type(exc).__name__})
        return "indisponible"
    return "ok"


def _index_check() -> str:
    status = hybrid.index_status()
    if not status["ready"]:
        return "absent : lancez POST /api/v1/ingest"
    return f"ok ({status['chunks']} passages, mode {status['mode']})"


app = create_app()
