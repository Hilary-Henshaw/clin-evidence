"""ClinEvidence — FastAPI application entry point."""

from __future__ import annotations

import logging
import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from clinevidence import __version__
from clinevidence.api.chat import router as chat_router
from clinevidence.api.media import router as media_router
from clinevidence.api.speech import router as speech_router
from clinevidence.middleware import RequestTracingMiddleware
from clinevidence.models.responses import HealthResponse
from clinevidence.settings import get_settings

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure structured JSON-style logging."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": (
                        '{"time": "%(asctime)s", '
                        '"level": "%(levelname)s", '
                        '"logger": "%(name)s", '
                        '"message": "%(message)s"}'
                    ),
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"],
            },
            "loggers": {
                "clinevidence": {
                    "level": "INFO",
                    "propagate": True,
                },
                "uvicorn": {
                    "level": "INFO",
                    "propagate": False,
                    "handlers": ["console"],
                },
            },
        }
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """Application lifespan: create directories on startup."""
    configure_logging()
    settings = get_settings()

    required_dirs = [
        Path(settings.upload_dir) / "sessions",
        Path(settings.upload_dir) / "images",
        Path(settings.upload_dir) / "audio",
        Path(settings.kb_qdrant_path),
        Path(settings.kb_docs_path),
        Path(settings.kb_parsed_docs_path),
        Path("./models"),
        Path("./data/raw"),
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ClinEvidence started",
        extra={"version": __version__},
    )
    yield
    logger.info("ClinEvidence shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(
        title="ClinEvidence",
        description=(
            "Clinical Evidence Retrieval & Diagnostic Support "
            "for ICU Teams — powered by multi-agent AI with "
            "LangGraph, Qdrant RAG, and medical image analysis."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,  # type: ignore[arg-type]
    )

    # CORS (permissive for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request tracing
    app.add_middleware(RequestTracingMiddleware)

    # Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Routers
    app.include_router(chat_router)
    app.include_router(media_router)
    app.include_router(speech_router)

    # Static file serving for audio outputs
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/uploads",
        StaticFiles(directory=str(upload_path)),
        name="uploads",
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Service health check",
    )
    async def health_check() -> HealthResponse:
        """Return service health and readiness status."""
        from clinevidence.dependencies import (
            _get_orchestrator_instance,
        )

        kb_ready = False
        try:
            from clinevidence.agents.orchestrator import (
                WorkflowOrchestrator,
            )

            orchestrator = _get_orchestrator_instance(settings)
            if isinstance(orchestrator, WorkflowOrchestrator):
                kb_ready = orchestrator.knowledge_base.is_ready
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            version=__version__,
            knowledge_base_ready=kb_ready,
        )

    @app.get(
        "/",
        include_in_schema=False,
    )
    async def root_redirect() -> RedirectResponse:
        """Redirect root to API documentation."""
        return RedirectResponse(url="/docs")

    return app


app = create_app()


def run() -> None:
    """Run the application via uvicorn (entry point)."""
    settings = get_settings()
    uvicorn.run(
        "clinevidence.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
