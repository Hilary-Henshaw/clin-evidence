"""FastAPI dependency functions for ClinEvidence."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, URLSafeSerializer

from clinevidence.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_SESSION_COOKIE_NAME = "clinevidence_session"


def get_app_settings() -> Settings:
    """Return cached application settings (FastAPI dependency)."""
    return get_settings()


@lru_cache(maxsize=1)
def _get_orchestrator_instance(
    settings: Settings,
) -> object:
    """Build the WorkflowOrchestrator singleton."""
    from clinevidence.agents.orchestrator import (
        WorkflowOrchestrator,
    )

    logger.info("Initialising WorkflowOrchestrator singleton")
    return WorkflowOrchestrator(settings)


def get_orchestrator(
    settings: Settings = Depends(get_app_settings),
) -> object:
    """Return singleton WorkflowOrchestrator instance."""
    return _get_orchestrator_instance(settings)


def _get_serializer(secret: str) -> URLSafeSerializer:
    """Build itsdangerous serializer."""
    return URLSafeSerializer(secret, salt="session")


def get_session(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> dict[str, str]:
    """Validate the signed session cookie.

    Returns the session dict with at least 'session_id'.
    Creates a new session if none exists.
    """
    secret = settings.session_secret.get_secret_value()
    serializer = _get_serializer(secret)
    raw: str | None = request.cookies.get(_SESSION_COOKIE_NAME)
    if raw is None:
        session_id = str(uuid.uuid4())
        logger.debug(
            "No session cookie found, creating new session",
            extra={"session_id": session_id},
        )
        return {"session_id": session_id}

    try:
        data: dict[str, str] = serializer.loads(raw)
        return data
    except BadSignature as exc:
        logger.warning(
            "Invalid session cookie signature",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or tampered session cookie.",
        ) from exc


def create_session(
    response: Response,
    settings: Settings = Depends(get_app_settings),
) -> str:
    """Create a new signed session cookie and return session_id."""
    secret = settings.session_secret.get_secret_value()
    serializer = _get_serializer(secret)
    session_id = str(uuid.uuid4())
    token: str = serializer.dumps({"session_id": session_id})
    response.set_cookie(
        key=_SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
    )
    logger.debug(
        "Session created",
        extra={"session_id": session_id},
    )
    return session_id
