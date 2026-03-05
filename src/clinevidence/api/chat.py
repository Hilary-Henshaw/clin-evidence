"""Chat endpoint for clinical query processing."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from clinevidence.dependencies import (
    get_orchestrator,
    get_session,
)
from clinevidence.models.requests import ChatRequest
from clinevidence.models.responses import (
    ChatResponse,
    SourceReference,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Clinical Queries"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Submit a clinical query",
    description=(
        "Routes the query through the multi-agent workflow: "
        "knowledge base retrieval, web evidence search, or "
        "direct clinical conversation."
    ),
)
async def submit_query(
    request: ChatRequest,
    session: dict[str, str] = Depends(get_session),
    orchestrator: Any = Depends(get_orchestrator),
) -> ChatResponse:
    """Process a clinical query through the orchestrator.

    The session_id in the request must match the session cookie.
    """
    session_id = session.get("session_id", "")
    if session_id and request.session_id != session_id:
        logger.warning(
            "Session ID mismatch",
            extra={
                "cookie_session": session_id,
                "request_session": request.session_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session ID does not match the session cookie.",
        )

    t0 = time.perf_counter()
    try:
        result = orchestrator.process(
            query=request.message,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.error(
            "Orchestrator processing failed",
            extra={"session_id": request.session_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An error occurred processing your query. Please try again."
            ),
        ) from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    response_text: str = result.get("response") or ("No response generated.")
    agent_used: str = str(result.get("selected_agent", "CONVERSATION"))
    confidence: float | None = result.get("routing_confidence")
    raw_sources: list[Any] = result.get("sources", [])
    requires_validation: bool = bool(result.get("requires_validation", False))

    sources: list[SourceReference] = []
    for src in raw_sources:
        if isinstance(src, str):
            if src.startswith("http"):
                sources.append(
                    SourceReference(
                        title=src,
                        url=src,
                        source_type="web",
                    )
                )
            else:
                sources.append(
                    SourceReference(
                        title=src,
                        source_type="document",
                    )
                )
        elif isinstance(src, dict):
            sources.append(
                SourceReference(
                    title=src.get("title", "Unknown"),
                    url=src.get("url"),
                    page=src.get("page"),
                    source_type=src.get("source_type", "document"),
                )
            )

    return ChatResponse(
        message=response_text,
        agent_used=agent_used,
        sources=sources,
        confidence=confidence,
        processing_time_ms=elapsed_ms,
        session_id=request.session_id,
        requires_validation=requires_validation,
    )
