"""Media upload and validation endpoints."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from werkzeug.utils import secure_filename

from clinevidence.agents.orchestrator import WorkflowOrchestrator
from clinevidence.dependencies import (
    get_app_settings,
    get_orchestrator,
    get_session,
)
from clinevidence.models.requests import ValidationRequest
from clinevidence.models.responses import (
    UploadResponse,
    ValidationResponse,
)
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Medical Imaging"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and analyse a medical image",
    description=(
        "Accepts a medical image file, detects the modality, "
        "and routes it to the appropriate imaging analyser."
    ),
)
async def upload_image(
    file: UploadFile,
    session: dict[str, str] = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
    orchestrator: Any = Depends(get_orchestrator),
) -> UploadResponse:
    """Upload a medical image for analysis."""
    session_id = session.get("session_id", str(uuid.uuid4()))

    filename = file.filename or "upload"
    safe_name = secure_filename(filename)
    if not safe_name:
        safe_name = f"{uuid.uuid4()}.jpg"

    ext = Path(safe_name).suffix.lstrip(".").lower()
    if ext not in settings.allowed_image_formats:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File format '.{ext}' is not supported. "
                f"Allowed: {settings.allowed_image_formats}"
            ),
        )

    upload_dir = Path(settings.upload_dir) / "images" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4()}_{safe_name}"
    dest_path = upload_dir / unique_name

    try:
        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File exceeds maximum size of "
                    f"{settings.max_upload_bytes // 1024 // 1024}"
                    f" MB."
                ),
            )

        dest_path.write_bytes(content)
        logger.info(
            "Image uploaded",
            extra={
                "filename": safe_name,
                "size_kb": len(content) // 1024,
                "session_id": session_id,
            },
        )

        assert isinstance(orchestrator, WorkflowOrchestrator)
        result = orchestrator.process(
            query=(f"Analyse the uploaded medical image: {safe_name}"),
            session_id=session_id,
            image_path=str(dest_path),
        )

    except HTTPException:
        if dest_path.exists():
            dest_path.unlink()
        raise
    except Exception:
        if dest_path.exists():
            dest_path.unlink()
        logger.error(
            "Image upload processing failed",
            extra={"session_id": session_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image analysis failed. Please try again.",
        ) from None

    response_text: str = result.get("response") or ("Analysis complete.")
    image_type: str = str(result.get("image_type", "OTHER"))
    confidence: float = float(result.get("routing_confidence", 0.0))
    requires_validation: bool = bool(result.get("requires_validation", False))

    return UploadResponse(
        filename=safe_name,
        image_type=image_type,
        analysis=response_text,
        confidence=confidence,
        requires_validation=requires_validation,
        session_id=session_id,
    )


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Submit human validation decision",
    description=(
        "Resumes a paused workflow with the clinician's approval "
        "or rejection of an AI-generated analysis."
    ),
)
async def validate_analysis(
    request: ValidationRequest,
    session: dict[str, str] = Depends(get_session),
    orchestrator: Any = Depends(get_orchestrator),
) -> ValidationResponse:
    """Resume a paused workflow after human validation."""
    session_id = session.get("session_id", "")
    if session_id and request.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=("Session ID does not match the session cookie."),
        )

    try:
        assert isinstance(orchestrator, WorkflowOrchestrator)
        orchestrator.resume_after_validation(
            session_id=request.session_id,
            approved=request.approved,
        )
    except Exception as exc:
        logger.error(
            "Validation resumption failed",
            extra={"session_id": request.session_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process validation decision.",
        ) from exc

    status_str = "approved" if request.approved else "rejected"
    message = f"Analysis {status_str}." + (
        f" Feedback recorded: {request.feedback}" if request.feedback else ""
    )

    logger.info(
        "Validation processed",
        extra={
            "session_id": request.session_id,
            "approved": request.approved,
        },
    )
    return ValidationResponse(
        status=status_str,
        message=message,
    )
