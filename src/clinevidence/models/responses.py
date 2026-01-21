"""Pydantic response schemas for ClinEvidence API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A cited source in an evidence-based response."""

    title: str
    url: str | None = None
    page: int | None = None
    source_type: str = "document"  # document | web | pubmed


class ChatResponse(BaseModel):
    """Response to a clinical query."""

    message: str
    agent_used: str = Field(
        description=(
            "Which agent handled the query: "
            "CONVERSATION, KNOWLEDGE_BASE, WEB_EVIDENCE, "
            "BRAIN_MRI, CHEST_XRAY, SKIN_LESION"
        )
    )
    sources: list[SourceReference] = []
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent confidence score, if available.",
    )
    processing_time_ms: int
    session_id: str
    requires_validation: bool = False


class UploadResponse(BaseModel):
    """Result of a medical image upload and analysis."""

    filename: str
    image_type: str = Field(
        description=(
            "Detected modality: BRAIN_MRI, CHEST_XRAY, "
            "SKIN_LESION, OTHER, NON_MEDICAL"
        )
    )
    analysis: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_validation: bool
    session_id: str


class ValidationResponse(BaseModel):
    """Acknowledgement of a human validation decision."""

    status: str = Field(description="'approved' or 'rejected'")
    message: str


class TranscribeResponse(BaseModel):
    """Transcribed text from an audio file."""

    text: str
    duration_ms: int | None = None


class SpeechResponse(BaseModel):
    """Path to the generated audio file."""

    audio_url: str
    duration_ms: int | None = None


class HealthResponse(BaseModel):
    """Service health status."""

    status: str = "ok"
    version: str
    knowledge_base_ready: bool
