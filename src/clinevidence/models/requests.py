"""Pydantic request schemas for ClinEvidence API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming text query from a clinician."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Clinical question or instruction.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique session identifier.",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Strip whitespace and validate non-empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message must not be blank.")
        return stripped


class ValidationRequest(BaseModel):
    """Human validation decision for a pending analysis."""

    session_id: str = Field(..., min_length=1, max_length=128)
    approved: bool = Field(
        ...,
        description=("True to approve the analysis, False to reject."),
    )
    feedback: str | None = Field(
        default=None,
        max_length=2048,
        description="Optional clinician feedback.",
    )


class SpeechRequest(BaseModel):
    """Text-to-speech synthesis request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to convert to speech.",
    )
    voice_id: str | None = Field(
        default=None,
        description="Eleven Labs voice ID override.",
    )
