"""LangGraph state definition for the ClinEvidence workflow."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class WorkflowState(TypedDict):
    """Shared state passed between all agent nodes."""

    # Conversation history (append-only via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # Session context
    session_id: str

    # Image context (set when an image is attached)
    image_path: str | None
    image_type: str | None
    # BRAIN_MRI | CHEST_XRAY | SKIN_LESION | OTHER | NON_MEDICAL

    # Routing decisions
    selected_agent: str | None
    # CONVERSATION | KNOWLEDGE_BASE | WEB_EVIDENCE
    # | BRAIN_MRI | CHEST_XRAY | SKIN_LESION
    routing_confidence: float

    # Output
    response: str | None
    sources: list[str]

    # Validation
    requires_validation: bool
    validation_approved: bool | None

    # Error tracking
    error: str | None
