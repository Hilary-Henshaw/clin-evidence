"""Shared pytest fixtures for ClinEvidence tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ── Environment setup ─────────────────────────────────────────────
@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all required environment variables for tests."""
    env_vars = {
        "deployment_name": "test-deployment",
        "model_name": "gpt-4o",
        "azure_endpoint": "https://test.openai.azure.com/",
        "openai_api_key": "test-openai-key",
        "openai_api_version": "2024-02-15",
        "embedding_deployment_name": "test-embedding",
        "embedding_model_name": "text-embedding-ada-002",
        "embedding_azure_endpoint": ("https://test.openai.azure.com/"),
        "embedding_openai_api_key": "test-embedding-key",
        "embedding_openai_api_version": "2024-02-15",
        "ELEVEN_LABS_API_KEY": "test-elevenlabs-key",
        "TAVILY_API_KEY": "test-tavily-key",
        "SESSION_SECRET_KEY": "test-secret-key-for-testing",
        "HUGGINGFACE_TOKEN": "test-hf-token",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def settings_fixture() -> Any:
    """Return a Settings instance with test values."""
    from clinevidence.settings import Settings

    return Settings(  # type: ignore[call-arg]
        **{
            "deployment_name": "test-deployment",
            "model_name": "gpt-4o",
            "azure_endpoint": "https://test.openai.azure.com/",
            "openai_api_key": "test-openai-key",
            "openai_api_version": "2024-02-15",
            "embedding_deployment_name": "test-embedding",
            "embedding_model_name": "text-embedding-ada-002",
            "embedding_azure_endpoint": ("https://test.openai.azure.com/"),
            "embedding_openai_api_key": "test-embedding-key",
            "embedding_openai_api_version": "2024-02-15",
            "ELEVEN_LABS_API_KEY": "test-elevenlabs-key",
            "TAVILY_API_KEY": "test-tavily-key",
            "SESSION_SECRET_KEY": "test-secret-for-tests",
        }
    )


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Return a mock AzureChatOpenAI with fixed output."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "This is a test clinical response with appropriate "
        "disclaimers. Always consult a qualified clinician."
    )
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def sample_pdf_path(tmp_path: Any) -> Any:
    """Create a minimal test text file (simulates a document)."""
    doc_path = tmp_path / "test_clinical_doc.txt"
    doc_path.write_text(
        "Clinical Evidence Test Document\n\n"
        "Sepsis management guidelines recommend early antibiotic "
        "administration within 1 hour of recognition. "
        "Fluid resuscitation with crystalloids at 30 mL/kg "
        "is recommended in the first 3 hours.\n\n"
        "References: Surviving Sepsis Campaign 2021."
    )
    return doc_path


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Return a MagicMock of WorkflowOrchestrator."""
    mock = MagicMock()
    mock.process.return_value = {
        "response": "Test clinical response with disclaimer.",
        "selected_agent": "CONVERSATION",
        "routing_confidence": 0.9,
        "sources": [],
        "requires_validation": False,
        "error": None,
        "image_type": None,
        "session_id": "test-session-123",
        "messages": [],
        "image_path": None,
        "validation_approved": None,
    }
    mock.knowledge_base.is_ready = True
    return mock


_TEST_SESSION_ID = "test-session"


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Return an AsyncClient wrapping the ClinEvidence app."""
    from clinevidence.dependencies import (
        get_orchestrator,
        get_session,
    )
    from clinevidence.main import create_app

    test_app = create_app()

    mock_orch = MagicMock()
    mock_orch.process.return_value = {
        "response": "Mock clinical answer with disclaimer.",
        "selected_agent": "CONVERSATION",
        "routing_confidence": 0.85,
        "sources": [],
        "requires_validation": False,
        "error": None,
        "image_type": None,
        "session_id": _TEST_SESSION_ID,
        "messages": [],
        "image_path": None,
        "validation_approved": None,
    }

    # Override both orchestrator and session dependencies
    test_app.dependency_overrides[get_orchestrator] = lambda: mock_orch
    test_app.dependency_overrides[get_session] = lambda: {
        "session_id": _TEST_SESSION_ID
    }

    transport = ASGITransport(app=test_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client
