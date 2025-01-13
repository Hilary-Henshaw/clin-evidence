"""Integration tests for the POST /v1/chat endpoint."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

_SESSION = "test-session"


class TestChatEndpoint:
    """Tests for the /v1/chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_returns_200_for_valid_request(
        self, async_client: AsyncClient
    ) -> None:
        """Valid chat request should return HTTP 200 with response body."""
        payload = {
            "message": "What is the first-line treatment "
            "for community-acquired pneumonia?",
            "session_id": "test-session",
        }
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "agent_used" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_chat_returns_422_for_empty_message(
        self, async_client: AsyncClient
    ) -> None:
        """Empty message string should fail Pydantic validation."""
        payload = {
            "message": "   ",
            "session_id": "test-session",
        }
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_returns_422_for_missing_session_id(
        self, async_client: AsyncClient
    ) -> None:
        """Request without session_id should fail validation."""
        payload = {"message": "Explain septic shock management."}
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_includes_processing_time_in_response(
        self, async_client: AsyncClient
    ) -> None:
        """Response should include processing_time_ms field."""
        payload = {
            "message": "What are the signs of pulmonary oedema?",
            "session_id": _SESSION,
        }
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_chat_response_contains_required_fields(
        self, async_client: AsyncClient
    ) -> None:
        """Response schema must include all required fields."""
        payload = {
            "message": "Explain the SOFA score.",
            "session_id": _SESSION,
        }
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        required_fields = {
            "message",
            "agent_used",
            "sources",
            "processing_time_ms",
            "session_id",
            "requires_validation",
        }
        assert required_fields.issubset(set(data.keys()))

    @pytest.mark.asyncio
    async def test_chat_returns_422_for_message_too_long(
        self, async_client: AsyncClient
    ) -> None:
        """Message exceeding 4096 characters should fail validation."""
        payload = {
            "message": "A" * 4097,
            "session_id": "test-session",
        }
        response = await async_client.post("/v1/chat", json=payload)

        assert response.status_code == 422
