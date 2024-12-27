"""Unit tests for the ModalityDetector agent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestModalityDetector:
    """Tests for ModalityDetector.detect."""

    @patch("clinevidence.agents.imaging.modality_detector.AzureChatOpenAI")
    def test_detect_chest_xray_returns_correct_type(
        self,
        mock_llm_cls: MagicMock,
        settings_fixture: object,
        tmp_path: Path,
    ) -> None:
        """Chest X-ray image should be classified as CHEST_XRAY."""
        img_path = tmp_path / "xray.jpg"
        img_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.content = json.dumps(
            {
                "image_type": "CHEST_XRAY",
                "reasoning": "Bilateral lung fields visible",
                "confidence": 0.93,
            }
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from clinevidence.agents.imaging.modality_detector import (
            ModalityDetector,
        )

        detector = ModalityDetector(settings_fixture)  # type: ignore[arg-type]
        result = detector.detect(img_path)

        assert result["image_type"] == "CHEST_XRAY"
        assert result["confidence"] == pytest.approx(0.93)
        assert "lung" in result["reasoning"].lower()

    @patch("clinevidence.agents.imaging.modality_detector.AzureChatOpenAI")
    def test_detect_raises_for_missing_image(
        self,
        mock_llm_cls: MagicMock,
        settings_fixture: object,
        tmp_path: Path,
    ) -> None:
        """Should raise ValueError for a non-existent image path."""
        from clinevidence.agents.imaging.modality_detector import (
            ModalityDetector,
        )

        mock_llm_cls.return_value = MagicMock()
        detector = ModalityDetector(settings_fixture)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="does not exist"):
            detector.detect(tmp_path / "nonexistent.jpg")

    @patch("clinevidence.agents.imaging.modality_detector.AzureChatOpenAI")
    def test_detect_handles_invalid_json_from_llm(
        self,
        mock_llm_cls: MagicMock,
        settings_fixture: object,
        tmp_path: Path,
    ) -> None:
        """Malformed LLM JSON should default to OTHER type."""
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 50)

        mock_resp = MagicMock()
        mock_resp.content = "Not a JSON response at all"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from clinevidence.agents.imaging.modality_detector import (
            ModalityDetector,
        )

        detector = ModalityDetector(settings_fixture)  # type: ignore[arg-type]
        result = detector.detect(img_path)

        assert result["image_type"] == "OTHER"
        assert result["confidence"] == 0.0

    @patch("clinevidence.agents.imaging.modality_detector.AzureChatOpenAI")
    def test_detect_non_medical_image_returns_non_medical_type(
        self,
        mock_llm_cls: MagicMock,
        settings_fixture: object,
        tmp_path: Path,
    ) -> None:
        """Photo of a landscape should be classified as NON_MEDICAL."""
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.content = json.dumps(
            {
                "image_type": "NON_MEDICAL",
                "reasoning": "No medical content detected",
                "confidence": 0.98,
            }
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from clinevidence.agents.imaging.modality_detector import (
            ModalityDetector,
        )

        detector = ModalityDetector(settings_fixture)  # type: ignore[arg-type]
        result = detector.detect(img_path)

        assert result["image_type"] == "NON_MEDICAL"
        assert result["confidence"] == pytest.approx(0.98)
