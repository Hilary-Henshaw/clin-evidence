"""Unit tests for the ResultRanker agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from clinevidence.agents.rag.result_ranker import ResultRanker


def _make_docs(n: int) -> list[Document]:
    """Create n test Documents."""
    return [
        Document(
            page_content=f"Clinical content chunk {i}",
            metadata={"title": f"Doc {i}", "chunk_index": i},
        )
        for i in range(n)
    ]


class TestResultRanker:
    """Tests for ResultRanker.rerank."""

    @patch("clinevidence.agents.rag.result_ranker.CrossEncoder")
    def test_rerank_orders_by_score_descending(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Documents should be returned sorted by score, highest first."""
        docs = _make_docs(3)
        scores = [0.3, 0.9, 0.6]  # doc 1 should rank first

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array(scores)
        mock_ce_cls.return_value = mock_encoder

        ranker = ResultRanker()
        result_docs, _ = ranker.rerank("sepsis treatment", docs, top_k=3)

        assert result_docs[0].page_content == docs[1].page_content
        assert len(result_docs) == 3

    @patch("clinevidence.agents.rag.result_ranker.CrossEncoder")
    def test_rerank_limits_to_top_k(self, mock_ce_cls: MagicMock) -> None:
        """Result count should not exceed top_k."""
        docs = _make_docs(5)
        scores = [0.5, 0.8, 0.3, 0.9, 0.1]

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array(scores)
        mock_ce_cls.return_value = mock_encoder

        ranker = ResultRanker()
        result_docs, _ = ranker.rerank("pneumonia antibiotics", docs, top_k=2)

        assert len(result_docs) == 2

    @patch("clinevidence.agents.rag.result_ranker.CrossEncoder")
    def test_rerank_extracts_image_paths_from_metadata(
        self, mock_ce_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Image paths in metadata should be returned."""
        img_file = tmp_path / "chest.png"
        img_file.write_bytes(b"PNG")

        docs = [
            Document(
                page_content="Chest X-ray findings",
                metadata={"image_paths": [str(img_file)]},
            )
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array([0.9])
        mock_ce_cls.return_value = mock_encoder

        ranker = ResultRanker()
        _, image_paths = ranker.rerank("chest xray", docs, top_k=1)

        assert len(image_paths) == 1
        assert image_paths[0] == img_file

    @patch("clinevidence.agents.rag.result_ranker.CrossEncoder")
    def test_rerank_handles_empty_document_list(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Empty input should return empty output without error."""
        mock_encoder = MagicMock()
        mock_ce_cls.return_value = mock_encoder

        ranker = ResultRanker()
        result_docs, image_paths = ranker.rerank("acute MI", [], top_k=3)

        assert result_docs == []
        assert image_paths == []
        mock_encoder.predict.assert_not_called()
