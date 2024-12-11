"""Integration tests for the RAG knowledge base pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestKnowledgeBase:
    """Tests for the KnowledgeBase RAG pipeline."""

    @patch("clinevidence.agents.rag.pipeline.KnowledgeStore")
    @patch("clinevidence.agents.rag.pipeline.AzureChatOpenAI")
    @patch("clinevidence.agents.rag.pipeline.AzureOpenAIEmbeddings")
    def test_knowledge_base_is_not_ready_when_empty(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_store_cls: MagicMock,
        settings_fixture: Any,
    ) -> None:
        """is_ready should return False when collection is empty."""
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False
        mock_store_cls.return_value = mock_store

        from clinevidence.agents.rag.pipeline import KnowledgeBase

        kb = KnowledgeBase(settings_fixture)
        assert kb.is_ready is False

    @patch("clinevidence.agents.rag.pipeline.DocumentExtractor")
    @patch("clinevidence.agents.rag.pipeline.DocumentFormatter")
    @patch("clinevidence.agents.rag.pipeline.KnowledgeStore")
    @patch("clinevidence.agents.rag.pipeline.AzureChatOpenAI")
    @patch("clinevidence.agents.rag.pipeline.AzureOpenAIEmbeddings")
    def test_ingest_indexes_document_chunks(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_formatter_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        settings_fixture: Any,
        tmp_path: Path,
    ) -> None:
        """Ingest should extract, chunk, and index documents."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"PDF content")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (
            "# Clinical Guidelines\nSepsis management...",
            [],
        )
        mock_extractor_cls.return_value = mock_extractor

        mock_formatter = MagicMock()
        mock_formatter.replace_image_placeholders.return_value = (
            "# Clinical Guidelines\nSepsis management..."
        )
        mock_formatter.semantic_chunk.return_value = [
            "Chunk 1: Sepsis management guidelines",
            "Chunk 2: Antibiotic therapy recommendations",
        ]
        mock_formatter_cls.return_value = mock_formatter

        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False
        mock_store_cls.return_value = mock_store

        from clinevidence.agents.rag.pipeline import KnowledgeBase

        kb = KnowledgeBase(settings_fixture)
        result = kb.ingest(pdf)

        assert result["chunks_indexed"] == 2
        assert result["images_processed"] == 0
        assert result["path"] == str(pdf)
        mock_store.add_documents.assert_called_once()

    @patch("clinevidence.agents.rag.pipeline.QueryEnricher")
    @patch("clinevidence.agents.rag.pipeline.ResultRanker")
    @patch("clinevidence.agents.rag.pipeline.AnswerSynthesizer")
    @patch("clinevidence.agents.rag.pipeline.KnowledgeStore")
    @patch("clinevidence.agents.rag.pipeline.AzureChatOpenAI")
    @patch("clinevidence.agents.rag.pipeline.AzureOpenAIEmbeddings")
    def test_query_returns_result_with_sources(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_synth_cls: MagicMock,
        mock_ranker_cls: MagicMock,
        mock_enricher_cls: MagicMock,
        settings_fixture: Any,
    ) -> None:
        """Query should return an answer with source citations."""
        mock_enricher = MagicMock()
        mock_enricher.enrich.return_value = (
            "septic shock norepinephrine vasopressors"
        )
        mock_enricher_cls.return_value = mock_enricher

        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = [
            Document(
                page_content="Norepinephrine is first-line.",
                metadata={"title": "Sepsis Guidelines 2021"},
            )
        ]
        mock_store_cls.return_value = mock_store

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = (
            [
                Document(
                    page_content="Norepinephrine is first-line.",
                    metadata={"title": "Sepsis Guidelines 2021"},
                )
            ],
            [],
        )
        mock_ranker_cls.return_value = mock_ranker

        mock_synth = MagicMock()
        mock_synth.synthesize.return_value = (
            "Norepinephrine is the first-line vasopressor.",
            ["Sepsis Guidelines 2021"],
            0.85,
        )
        mock_synth_cls.return_value = mock_synth

        from clinevidence.agents.rag.pipeline import KnowledgeBase

        kb = KnowledgeBase(settings_fixture)
        result = kb.query(
            "What is the first-line vasopressor in septic shock?",
            [],
        )

        assert "Norepinephrine" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["confidence"] == pytest.approx(0.85)

    @patch("clinevidence.agents.rag.pipeline.QueryEnricher")
    @patch("clinevidence.agents.rag.pipeline.ResultRanker")
    @patch("clinevidence.agents.rag.pipeline.AnswerSynthesizer")
    @patch("clinevidence.agents.rag.pipeline.KnowledgeStore")
    @patch("clinevidence.agents.rag.pipeline.AzureChatOpenAI")
    @patch("clinevidence.agents.rag.pipeline.AzureOpenAIEmbeddings")
    def test_query_falls_back_on_low_confidence(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_synth_cls: MagicMock,
        mock_ranker_cls: MagicMock,
        mock_enricher_cls: MagicMock,
        settings_fixture: Any,
    ) -> None:
        """Query with confidence below threshold should still return."""
        mock_enricher = MagicMock()
        mock_enricher.enrich.return_value = "rare disease query"
        mock_enricher_cls.return_value = mock_enricher

        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = []
        mock_store_cls.return_value = mock_store

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = ([], [])
        mock_ranker_cls.return_value = mock_ranker

        mock_synth = MagicMock()
        mock_synth.synthesize.return_value = (
            "No relevant evidence found. Please consult a clinician.",
            [],
            0.0,
        )
        mock_synth_cls.return_value = mock_synth

        from clinevidence.agents.rag.pipeline import KnowledgeBase

        kb = KnowledgeBase(settings_fixture)
        result = kb.query(
            "Treatment for a very rare genetic syndrome?",
            [],
        )

        assert result["confidence"] == pytest.approx(0.0)
        assert result["sources"] == []
