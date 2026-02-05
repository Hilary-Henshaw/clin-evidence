"""Unit tests for the QueryEnricher agent."""

from __future__ import annotations

from unittest.mock import MagicMock

from clinevidence.agents.rag.query_enricher import QueryEnricher


def _make_llm(classify_response: str, enrich_response: str) -> MagicMock:
    """Build a mock LLM with two sequential responses."""
    mock_llm = MagicMock()
    responses = []
    for text in [classify_response, enrich_response]:
        r = MagicMock()
        r.content = text
        responses.append(r)
    mock_llm.invoke.side_effect = responses
    return mock_llm


class TestQueryEnricher:
    """Tests for QueryEnricher.enrich."""

    def test_enrich_medical_query_adds_synonyms(self) -> None:
        """Medical query should return enriched version with extra terms."""
        enriched_text = (
            "septic shock hypotension vasopressors "
            "noradrenaline norepinephrine fluid resuscitation "
            "crystalloids"
        )
        mock_llm = _make_llm(
            classify_response="medical",
            enrich_response=enriched_text,
        )

        enricher = QueryEnricher(mock_llm)
        result = enricher.enrich("What is the treatment for septic shock?")

        assert result == enriched_text
        assert len(result) > len("What is the treatment for septic shock?")
        assert mock_llm.invoke.call_count == 2

    def test_enrich_returns_original_on_llm_failure(
        self,
    ) -> None:
        """When LLM raises, the original query is returned unchanged."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        enricher = QueryEnricher(mock_llm)
        original = "myocardial infarction management"
        result = enricher.enrich(original)

        assert result == original

    def test_enrich_non_medical_query_unchanged(self) -> None:
        """Non-medical query should pass through without enrichment."""
        mock_resp = MagicMock()
        mock_resp.content = "general"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp

        enricher = QueryEnricher(mock_llm)
        original = "What is the weather today?"
        result = enricher.enrich(original)

        assert result == original
        # Should only call once (classification only)
        assert mock_llm.invoke.call_count == 1

    def test_enrich_falls_back_on_empty_llm_output(
        self,
    ) -> None:
        """If LLM returns empty enrichment, original query is used."""
        classify_resp = MagicMock()
        classify_resp.content = "medical"
        enrich_resp = MagicMock()
        enrich_resp.content = "   "  # whitespace-only

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            classify_resp,
            enrich_resp,
        ]

        enricher = QueryEnricher(mock_llm)
        original = "acute kidney injury treatment"
        result = enricher.enrich(original)

        assert result == original
