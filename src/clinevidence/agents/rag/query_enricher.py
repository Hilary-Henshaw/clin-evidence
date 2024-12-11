"""Query enrichment with medical synonyms and related concepts."""

from __future__ import annotations

import logging

from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

_MEDICAL_INDICATOR_PROMPT = (
    "You are a medical terminology expert. "
    "Determine whether the following query is medical in nature. "
    "Respond with exactly one word: 'medical' or 'general'.\n\n"
    "Query: {query}"
)

_ENRICHMENT_PROMPT = (
    "You are a clinical information specialist. "
    "Expand the following medical query by adding relevant "
    "synonyms, related clinical terms, ICD codes if applicable, "
    "and alternative phrasings that would help retrieve "
    "evidence-based literature. "
    "Return only the enriched query string — no explanation.\n\n"
    "Original query: {query}"
)


class QueryEnricher:
    """Expands medical queries with synonyms and related concepts."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def enrich(self, query: str) -> str:
        """Enrich a query with medical synonyms and related terms.

        Returns the original query unchanged if it is not medical
        in nature or if the LLM call fails.

        Args:
            query: The original clinical question.

        Returns:
            Enriched query string, or original on failure.
        """
        try:
            is_medical = self._is_medical_query(query)
        except Exception:
            logger.warning(
                "Medical classification failed, returning original query",
                exc_info=True,
            )
            return query

        if not is_medical:
            logger.debug(
                "Non-medical query detected, skipping enrichment",
                extra={"query_len": len(query)},
            )
            return query

        try:
            enriched = self._expand_query(query)
            logger.info(
                "Query enriched",
                extra={
                    "original_len": len(query),
                    "enriched_len": len(enriched),
                },
            )
            return enriched
        except Exception:
            logger.warning(
                "Query enrichment failed, returning original query",
                exc_info=True,
            )
            return query

    def _is_medical_query(self, query: str) -> bool:
        """Classify the query as medical or general."""
        prompt = _MEDICAL_INDICATOR_PROMPT.format(query=query)
        result = self._llm.invoke(prompt)
        classification = str(result.content).strip().lower()
        return classification.startswith("medical")

    def _expand_query(self, query: str) -> str:
        """Use the LLM to expand the query with clinical terms."""
        prompt = _ENRICHMENT_PROMPT.format(query=query)
        result = self._llm.invoke(prompt)
        enriched = str(result.content).strip()
        if not enriched:
            return query
        return enriched
