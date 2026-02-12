"""Combined evidence searcher: Tavily + PubMed with dedup."""

from __future__ import annotations

import logging
from typing import Any

from clinevidence.agents.search.pubmed_client import (
    PubMedSearchClient,
)
from clinevidence.agents.search.tavily_client import (
    TavilySearchClient,
)

logger = logging.getLogger(__name__)

_TAVILY_THRESHOLD = 2  # Fallback to PubMed if fewer results


class EvidenceSearcher:
    """Combines Tavily web search and PubMed literature search."""

    def __init__(
        self,
        tavily_client: TavilySearchClient,
        pubmed_client: PubMedSearchClient,
    ) -> None:
        self._tavily = tavily_client
        self._pubmed = pubmed_client

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search for clinical evidence across web and PubMed.

        Runs Tavily first; if fewer than threshold results are
        returned, also queries PubMed. Deduplicates by URL.

        Args:
            query: Clinical search query.
            max_results: Maximum results per source.

        Returns:
            Deduplicated list of evidence results.
        """
        logger.info(
            "Evidence search started",
            extra={"query_len": len(query)},
        )

        tavily_results = self._tavily.search(query, max_results=max_results)
        logger.debug(
            "Tavily returned results",
            extra={"count": len(tavily_results)},
        )

        combined = list(tavily_results)

        if len(tavily_results) < _TAVILY_THRESHOLD:
            logger.info(
                "Tavily results below threshold, querying PubMed",
                extra={
                    "tavily_count": len(tavily_results),
                    "threshold": _TAVILY_THRESHOLD,
                },
            )
            pubmed_results = self._pubmed.search(
                query, max_results=max_results
            )
            combined.extend(pubmed_results)
            logger.debug(
                "PubMed returned results",
                extra={"count": len(pubmed_results)},
            )

        deduplicated = self._deduplicate(combined)
        logger.info(
            "Evidence search complete",
            extra={
                "raw_count": len(combined),
                "dedup_count": len(deduplicated),
            },
        )
        return deduplicated

    def _deduplicate(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate results by URL."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in results:
            url = item.get("url", "")
            if url and url in seen:
                continue
            seen.add(url)
            unique.append(item)
        return unique
