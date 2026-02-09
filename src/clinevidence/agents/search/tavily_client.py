"""Tavily web search client for real-time clinical evidence."""

from __future__ import annotations

import logging
from typing import Any

from tavily import TavilyClient

logger = logging.getLogger(__name__)


class TavilySearchClient:
    """Wraps the Tavily API for clinical evidence search."""

    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search for clinical evidence using Tavily.

        Args:
            query: Clinical search query.
            max_results: Maximum results to return.

        Returns:
            List of result dicts with keys:
            title, url, content, score.
        """
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
            )
            results: list[dict[str, Any]] = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                )
            logger.info(
                "Tavily search complete",
                extra={
                    "query_len": len(query),
                    "results": len(results),
                },
            )
            return results
        except Exception:
            logger.warning(
                "Tavily search failed",
                exc_info=True,
            )
            return []
