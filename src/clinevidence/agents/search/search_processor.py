"""Evidence search processor: search + LLM synthesis."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI

from clinevidence.agents.search.evidence_searcher import (
    EvidenceSearcher,
)
from clinevidence.agents.search.pubmed_client import (
    PubMedSearchClient,
)
from clinevidence.agents.search.tavily_client import (
    TavilySearchClient,
)
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = """\
You are ClinEvidence, a clinical AI assistant. You have been
given real-time web and PubMed search results for a clinical
question. Synthesise a medically accurate, evidence-based answer.

Guidelines:
- Cite sources by their title and URL: [Title](URL)
- Do not invent facts not present in the search results.
- Be concise but clinically thorough.
- Always include the disclaimer below.

> **Disclaimer**: This information is synthesised from publicly
> available sources and AI search results. It does not replace
> clinical judgement. Always consult qualified clinicians for
> patient-specific decisions.
"""

_QUERY_BUILD_PROMPT = """\
You are a clinical search specialist. Given the following
conversation history and user question, generate an optimised
search query for medical literature databases. Return ONLY the
query string, no explanation.

Recent conversation:
{history}

User question: {question}
"""


class EvidenceSearchProcessor:
    """Processes clinical queries using web evidence and LLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = self._build_llm()
        self._searcher = self._build_searcher()

    def _build_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=self._settings.llm_deployment,
            azure_endpoint=self._settings.azure_endpoint,
            api_key=self._settings.openai_api_key,
            api_version=self._settings.openai_api_version,
            temperature=self._settings.search_temperature,
        )

    def _build_searcher(self) -> EvidenceSearcher:
        tavily = TavilySearchClient(
            api_key=(self._settings.tavily_api_key.get_secret_value())
        )
        pubmed = PubMedSearchClient()
        return EvidenceSearcher(tavily, pubmed)

    def process(
        self,
        query: str,
        chat_history: list[BaseMessage],
    ) -> tuple[str, list[str]]:
        """Process a query via evidence search and synthesis.

        Args:
            query: The clinical question.
            chat_history: Prior conversation messages.

        Returns:
            Tuple of (synthesised_answer, list_of_source_urls).
        """
        search_query = self._build_search_query(query, chat_history)
        results = self._searcher.search(
            search_query,
            max_results=self._settings.search_max_results,
        )

        if not results:
            no_result = (
                "No clinical evidence was found for this query "
                "in available web and PubMed sources.\n\n"
                "> **Disclaimer**: Always consult qualified "
                "clinicians for patient-specific decisions."
            )
            return no_result, []

        context = self._format_results(results)
        source_urls = [r["url"] for r in results if r.get("url")]

        messages: list[Any] = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Clinical question: {query}\n\nSearch results:\n{context}"
                ),
            },
        ]

        try:
            response = self._llm.invoke(messages)
            answer = str(response.content).strip()
        except Exception:
            logger.error("Search synthesis failed", exc_info=True)
            answer = (
                "An error occurred synthesising search results. "
                "Please review the sources directly.\n\n"
                "> **Disclaimer**: Always consult qualified "
                "clinicians for patient-specific decisions."
            )

        logger.info(
            "Evidence search processed",
            extra={
                "results": len(results),
                "answer_len": len(answer),
            },
        )
        return answer, source_urls

    def _build_search_query(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> str:
        """Build an optimised search query from context."""
        if not chat_history:
            return question

        def _role(msg: object) -> str:
            cls = msg.__class__.__name__
            return "User" if cls == "HumanMessage" else "Assistant"

        history_text = "\n".join(
            f"{_role(m)}: {str(m.content)[:200]}"  # type: ignore[union-attr]
            for m in chat_history[-4:]
        )
        prompt = _QUERY_BUILD_PROMPT.format(
            history=history_text, question=question
        )
        try:
            result = self._llm.invoke(prompt)
            refined = str(result.content).strip()
            if refined:
                return refined
        except Exception:
            logger.warning(
                "Search query refinement failed",
                exc_info=True,
            )
        return question

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results as numbered context blocks."""
        parts: list[str] = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", f"Source {i}")
            url = item.get("url", "")
            content = item.get("content", "")
            header = f"[{i}] {title}"
            if url:
                header += f" | {url}"
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)
