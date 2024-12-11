"""Clinical answer synthesis from retrieved evidence."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are ClinEvidence, an AI clinical decision-support assistant
for ICU teams. Your role is to synthesise evidence-based answers
from retrieved clinical literature.

Guidelines:
- Base your answer ONLY on the provided context documents.
- Cite sources explicitly using [Source N] notation.
- Format data, comparisons, or measurements in markdown tables.
- Never fabricate citations, URLs, or statistics not in context.
- Always conclude with a medical disclaimer.
- Be concise but clinically complete.

DISCLAIMER TEMPLATE (always include):
> **Disclaimer**: This information is AI-generated from clinical
> literature and is intended to support, not replace, clinical
> judgement. Always verify with current guidelines and consult
> qualified clinicians for patient-specific decisions.
"""

_USER_PROMPT_TEMPLATE = """\
Clinical Question: {question}

Retrieved Context:
{context}

Provide a thorough, evidence-based answer with source citations.
"""


class AnswerSynthesizer:
    """Synthesises clinical answers from retrieved documents."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def synthesize(
        self,
        query: str,
        context_docs: list[Document],
        chat_history: list[BaseMessage],
        image_paths: list[Path],
    ) -> tuple[str, list[str], float]:
        """Synthesise a clinical answer from retrieved evidence.

        Args:
            query: The clinical question to answer.
            context_docs: Retrieved and reranked documents.
            chat_history: Prior conversation messages.
            image_paths: Image files associated with context.

        Returns:
            Tuple of (answer_text, source_titles, confidence).
        """
        if not context_docs:
            fallback = (
                "No relevant clinical evidence was found in "
                "the knowledge base for this query.\n\n"
                "> **Disclaimer**: Please consult current "
                "clinical guidelines and qualified clinicians."
            )
            return fallback, [], 0.0

        context_str = self._build_context(context_docs)
        source_titles = self._extract_titles(context_docs)
        confidence = self._compute_confidence(query, context_docs)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        for hist_msg in chat_history[-6:]:
            role = (
                "user"
                if hist_msg.__class__.__name__ == "HumanMessage"
                else "assistant"
            )
            messages.append({"role": role, "content": str(hist_msg.content)})

        user_content = _USER_PROMPT_TEMPLATE.format(
            question=query, context=context_str
        )
        messages.append({"role": "user", "content": user_content})

        try:
            result = self._llm.invoke(messages)
            answer = str(result.content).strip()
        except Exception:
            logger.error("Answer synthesis failed", exc_info=True)
            answer = (
                "An error occurred synthesising the answer. "
                "Please retry or consult clinical resources "
                "directly.\n\n"
                "> **Disclaimer**: Always consult qualified "
                "clinicians for patient-specific decisions."
            )

        logger.info(
            "Answer synthesised",
            extra={
                "source_count": len(source_titles),
                "confidence": round(confidence, 3),
                "answer_len": len(answer),
            },
        )
        return answer, source_titles, confidence

    def _build_context(self, docs: list[Document]) -> str:
        """Format documents as numbered context blocks."""
        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", f"Document {i}")
            parts.append(f"[Source {i}] {title}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_titles(self, docs: list[Document]) -> list[str]:
        """Extract titles from document metadata."""
        titles: list[str] = []
        for i, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", f"Document {i}")
            url = doc.metadata.get("url", "")
            entry = str(title)
            if url:
                entry = f"{title} ({url})"
            titles.append(entry)
        return titles

    def _compute_confidence(self, query: str, docs: list[Document]) -> float:
        """Estimate confidence as query-term coverage in context."""
        if not docs:
            return 0.0
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
        combined = " ".join(doc.page_content.lower() for doc in docs)
        matched = sum(1 for term in query_terms if term in combined)
        return round(matched / len(query_terms), 3)
