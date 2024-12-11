"""Markdown enrichment and semantic chunking for clinical docs."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(
    r"!\[.*?\]\(.*?\)|<!-- image: .*? -->",
    re.IGNORECASE,
)


class DocumentFormatter:
    """Enriches raw markdown with image summaries and chunks it."""

    def replace_image_placeholders(
        self,
        markdown: str,
        images: list[Path],
        llm: AzureChatOpenAI,
    ) -> str:
        """Replace image markers with GPT-4o generated summaries.

        Args:
            markdown: Raw markdown from document extraction.
            images: Extracted image paths to summarise.
            llm: AzureChatOpenAI instance for vision tasks.

        Returns:
            Enriched markdown with figure summaries inline.
        """
        if not images:
            return markdown

        summaries: list[str] = []
        for img_path in images:
            try:
                summary = self._summarise_image(img_path, llm)
                summaries.append(f"\n\n**[Figure Summary]**: {summary}\n\n")
            except Exception:
                logger.warning(
                    "Image summarisation failed",
                    extra={"image": str(img_path)},
                    exc_info=True,
                )
                summaries.append("\n\n**[Figure Summary]**: (unavailable)\n\n")

        placeholder_count = len(_PLACEHOLDER_RE.findall(markdown))
        idx = 0

        def _replacer(match: re.Match[str]) -> str:
            nonlocal idx
            if idx < len(summaries):
                replacement = summaries[idx]
                idx += 1
                return replacement
            return match.group(0)

        enriched = _PLACEHOLDER_RE.sub(_replacer, markdown)
        logger.debug(
            "Image placeholders replaced",
            extra={
                "placeholders": placeholder_count,
                "summaries": len(summaries),
            },
        )
        return enriched

    def _summarise_image(self, img_path: Path, llm: AzureChatOpenAI) -> str:
        """Generate a clinical summary of a medical image."""
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        with img_path.open("rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        suffix = img_path.suffix.lstrip(".").lower()
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"

        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are a clinical document analyst. "
                        "Describe this figure from a medical "
                        "document concisely (2-4 sentences), "
                        "focusing on clinically relevant content "
                        "such as anatomy, pathology, measurements,"
                        " or data trends visible in the image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "low",
                    },
                },
            ]
        )
        response = llm.invoke([msg])
        return str(response.content).strip()

    def semantic_chunk(
        self,
        text: str,
        llm: AzureChatOpenAI,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into semantically coherent chunks.

        Uses an LLM to identify optimal split points, then
        applies character-level overlap for context continuity.

        Args:
            text: Document text to chunk.
            llm: AzureChatOpenAI for semantic splitting.
            chunk_size: Target characters per chunk.
            overlap: Overlap characters between chunks.

        Returns:
            List of text chunks with applied overlap.
        """
        if len(text) <= chunk_size:
            return [text]

        prompt = (
            f"You are chunking a clinical document. "
            f"Split the following text into coherent sections "
            f"of approximately {chunk_size} characters each. "
            f"Split at paragraph or section boundaries, not "
            f"mid-sentence. Return each chunk separated by "
            f"exactly '---CHUNK---'.\n\n"
            f"{text[:12000]}"
        )
        try:
            result = llm.invoke(prompt)
            raw = str(result.content)
            parts = [p.strip() for p in raw.split("---CHUNK---") if p.strip()]
            if not parts:
                raise ValueError("LLM returned empty chunks")
        except Exception:
            logger.warning(
                "LLM chunking failed, falling back to character split",
                exc_info=True,
            )
            parts = self._character_chunk(text, chunk_size)

        chunks: list[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                chunks.append(part)
            else:
                tail = chunks[-1][-overlap:] if overlap else ""
                chunks.append(tail + part)

        logger.debug(
            "Document chunked",
            extra={"chunk_count": len(chunks)},
        )
        return chunks

    def _character_chunk(self, text: str, chunk_size: int) -> list[str]:
        """Fallback character-level chunking."""
        return [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
