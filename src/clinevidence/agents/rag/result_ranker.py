"""Cross-encoder reranking for retrieved clinical evidence."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-6"


class ResultRanker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self) -> None:
        self._encoder: CrossEncoder | None = None

    def _get_encoder(self) -> CrossEncoder:  # type: ignore[type-arg]
        """Lazy-load the cross-encoder model."""
        if self._encoder is None:
            logger.info(
                "Loading cross-encoder model",
                extra={"model": _CROSS_ENCODER_MODEL},
            )
            self._encoder = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512)
        return self._encoder

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> tuple[list[Document], list[Path]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The clinical question for scoring.
            documents: Retrieved documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Tuple of (top_k_documents, referenced_image_paths).
        """
        if not documents:
            logger.debug("No documents to rerank")
            return [], []

        encoder = self._get_encoder()
        pairs = [[query, doc.page_content] for doc in documents]
        scores: list[float] = encoder.predict(pairs).tolist()

        scored = sorted(
            zip(scores, documents, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        top_docs = [doc for _, doc in scored[:top_k]]
        top_scores = [score for score, _ in scored[:top_k]]

        logger.info(
            "Reranking complete",
            extra={
                "input_count": len(documents),
                "output_count": len(top_docs),
                "top_score": (round(top_scores[0], 4) if top_scores else None),
            },
        )

        image_paths = self._extract_image_paths(top_docs)
        return top_docs, image_paths

    def _extract_image_paths(self, documents: list[Document]) -> list[Path]:
        """Extract image file paths referenced in doc metadata."""
        paths: list[Path] = []
        for doc in documents:
            raw = doc.metadata.get("image_paths", [])
            if isinstance(raw, list):
                for entry in raw:
                    p = Path(str(entry))
                    if p.exists():
                        paths.append(p)
            elif isinstance(raw, str) and raw:
                p = Path(raw)
                if p.exists():
                    paths.append(p)
        return paths
