"""Main RAG pipeline orchestrating the full knowledge base."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from typing_extensions import TypedDict

from clinevidence.agents.rag.answer_synthesizer import (
    AnswerSynthesizer,
)
from clinevidence.agents.rag.document_extractor import (
    DocumentExtractor,
)
from clinevidence.agents.rag.document_formatter import (
    DocumentFormatter,
)
from clinevidence.agents.rag.knowledge_store import KnowledgeStore
from clinevidence.agents.rag.query_enricher import QueryEnricher
from clinevidence.agents.rag.result_ranker import ResultRanker
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)


class IngestResult(TypedDict):
    """Result of a document ingestion operation."""

    path: str
    chunks_indexed: int
    images_processed: int
    elapsed_s: float


class QueryResult(TypedDict):
    """Result of a knowledge base query."""

    answer: str
    sources: list[str]
    confidence: float
    image_paths: list[Path]


class KnowledgeBase:
    """Orchestrates the full RAG pipeline for clinical evidence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = self._build_llm()
        self._embeddings = self._build_embeddings()
        self._extractor = DocumentExtractor()
        self._formatter = DocumentFormatter()
        self._store = KnowledgeStore()
        self._enricher = QueryEnricher(self._llm)
        self._ranker = ResultRanker()
        self._synthesizer = AnswerSynthesizer(self._llm)
        self._initialise_store()

    def _build_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=self._settings.llm_deployment,
            azure_endpoint=self._settings.azure_endpoint,
            api_key=self._settings.openai_api_key,
            api_version=self._settings.openai_api_version,
            temperature=self._settings.rag_temperature,
        )

    def _build_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            azure_deployment=(self._settings.embedding_deployment),
            azure_endpoint=(self._settings.embedding_azure_endpoint),
            api_key=self._settings.embedding_openai_api_key,
            api_version=(self._settings.embedding_openai_api_version),
        )

    def _initialise_store(self) -> None:
        """Initialise the KnowledgeStore with embedding function."""
        qdrant_api_key: str | None = None
        if self._settings.qdrant_api_key:
            qdrant_api_key = self._settings.qdrant_api_key.get_secret_value()
        self._store.initialise(
            embedding_fn=self._embeddings,
            collection_name=self._settings.kb_collection_name,
            embedding_dim=self._settings.kb_embedding_dim,
            qdrant_path=self._settings.kb_qdrant_path,
            qdrant_url=self._settings.qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )

    def ingest(self, source: Path) -> IngestResult:
        """Ingest a clinical document into the knowledge base.

        Pipeline: extract → format → chunk → embed → index.

        Args:
            source: Path to the document file.

        Returns:
            IngestResult with statistics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Ingestion started",
            extra={"source": str(source)},
        )

        # Step 1: Extract
        markdown, images = self._extractor.extract(source)

        # Step 2: Format (replace image placeholders)
        enriched_markdown = self._formatter.replace_image_placeholders(
            markdown, images, self._llm
        )

        # Step 3: Chunk
        chunks = self._formatter.semantic_chunk(
            enriched_markdown,
            self._llm,
            chunk_size=self._settings.kb_chunk_size,
            overlap=self._settings.kb_chunk_overlap,
        )

        # Step 4 & 5: Embed and index
        metadata = [
            {
                "title": source.stem,
                "source_file": str(source),
                "chunk_index": i,
                "image_paths": [str(p) for p in images],
            }
            for i in range(len(chunks))
        ]
        self._store.add_documents(chunks, metadata)

        elapsed = round(time.perf_counter() - t0, 2)
        logger.info(
            "Ingestion complete",
            extra={
                "source": str(source),
                "chunks": len(chunks),
                "images": len(images),
                "elapsed_s": elapsed,
            },
        )
        return IngestResult(
            path=str(source),
            chunks_indexed=len(chunks),
            images_processed=len(images),
            elapsed_s=elapsed,
        )

    def query(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> QueryResult:
        """Query the knowledge base for clinical evidence.

        Pipeline: enrich → retrieve → rerank → synthesize.

        Args:
            question: Clinical question from the clinician.
            chat_history: Conversation history.

        Returns:
            QueryResult with answer, sources, and confidence.
        """
        # Step 1: Enrich query
        enriched = self._enricher.enrich(question)

        # Step 2: Retrieve
        raw_docs = self._store.hybrid_search(
            enriched, top_k=self._settings.kb_retrieval_top_k
        )

        # Step 3: Rerank
        ranked_docs, image_paths = self._ranker.rerank(
            enriched,
            raw_docs,
            top_k=self._settings.kb_reranker_top_k,
        )

        # Step 4: Synthesize
        answer, sources, confidence = self._synthesizer.synthesize(
            question, ranked_docs, chat_history, image_paths
        )

        logger.info(
            "Query processed",
            extra={
                "confidence": round(confidence, 3),
                "sources": len(sources),
            },
        )
        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            image_paths=image_paths,
        )

    @property
    def is_ready(self) -> bool:
        """Return True if the knowledge base has documents."""
        return self._store.collection_exists()
