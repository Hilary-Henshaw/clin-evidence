"""Hybrid Qdrant vector store for clinical evidence retrieval."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Hybrid vector store backed by Qdrant for clinical evidence."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._store: QdrantVectorStore | None = None
        self._collection: str = ""
        self._embedding_dim: int = 1536

    def initialise(
        self,
        embedding_fn: Embeddings,
        collection_name: str,
        embedding_dim: int,
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        """Connect to Qdrant and prepare the collection.

        Args:
            embedding_fn: Embedding function for dense vectors.
            collection_name: Qdrant collection name.
            embedding_dim: Embedding vector dimension.
            qdrant_path: Local path for file-based Qdrant.
            qdrant_url: Remote Qdrant URL (overrides path).
            qdrant_api_key: API key for remote Qdrant.
        """
        self._collection = collection_name
        self._embedding_dim = embedding_dim

        if qdrant_url:
            self._client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            logger.info(
                "Connected to remote Qdrant",
                extra={"url": qdrant_url},
            )
        else:
            path = qdrant_path or "./data/qdrant_db"
            self._client = QdrantClient(path=path)
            logger.info(
                "Using local Qdrant",
                extra={"path": path},
            )

        self._ensure_collection()

        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=self._collection,
            embedding=embedding_fn,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        logger.info(
            "KnowledgeStore initialised",
            extra={"collection": self._collection},
        )

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        assert self._client is not None
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Qdrant collection created",
                extra={"collection": self._collection},
            )

    def add_documents(
        self,
        chunks: list[str],
        metadata: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """Index document chunks into the vector store.

        Args:
            chunks: List of text chunks to index.
            metadata: Corresponding metadata for each chunk.

        Raises:
            RuntimeError: If the store has not been initialised.
            ValueError: If chunks and metadata lengths differ.
        """
        if self._store is None:
            raise RuntimeError(
                "KnowledgeStore not initialised. Call initialise() first."
            )
        if len(chunks) != len(metadata):
            raise ValueError(
                f"chunks ({len(chunks)}) and metadata "
                f"({len(metadata)}) must have equal length."
            )
        docs = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(chunks, metadata, strict=True)
        ]
        self._store.add_documents(docs)
        logger.info(
            "Documents indexed",
            extra={"count": len(docs)},
        )

    def hybrid_search(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top-k docs via hybrid BM25 + dense search.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of matching Document objects.

        Raises:
            RuntimeError: If the store has not been initialised.
        """
        if self._store is None:
            raise RuntimeError("KnowledgeStore not initialised.")
        results: list[Document] = list(
            self._store.similarity_search(query, k=top_k)
        )
        logger.debug(
            "Hybrid search complete",
            extra={
                "query_len": len(query),
                "hits": len(results),
            },
        )
        return results

    def collection_exists(self) -> bool:
        """Return True if the collection has been populated."""
        if self._client is None:
            return False
        existing = [c.name for c in self._client.get_collections().collections]
        return self._collection in existing
