"""
ChromaDB vector store module for persistent document storage and similarity search.
"""

from __future__ import annotations

from typing import Any

import chromadb


class VectorStore:
    """
    Wraps a ChromaDB persistent client for storing and searching document embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "customer_support",
    ) -> None:
        """
        Initialize the VectorStore.

        Args:
            persist_directory: Path to ChromaDB persistent storage directory.
            collection_name: Name of the ChromaDB collection.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add documents to the collection in batches.

        Args:
            ids: Unique identifiers for each document.
            documents: Document text strings.
            embeddings: Pre-computed embedding vectors.
            metadatas: Optional metadata dicts for each document.
        """
        batch_size = 5000  # ChromaDB recommended batch limit
        total = len(ids)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_embs = embeddings[start:end]
            batch_meta = metadatas[start:end] if metadatas else None

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_meta,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar documents to a query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: 'document', 'metadata', 'distance'.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        items: list[dict[str, Any]] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                items.append({
                    "document": doc,
                    "metadata": meta or {},
                    "distance": dist,
                })

        return items

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
