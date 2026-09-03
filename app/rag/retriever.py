"""
Retriever module orchestrating embedding of queries and vector store lookups.
"""

from __future__ import annotations

from typing import Any
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """
    Coordinates embedding user queries and searching the vector store for relevant context chunks.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        """
        Initialize the Retriever.

        Args:
            embedder: Embedder instance for query vectorization.
            vector_store: VectorStore instance holding the documents.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Retrieve relevant knowledge base chunks for a customer query.

        Args:
            query: The user query string.
            top_k: Number of results to return (default 3).

        Returns:
            List of dicts with keys:
                - 'document': Original instruction text
                - 'response': Support response from metadata
                - 'intent': Intent category from metadata
                - 'category': Category from metadata
                - 'distance': Cosine distance from query embedding
                - 'metadata': Full metadata dict
        """
        query_embedding = self.embedder.embed_single(query)
        search_results = self.vector_store.search(query_embedding, top_k=top_k)

        formatted_results: list[dict[str, Any]] = []
        for item in search_results:
            metadata = item.get("metadata") or {}
            formatted_results.append({
                "document": item.get("document", ""),
                "response": metadata.get("response", ""),
                "intent": metadata.get("intent", ""),
                "category": metadata.get("category", ""),
                "distance": item.get("distance"),
                "metadata": metadata,
            })
        return formatted_results

