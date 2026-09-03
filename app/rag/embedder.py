"""
Embedding module using sentence-transformers for generating dense vector representations.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a SentenceTransformer model for embedding text into dense vectors.
    Default model: all-MiniLM-L6-v2 (384-dimensional).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the Embedder.

        Args:
            model_name: Name or path of the sentence-transformer model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display a progress bar.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: A single string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed([text])[0]
