"""
Tests for the RAG pipeline (embedder, vector store, retriever, generator).

These tests verify:
  - Embedder produces correct-dimension vectors
  - VectorStore can add and search documents
  - Retriever chains embed + search correctly
  - Generator produces responses (mocked LLM)
  - Prompt formatting is correct
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from prompts.rag_prompt import RAG_SYSTEM_PROMPT, EMPATHY_PREFIX, format_rag_prompt


# ── Prompt Tests ───────────────────────────────────────────────────────────────

class TestRAGPrompt:
    """Test the RAG prompt template."""

    def test_system_prompt_not_empty(self):
        assert RAG_SYSTEM_PROMPT, "RAG system prompt should not be empty"

    def test_empathy_prefix_not_empty(self):
        assert EMPATHY_PREFIX, "Empathy prefix should not be empty"

    def test_format_rag_prompt_returns_dict(self):
        chunks = [
            {"document": "How to track order", "metadata": {"response": "You can track..."}},
        ]
        result = format_rag_prompt("Where is my order?", "neutral", chunks)
        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result

    def test_format_rag_prompt_includes_sentiment(self):
        chunks = [
            {"document": "Refund policy", "metadata": {"response": "Refunds are processed..."}},
        ]
        result = format_rag_prompt("I want a refund", "negative", chunks)
        assert "negative" in result["system"]

    def test_format_rag_prompt_includes_context(self):
        chunks = [
            {"document": "Delivery info", "metadata": {"response": "Delivery takes 3-5 days"}},
        ]
        result = format_rag_prompt("When will it arrive?", "neutral", chunks)
        assert "Delivery takes 3-5 days" in result["system"]

    def test_format_rag_prompt_includes_user_message(self):
        chunks = [{"document": "test", "metadata": {"response": "test response"}}]
        result = format_rag_prompt("My specific question", "neutral", chunks)
        assert "My specific question" in result["user"]

    def test_format_rag_prompt_handles_empty_chunks(self):
        result = format_rag_prompt("test question", "neutral", [])
        assert isinstance(result, dict)
        assert "system" in result


# ── Embedder Tests ─────────────────────────────────────────────────────────────

class TestEmbedder:
    """Test the sentence-transformer embedder."""

    @pytest.fixture(scope="class")
    def embedder(self):
        from app.rag.embedder import Embedder
        return Embedder(model_name="all-MiniLM-L6-v2")

    def test_embed_returns_list(self, embedder):
        result = embedder.embed(["Hello world"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_embed_correct_dimensions(self, embedder):
        result = embedder.embed(["Hello world"])
        # all-MiniLM-L6-v2 produces 384-dim vectors
        assert len(result[0]) == 384

    def test_embed_multiple_texts(self, embedder):
        texts = ["Hello", "World", "Test"]
        result = embedder.embed(texts)
        assert len(result) == 3

    def test_embed_single(self, embedder):
        result = embedder.embed_single("Hello world")
        assert isinstance(result, list)
        assert len(result) == 384


# ── VectorStore Tests ──────────────────────────────────────────────────────────

class TestVectorStore:
    """Test ChromaDB vector store with a temporary collection."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        from app.rag.vector_store import VectorStore
        return VectorStore(
            persist_directory=str(tmp_path / "test_chroma"),
            collection_name="test_collection",
        )

    @pytest.fixture
    def embedder(self):
        from app.rag.embedder import Embedder
        return Embedder(model_name="all-MiniLM-L6-v2")

    def test_empty_store_count(self, temp_store):
        assert temp_store.count() == 0

    def test_add_and_count(self, temp_store, embedder):
        docs = ["How to track order", "Refund policy"]
        embeddings = embedder.embed(docs)
        temp_store.add_documents(
            ids=["1", "2"],
            documents=docs,
            embeddings=embeddings,
            metadatas=[{"intent": "track_order"}, {"intent": "get_refund"}],
        )
        assert temp_store.count() == 2

    def test_search_returns_results(self, temp_store, embedder):
        docs = ["How to track your order status", "How to get a refund"]
        embeddings = embedder.embed(docs)
        temp_store.add_documents(
            ids=["1", "2"],
            documents=docs,
            embeddings=embeddings,
            metadatas=[
                {"intent": "track_order", "response": "Track at..."},
                {"intent": "get_refund", "response": "Refund in 5 days"},
            ],
        )
        query_emb = embedder.embed_single("Where is my package?")
        results = temp_store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert "document" in results[0]
        assert "metadata" in results[0]

    def test_search_relevance(self, temp_store, embedder):
        """The most relevant doc should be returned first."""
        docs = [
            "You can track your order using the tracking number",
            "To change your password, go to account settings",
        ]
        embeddings = embedder.embed(docs)
        temp_store.add_documents(
            ids=["1", "2"],
            documents=docs,
            embeddings=embeddings,
            metadatas=[{"intent": "track_order"}, {"intent": "recover_password"}],
        )
        query_emb = embedder.embed_single("Where is my order?")
        results = temp_store.search(query_emb, top_k=1)
        assert "track" in results[0]["document"].lower()


# ── Retriever Tests ────────────────────────────────────────────────────────────

class TestRetriever:
    """Test the retriever chains embedder + vector store."""

    @pytest.fixture
    def retriever_setup(self, tmp_path):
        from app.rag.embedder import Embedder
        from app.rag.vector_store import VectorStore
        from app.rag.retriever import Retriever

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        store = VectorStore(
            persist_directory=str(tmp_path / "test_chroma"),
            collection_name="test_retriever",
        )
        docs = ["Track your order with tracking number", "Get a refund within 30 days"]
        embeddings = embedder.embed(docs)
        store.add_documents(
            ids=["1", "2"],
            documents=docs,
            embeddings=embeddings,
            metadatas=[
                {"intent": "track_order", "response": "Use tracking ID", "category": "ORDER"},
                {"intent": "get_refund", "response": "Refund in 5 days", "category": "REFUND"},
            ],
        )
        return Retriever(embedder, store)

    def test_retrieve_returns_list(self, retriever_setup):
        results = retriever_setup.retrieve("Where is my order?", top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_retrieve_results_have_metadata(self, retriever_setup):
        results = retriever_setup.retrieve("How to get refund?", top_k=1)
        assert len(results) >= 1
        assert "metadata" in results[0]


# ── Generator Tests (mocked) ──────────────────────────────────────────────────

class TestGenerator:
    """Test the LLM generator with mocked API calls."""

    @pytest.fixture
    def generator(self):
        from app.rag.generator import Generator
        return Generator(api_key="test-key", model_name="test-model")

    def _mock_response(self, content: str):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_generate_returns_string(self, generator):
        mock_resp = self._mock_response("Your order is on the way!")
        chunks = [{"document": "Order tracking", "metadata": {"response": "Track at..."}}]
        with patch.object(generator.client.chat.completions, "create", return_value=mock_resp):
            result = generator.generate("Where is my order?", "neutral", chunks)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_generate_api_error_fallback(self, generator):
        chunks = [{"document": "test", "metadata": {"response": "test"}}]
        with patch.object(
            generator.client.chat.completions, "create",
            side_effect=Exception("API Error")
        ):
            result = generator.generate("test", "neutral", chunks)
            assert "apologize" in result.lower() or "sorry" in result.lower()
