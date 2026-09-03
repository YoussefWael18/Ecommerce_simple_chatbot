"""
Tests for the FastAPI endpoints.

These tests verify:
  - /health endpoint returns correct structure
  - /chat endpoint processes messages correctly
  - /chat endpoint rejects empty messages
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _get_test_client():
    """Create a TestClient with a mocked pipeline."""
    from app.pipeline import ChatPipeline

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.count.return_value = 100
    mock_pipeline.process.return_value = {
        "response": "Your order is on the way!",
        "language": "en",
        "sentiment": "neutral",
        "intent": "order_status",
        "escalate": False,
    }

    with patch("app.main.ChatPipeline", return_value=mock_pipeline):
        from app.main import app
        # Set the pipeline directly
        import app.main as main_module
        main_module.pipeline = mock_pipeline
        client = TestClient(app, raise_server_exceptions=False)
        yield client, mock_pipeline
        main_module.pipeline = None


@pytest.fixture
def client_and_pipeline():
    yield from _get_test_client()


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_ok(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_lists_models(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.get("/health")
        data = response.json()
        assert len(data["models_loaded"]) > 0

    def test_health_shows_document_count(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.get("/health")
        data = response.json()
        assert data["vector_store_documents"] == 100


class TestChatEndpoint:
    """Test the /chat endpoint."""

    def test_chat_valid_message(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.post(
            "/chat",
            json={"message": "Where is my order?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "language" in data
        assert "sentiment" in data
        assert "intent" in data
        assert "escalate" in data

    def test_chat_empty_message_rejected(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.post(
            "/chat",
            json={"message": ""},
        )
        assert response.status_code == 400

    def test_chat_whitespace_message_rejected(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.post(
            "/chat",
            json={"message": "   "},
        )
        assert response.status_code == 400

    def test_chat_calls_pipeline_process(self, client_and_pipeline):
        client, mock_pipeline = client_and_pipeline
        client.post("/chat", json={"message": "Hello!"})
        mock_pipeline.process.assert_called_once_with("Hello!")

    def test_chat_returns_correct_structure(self, client_and_pipeline):
        client, _ = client_and_pipeline
        response = client.post(
            "/chat",
            json={"message": "I need help with my order"},
        )
        data = response.json()
        assert isinstance(data["response"], str)
        assert isinstance(data["language"], str)
        assert isinstance(data["sentiment"], str)
        assert isinstance(data["intent"], str)
        assert isinstance(data["escalate"], bool)
