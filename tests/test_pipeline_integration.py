"""
Integration tests for the full chatbot pipeline.

These tests verify end-to-end routing logic:
  - Greeting → canned response (no RAG)
  - Complaint → empathy + RAG + escalation flag
  - Negative sentiment → empathy prefix
  - Standard query → RAG response
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from prompts.rag_prompt import EMPATHY_PREFIX


class TestPipelineRouting:
    """Test the routing logic of the pipeline without loading heavy models."""

    def _make_mock_pipeline(self):
        """Create a ChatPipeline with all components mocked."""
        from app.pipeline import ChatPipeline

        with patch.object(ChatPipeline, "__init__", lambda self: None):
            pipeline = ChatPipeline()

        # Mock all components
        pipeline.lang_detector = MagicMock()
        pipeline.sentiment_analyzer = MagicMock()
        pipeline.intent_classifier = MagicMock()
        pipeline.embedder = MagicMock()
        pipeline.vector_store = MagicMock()
        pipeline.retriever = MagicMock()
        pipeline.generator = MagicMock()

        # Default returns
        pipeline.lang_detector.predict.return_value = "en"
        pipeline.sentiment_analyzer.predict.return_value = {
            "sentiment": "neutral", "confidence": 0.9
        }
        pipeline.retriever.retrieve.return_value = [
            {"document": "test doc", "metadata": {"response": "test response"}}
        ]
        pipeline.generator.generate.return_value = "Here is your answer."

        return pipeline

    def test_greeting_returns_canned_response(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "greeting", "confidence": "high"
        }

        result = pipeline.process("Hello there!")

        assert result["intent"] == "greeting"
        assert result["escalate"] is False
        # Should NOT call RAG for greetings
        pipeline.retriever.retrieve.assert_not_called()
        pipeline.generator.generate.assert_not_called()

    def test_complaint_triggers_escalation(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "complaint", "confidence": "high"
        }

        result = pipeline.process("This service is terrible!")

        assert result["intent"] == "complaint"
        assert result["escalate"] is True
        assert "human agent" in result["response"].lower() or "follow up" in result["response"].lower()
        # Should still call RAG for factual content
        pipeline.retriever.retrieve.assert_called_once()
        pipeline.generator.generate.assert_called_once()

    def test_complaint_includes_empathy(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "complaint", "confidence": "high"
        }

        result = pipeline.process("I want to file a complaint!")

        assert EMPATHY_PREFIX in result["response"]

    def test_negative_sentiment_adds_empathy(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "order_status", "confidence": "high"
        }
        pipeline.sentiment_analyzer.predict.return_value = {
            "sentiment": "negative", "confidence": 0.85
        }

        result = pipeline.process("WHERE IS MY DAMN ORDER?!")

        assert result["intent"] == "order_status"
        assert result["escalate"] is False
        assert EMPATHY_PREFIX in result["response"]

    def test_standard_query_uses_rag(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "order_status", "confidence": "high"
        }

        result = pipeline.process("When will my order arrive?")

        assert result["intent"] == "order_status"
        assert result["escalate"] is False
        pipeline.retriever.retrieve.assert_called_once()
        pipeline.generator.generate.assert_called_once()

    def test_billing_query_uses_rag(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "billing_refunds", "confidence": "high"
        }

        result = pipeline.process("I need a refund")

        assert result["intent"] == "billing_refunds"
        pipeline.retriever.retrieve.assert_called_once()

    def test_account_query_uses_rag(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "account_management", "confidence": "high"
        }

        result = pipeline.process("How do I reset my password?")

        assert result["intent"] == "account_management"
        pipeline.retriever.retrieve.assert_called_once()

    def test_result_has_all_required_fields(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "order_status", "confidence": "high"
        }

        result = pipeline.process("Where is my order?")

        assert "response" in result
        assert "language" in result
        assert "sentiment" in result
        assert "intent" in result
        assert "escalate" in result

    def test_language_is_detected(self):
        pipeline = self._make_mock_pipeline()
        pipeline.lang_detector.predict.return_value = "es"
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "order_status", "confidence": "high"
        }

        result = pipeline.process("¿Dónde está mi pedido?")

        assert result["language"] == "es"

    def test_out_of_scope_uses_rag(self):
        pipeline = self._make_mock_pipeline()
        pipeline.intent_classifier.predict.return_value = {
            "intent_group": "out_of_scope", "confidence": "low"
        }

        result = pipeline.process("What's the weather like?")

        assert result["intent"] == "out_of_scope"
        # out_of_scope still goes through RAG — RAG will say it can't help
        pipeline.retriever.retrieve.assert_called_once()
