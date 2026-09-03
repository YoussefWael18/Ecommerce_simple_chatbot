"""
Tests for the intent classifier (few-shot via OpenRouter).

These tests verify:
  - The intent prompt is well-formed
  - The classifier can parse valid LLM responses
  - All 7 intent groups are recognized
  - Edge cases are handled gracefully
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from prompts.intent_prompt import INTENT_SYSTEM_PROMPT, INTENT_CATEGORIES, INTENT_CATEGORY_DESCRIPTIONS
from app.models.intent_classifier import IntentClassifier


# ── Prompt Tests ───────────────────────────────────────────────────────────────

class TestIntentPrompt:
    """Test the intent classification prompt is well-structured."""

    def test_system_prompt_not_empty(self):
        assert INTENT_SYSTEM_PROMPT, "System prompt should not be empty"

    def test_system_prompt_mentions_all_categories(self):
        for category in INTENT_CATEGORIES:
            assert category in INTENT_SYSTEM_PROMPT, (
                f"System prompt should mention category '{category}'"
            )

    def test_system_prompt_mentions_json(self):
        assert "JSON" in INTENT_SYSTEM_PROMPT or "json" in INTENT_SYSTEM_PROMPT, (
            "System prompt should instruct JSON output format"
        )

    def test_categories_have_7_groups(self):
        assert len(INTENT_CATEGORIES) == 7, "Should have exactly 7 intent groups"

    def test_category_descriptions_match_categories(self):
        assert set(INTENT_CATEGORIES.keys()) == set(INTENT_CATEGORY_DESCRIPTIONS.keys()), (
            "Category descriptions should match category keys"
        )

    def test_expected_categories_exist(self):
        expected = [
            "greeting", "order_status", "order_management",
            "billing_refunds", "account_management", "complaint", "out_of_scope"
        ]
        for cat in expected:
            assert cat in INTENT_CATEGORIES, f"Missing expected category: {cat}"


# ── Classifier Tests (mocked API) ─────────────────────────────────────────────

class TestIntentClassifier:
    """Test the IntentClassifier with mocked OpenRouter responses."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier(api_key="test-key", model_name="test-model")

    def _mock_response(self, content: str):
        """Create a mock OpenAI-style response."""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_predict_greeting(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "greeting", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("Hello there!")
            assert result["intent_group"] == "greeting"
            assert result["confidence"] == "high"

    def test_predict_order_status(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "order_status", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("Where is my order?")
            assert result["intent_group"] == "order_status"

    def test_predict_complaint(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "complaint", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("This is terrible service!")
            assert result["intent_group"] == "complaint"

    def test_predict_billing_refunds(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "billing_refunds", "confidence": "medium"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("I need a refund for my order")
            assert result["intent_group"] == "billing_refunds"

    def test_predict_account_management(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "account_management", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("I forgot my password")
            assert result["intent_group"] == "account_management"

    def test_predict_order_management(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "order_management", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("I want to cancel my order")
            assert result["intent_group"] == "order_management"

    def test_predict_out_of_scope(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "out_of_scope", "confidence": "low"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("What is the meaning of life?")
            assert result["intent_group"] == "out_of_scope"

    def test_malformed_json_fallback(self, classifier):
        """If the LLM returns garbage, classifier should fallback gracefully."""
        mock_resp = self._mock_response("Sorry, I can't understand your request.")
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("test message")
            assert result["intent_group"] == "out_of_scope"
            assert result["confidence"] == "low"

    def test_api_error_fallback(self, classifier):
        """If the API call fails, classifier should return a safe fallback."""
        with patch.object(
            classifier.client.chat.completions, "create",
            side_effect=Exception("API Error")
        ):
            result = classifier.predict("test message")
            assert result["intent_group"] == "out_of_scope"
            assert result["confidence"] == "low"

    def test_result_has_required_keys(self, classifier):
        mock_resp = self._mock_response('{"intent_group": "greeting", "sentiment": "positive", "confidence": "high"}')
        with patch.object(classifier.client.chat.completions, "create", return_value=mock_resp):
            result = classifier.predict("Hi")
            assert "intent_group" in result
            assert "sentiment" in result
            assert "confidence" in result
            assert result["sentiment"] == "positive"
