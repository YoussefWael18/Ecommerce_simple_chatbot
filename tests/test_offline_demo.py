"""No-key behavior used only for the transparent offline demonstration."""

from unittest.mock import patch

from app.models.intent_classifier import IntentClassifier
from app.rag.generator import Generator


def test_generator_without_key_returns_labeled_retrieved_excerpt():
    generator = Generator(api_key="offline", model_name="test")
    generator.api_key = ""
    chunks = [{"document": "Refunds take five days.",
               "metadata": {"source": "refund.pdf", "page": 2}}]
    with patch.object(generator.client.chat.completions, "create") as call:
        answer = generator.generate("When is my refund?", "neutral", chunks)
    assert "refund.pdf - page 2" in answer
    assert "Refunds take five days." in answer
    call.assert_not_called()


def test_intent_without_key_skips_network():
    classifier = IntentClassifier(api_key="offline", model_name="test")
    classifier.api_key = ""
    with patch.object(classifier.client.chat.completions, "create") as call:
        result = classifier.predict("Where is my order?")
    assert result["intent_group"] == "out_of_scope"
    call.assert_not_called()
