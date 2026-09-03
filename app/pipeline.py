"""
Integrated chatbot pipeline.
Orchestrates Language Detection → Sentiment → Intent Classification → Routing → Response.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for prompt imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import (
    LANG_CLASSIFIER_PATH,
    ROBERTA_SENTIMENT_PATH,
    ROBERTA_TOKENIZER_PATH,
    BILSTM_SENTIMENT_PATH,
    OPENROUTER_API_KEY,
    LLM_MODEL_NAME,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RAG_TOP_K,
)
from app.models.language_detector import LanguageDetector
from app.models.sentiment import SentimentAnalyzer
from app.models.intent_classifier import IntentClassifier
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.generator import Generator
from prompts.greeting_responses import get_greeting_response
from prompts.rag_prompt import EMPATHY_PREFIX


class ChatPipeline:
    """
    End-to-end pipeline for processing customer messages.
    
    Flow:
      1. Language Detection → detected_language
      2. Sentiment Analysis → detected_sentiment
      3. Intent Classification (few-shot via OpenRouter) → intent_group
      4. Routing:
         - greeting → canned response (no RAG)
         - complaint → empathy + RAG + escalation flag
         - negative sentiment → empathy prefix + RAG
         - all others → standard RAG
    """

    def __init__(self):
        """Initialize all pipeline components."""
        print("Loading pipeline components...")

        # 1. Language Detector
        self.lang_detector = LanguageDetector(str(LANG_CLASSIFIER_PATH))
        print("  ✓ Language detector loaded")

        # 2. Sentiment Analyzer (RoBERTa)
        self.sentiment_analyzer = SentimentAnalyzer(
            roberta_model_path=str(ROBERTA_SENTIMENT_PATH),
            tokenizer_path=str(ROBERTA_TOKENIZER_PATH),
            bilstm_model_path=str(BILSTM_SENTIMENT_PATH),
        )
        print("  ✓ Sentiment analyzer loaded")

        # 3. Intent Classifier (few-shot via OpenRouter)
        self.intent_classifier = IntentClassifier(
            api_key=OPENROUTER_API_KEY,
            model_name=LLM_MODEL_NAME,
        )
        print("  ✓ Intent classifier ready")

        # 4. RAG components
        self.embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = VectorStore(
            persist_directory=str(CHROMA_DIR),
            collection_name=CHROMA_COLLECTION_NAME,
        )
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.generator = Generator(
            api_key=OPENROUTER_API_KEY,
            model_name=LLM_MODEL_NAME,
        )
        print("  ✓ RAG pipeline ready")
        print(f"  ✓ Vector store has {self.vector_store.count()} documents")
        print("Pipeline fully loaded!")

    def process(self, message: str) -> dict:
        """
        Process a customer message through the full pipeline.

        Args:
            message: The customer's input message.

        Returns:
            dict with keys: response, language, sentiment, intent, escalate
        """
        # ── Stage 1: Language Detection ────────────────────────────────────
        language = self.lang_detector.predict(message)

        # ── Stage 2 & 3: Intent & Emotion Classification (LLM Few-Shot) ────
        intent_result = self.intent_classifier.predict(message)
        intent_group = intent_result.get("intent_group", "out_of_scope")
        # Use accurate LLM emotional classification; fallback to RoBERTa if missing
        sentiment = intent_result.get("sentiment")
        if not sentiment:
            sentiment_result = self.sentiment_analyzer.predict(message, model_type="roberta")
            sentiment = sentiment_result.get("sentiment", "neutral")

        # ── Stage 4: Routing & Response Generation ─────────────────────────
        escalate = False
        response = ""

        if intent_group == "greeting":
            # Greetings / goodbye / gratitude — detect conversational sub-type
            msg_lower = message.lower()
            if any(w in msg_lower for w in ["thank", "thx", "appreciate", "gracias", "danke"]):
                sub_type = "gratitude"
            elif any(w in msg_lower for w in ["bye", "see you", "goodbye", "adios", "tschüss"]):
                sub_type = "goodbye"
            else:
                sub_type = "greeting"
            response = get_greeting_response(sub_type)

        elif intent_group == "complaint":
            # Complaint — empathy + RAG + escalation flag
            escalate = True
            retrieved = self.retriever.retrieve(message, top_k=RAG_TOP_K)
            rag_response = self.generator.generate(
                user_message=message,
                detected_sentiment=sentiment,
                retrieved_chunks=retrieved,
                detected_language=language,
            )
            response = (
                f"{EMPATHY_PREFIX}\n\n{rag_response}\n\n"
                "I've also flagged this for our support team to review personally. "
                "A human agent will follow up with you shortly."
            )

        else:
            # All other intents — standard RAG flow
            retrieved = self.retriever.retrieve(message, top_k=RAG_TOP_K)

            # If sentiment is negative, add empathy prefix
            if sentiment == "negative":
                rag_response = self.generator.generate(
                    user_message=message,
                    detected_sentiment=sentiment,
                    retrieved_chunks=retrieved,
                    detected_language=language,
                )
                response = f"{EMPATHY_PREFIX}\n\n{rag_response}"
            else:
                response = self.generator.generate(
                    user_message=message,
                    detected_sentiment=sentiment,
                    retrieved_chunks=retrieved,
                    detected_language=language,
                )

        return {
            "response": response,
            "language": language,
            "sentiment": sentiment,
            "intent": intent_group,
            "escalate": escalate,
        }
