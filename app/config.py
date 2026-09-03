"""
Configuration module for the RAG-based customer support chatbot.
Loads environment variables and defines paths, model settings, and API config.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # /home/pccv/snap
MODEL_DIR = BASE_DIR / "Model_pickle"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY") or ""

# ── Model Paths ────────────────────────────────────────────────────────────────
LANG_CLASSIFIER_PATH = MODEL_DIR / "lang_classifier .pkl"       # Note: space in filename
ROBERTA_SENTIMENT_PATH = MODEL_DIR / "roberta_sentiment.pkl"
ROBERTA_TOKENIZER_PATH = MODEL_DIR / "roberta_tokenizer"
BILSTM_SENTIMENT_PATH = MODEL_DIR / "bilstm_sentiment.pkl"

# ── OpenRouter Settings ────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = "minimax/minimax-m3:free"

# ── RAG Settings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "customer_support"
RAG_TOP_K = 3

# ── Bitext Dataset ─────────────────────────────────────────────────────────────
BITEXT_DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
