"""
FastAPI application for the RAG-based customer support chatbot.
Endpoints:
  POST /chat   — process a customer message through the full pipeline
  GET  /health — check system status and loaded models
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline import ChatPipeline


# ── Request / Response schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    language: str
    sentiment: str
    intent: str
    escalate: bool


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    vector_store_documents: int


# ── Application lifecycle ──────────────────────────────────────────────────────

pipeline: ChatPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup, clean up on shutdown."""
    global pipeline
    print("Starting up — loading pipeline...")
    pipeline = ChatPipeline()
    yield
    print("Shutting down.")
    pipeline = None


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="E-commerce Customer Support Chatbot",
    description="RAG-based chatbot with language detection, sentiment analysis, and intent classification.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a customer message through the full pipeline.

    Returns the generated response along with detected language,
    sentiment, intent, and whether the case should be escalated.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    result = pipeline.process(request.message.strip())
    return ChatResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check system health and list loaded models."""
    if pipeline is None:
        return HealthResponse(
            status="initializing",
            models_loaded=[],
            vector_store_documents=0,
        )

    models = [
        "language_detector (TF-IDF + MultinomialNB)",
        "sentiment_analyzer (DistilRoBERTa)",
        "intent_classifier (OpenRouter few-shot)",
        "rag_embedder (all-MiniLM-L6-v2)",
        "rag_generator (OpenRouter LLM)",
    ]
    doc_count = pipeline.vector_store.count()

    return HealthResponse(
        status="ok",
        models_loaded=models,
        vector_store_documents=doc_count,
    )
