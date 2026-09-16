#!/usr/bin/env python3
"""Build the persistent policy-document index from data/raw."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL_NAME
from app.rag.documents import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents, load_documents
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore

def build():
    pages, issues = load_documents(ROOT / "data" / "raw")
    chunks = chunk_documents(pages)
    if not chunks:
        raise ValueError("No extractable .txt/.pdf documents in data/raw")
    embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)
    store = VectorStore(str(CHROMA_DIR), CHROMA_COLLECTION_NAME)
    existing = set(store.collection.get(include=[])["ids"])
    current = {c["id"] for c in chunks}
    if existing - current:
        store.collection.delete(ids=list(existing - current))
    store.add_documents(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embedder.embed([c["text"] for c in chunks]),
        metadatas=[c["metadata"] for c in chunks],
    )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (CHROMA_DIR / "config.json").write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL_NAME, "collection": CHROMA_COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "pages": len(pages), "chunks": len(chunks), "issues": issues,
    }, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages; parsing issues: {issues}")
    return store

if __name__ == "__main__":
    build()
