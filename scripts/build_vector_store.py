#!/usr/bin/env python3
"""
Build the ChromaDB vector store from the Bitext customer-support dataset.

This script:
  1. Downloads the Bitext dataset from HuggingFace
  2. Embeds the 'instruction' column using all-MiniLM-L6-v2
  3. Stores embeddings + metadata (response, intent, category) in ChromaDB

Run once before starting the chatbot:
    python scripts/build_vector_store.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datasets import load_dataset
from app.config import (
    BITEXT_DATASET_NAME,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore


def main():
    print(f"Loading dataset: {BITEXT_DATASET_NAME}")
    ds = load_dataset(BITEXT_DATASET_NAME)

    # The dataset has a single 'train' split
    df = ds["train"].to_pandas()
    print(f"  → {len(df)} rows loaded")
    print(f"  → Columns: {list(df.columns)}")
    print(f"  → Intent distribution:\n{df['intent'].value_counts().to_string()}\n")

    # Prepare documents and metadata
    instructions = df["instruction"].tolist()
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "response": str(row.get("response", "")),
            "intent": str(row.get("intent", "")),
            "category": str(row.get("category", "")),
        })

    # Generate unique IDs
    ids = [f"doc_{i}" for i in range(len(instructions))]

    # Embed all instructions
    print(f"Embedding {len(instructions)} instructions with {EMBEDDING_MODEL_NAME}...")
    embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)
    embeddings = embedder.embed(instructions, batch_size=128, show_progress_bar=True)
    print(f"  → Embedding shape: {len(embeddings)} x {len(embeddings[0])}")

    # Create ChromaDB persistent directory
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Store in ChromaDB
    print(f"Storing in ChromaDB at {CHROMA_DIR}...")
    store = VectorStore(
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    store.add_documents(
        ids=ids,
        documents=instructions,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  → {store.count()} documents stored in collection '{CHROMA_COLLECTION_NAME}'")
    print("Done! Vector store is ready.")

    # Quick sanity check
    print("\n--- Sanity Check ---")
    test_query = "Where is my order?"
    query_emb = embedder.embed_single(test_query)
    results = store.search(query_emb, top_k=3)
    print(f"Query: '{test_query}'")
    for i, r in enumerate(results):
        print(f"  Result {i+1}: [intent={r['metadata'].get('intent', 'N/A')}] {r['document'][:100]}...")


if __name__ == "__main__":
    main()
