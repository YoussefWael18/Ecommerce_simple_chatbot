"""Load policy files and split them into traceable passages."""
import re
from pathlib import Path
from pypdf import PdfReader

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def load_documents(directory: Path) -> tuple[list[dict], list[dict]]:
    pages, issues = [], []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                content = [(n, page.extract_text() or "") for n, page in enumerate(PdfReader(str(path)).pages, 1)]
            else:
                content = [(None, path.read_text(encoding="utf-8"))]
            for page, text in content:
                if text.strip():
                    pages.append({"source": path.name, "page": page, "text": text})
                else:
                    issues.append({"file": path.name, "page": page, "reason": "No extracted text; may require OCR"})
        except Exception as exc:
            issues.append({"file": path.name, "page": None, "reason": f"Parsing failed: {exc}"})
    return pages, issues

def chunk_documents(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    chunks = []
    for page in pages:
        text = clean_text(page["text"])
        start, index = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            passage = text[start:end].strip()
            if passage:
                chunk_id = f"{page['source']}:p{page['page'] or 0}:c{index}"
                metadata = {"source": page["source"], "chunk_id": chunk_id}
                if page["page"] is not None:
                    metadata["page"] = page["page"]
                chunks.append({"id": chunk_id, "text": passage, "metadata": metadata})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
            index += 1
    return chunks

def source_labels(chunks: list[dict]) -> list[str]:
    labels = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if meta.get("source"):
            label = f"{meta['source']} - page {meta['page']}" if meta.get("page") else meta["source"]
            if label not in labels:
                labels.append(label)
    return labels
