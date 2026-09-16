from app.rag.documents import load_documents, chunk_documents, source_labels


def test_load_txt_and_chunk_metadata(tmp_path):
    (tmp_path / "policy.txt").write_text("Returns are possible. " * 70, encoding="utf-8")
    pages, issues = load_documents(tmp_path)
    assert len(pages) == 1 and not issues
    chunks = chunk_documents(pages)
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["source"] == "policy.txt"
    assert chunks[0]["metadata"]["chunk_id"] == "policy.txt:p0:c0"
    assert source_labels(chunks) == ["policy.txt"]


def test_pdf_without_text_flagged(tmp_path):
    from pypdf import PdfWriter
    with (tmp_path / "scan.pdf").open("wb") as output:
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(output)
    pages, issues = load_documents(tmp_path)
    assert not pages
    assert "OCR" in issues[0]["reason"]


def test_persistent_retrieval_keeps_source(tmp_path):
    from app.rag.vector_store import VectorStore
    from app.rag.retriever import Retriever

    class FakeEmbedder:
        def embed_single(self, query):
            return [1.0, 0.0]

    path = str(tmp_path / "chroma")
    store = VectorStore(path, "policy_test")
    store.add_documents(["shipping:p1:c0"], ["Standard shipping takes three days"],
                        [[1.0, 0.0]], [{"source": "shipping.pdf", "page": 1,
                                         "chunk_id": "shipping:p1:c0"}])
    reopened = VectorStore(path, "policy_test")
    result = Retriever(FakeEmbedder(), reopened).retrieve("shipping", top_k=1)
    assert result[0]["metadata"]["page"] == 1
    assert source_labels(result) == ["shipping.pdf - page 1"]
