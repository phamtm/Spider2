from sol01.loading.docs import (
    DOCUMENTS_ROOT,
    load_document_chunks,
    load_document_text,
)


def test_documents_root_exists():
    assert DOCUMENTS_ROOT.exists()


def test_rfm_document_is_chunked_by_heading():
    chunks = load_document_chunks("RFM.md")

    headings = [chunk.heading for chunk in chunks if chunk.heading]

    assert any(heading == "Introduction to the RFM Model" for heading in headings)
    assert any(heading == "RFM Segmentation Logic" for heading in headings)
    assert any(chunk.kind == "paragraph" for chunk in chunks)


def test_load_document_text_returns_whole_rfm_document():
    text = load_document_text("RFM.md")

    assert text.startswith("# Introduction to the RFM Model")
    assert "## RFM Segmentation Logic" in text
