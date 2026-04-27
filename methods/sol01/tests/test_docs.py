from sol01.docs import DOCUMENTS_ROOT, get_metric_definition, load_document_chunks


def test_documents_root_exists():
    assert DOCUMENTS_ROOT.exists()


def test_rfm_document_is_chunked_by_heading():
    chunks = load_document_chunks("RFM.md")

    headings = [chunk.heading for chunk in chunks if chunk.heading]

    assert any(heading == "Introduction to the RFM Model" for heading in headings)
    assert any(heading == "RFM Segmentation Logic" for heading in headings)
    assert any(chunk.kind == "paragraph" for chunk in chunks)


def test_get_metric_definition_prefers_task_external_knowledge():
    metric = get_metric_definition("RFM", instance_id="local003")

    assert metric.source_file == "RFM.md"
    assert "Recency" in metric.definition
    assert metric.confidence > 0.8


def test_get_metric_definition_finds_retention_rate_in_corpus():
    metric = get_metric_definition("retention rate")

    assert metric.source_file == "retention_rate.md"
    assert "N-Day retention" in metric.definition
    assert metric.metric_name == "retention rate"


def test_get_metric_definition_finds_tip_rate_in_corpus():
    metric = get_metric_definition("tip_rate")

    assert metric.source_file == "taxi_tip_rate.md"
    assert "no tip" in metric.definition
    assert metric.confidence > 0.5
