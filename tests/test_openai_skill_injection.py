from types import SimpleNamespace

from am_mvt.extraction import openai_extractor
from am_mvt.ingestion import llm_source_screening


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def test_extraction_request_injects_skill_and_json_schema(monkeypatch):
    client = FakeClient('{"records": []}')
    monkeypatch.setattr(openai_extractor, "get_openai_client", lambda: client)

    openai_extractor.extract_records_from_chunk(
        chunk_text="No extractable values.",
        source_file="paper.pdf",
        chunk_id="chunk_1",
    )

    request = client.responses.kwargs
    system_prompt = request["input"][0]["content"]
    assert "# Goal" in system_prompt
    assert "one experimental condition" in system_prompt.lower()
    assert request["text"]["format"]["strict"] is True


def test_pdf_vision_request_uses_responses_input_file(tmp_path, monkeypatch):
    client = FakeClient('{"records": []}')
    monkeypatch.setattr(openai_extractor, "get_openai_client", lambda: client)
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    result = openai_extractor.extract_records_from_pdf(
        pdf_path,
        source_file="scan.pdf",
        chunk_id="scan_chunk_0000",
        max_retries=1,
    )

    content = client.responses.kwargs["input"][1]["content"]
    file_item = next(item for item in content if item["type"] == "input_file")
    assert file_item["filename"] == "scan.pdf"
    assert file_item["file_data"].startswith("data:application/pdf;base64,")
    assert result["_metadata"]["input_mode"] == "pdf_vision"


def test_source_screening_request_injects_skill_and_web_search():
    client = FakeClient('{"candidates": []}')
    scope = llm_source_screening.MEETING_ONE_JOURNAL_SCOPE[0]

    llm_source_screening.screen_one_journal(
        client=client,
        journal_scope=scope,
        per_journal_limit=1,
        year_from=2020,
        year_to=2026,
        model="test-model",
        focus_area="fatigue",
        retry_count=1,
    )

    request = client.responses.kwargs
    system_prompt = request["input"][0]["content"]
    assert "# Goal" in system_prompt
    assert "source-screening" in system_prompt.lower()
    assert request["tools"] == [{"type": "web_search"}]
    assert request["text"]["format"]["strict"] is True
