from __future__ import annotations

import json

from am_mvt.extraction.batch_extract import (
    chunk_is_skipped,
    chunk_has_usable_text,
    load_extraction_skip_stems,
    output_has_error,
    output_needs_pdf_vision,
    select_chunks_for_extraction,
    write_extraction_result,
)
from am_mvt.extraction.postprocess import load_llm_json_outputs


def write_output(path, *, error=None):
    payload = {
        "records": [],
        "_metadata": {"error": error} if error else {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pending_limit_is_applied_after_successful_outputs_are_skipped(tmp_path):
    chunk_files = []

    for index in range(5):
        chunk = tmp_path / f"paper_chunk_{index:04d}.txt"
        chunk.write_text("text", encoding="utf-8")
        chunk_files.append(chunk)

    write_output(tmp_path / "paper_chunk_0000.json")
    write_output(tmp_path / "paper_chunk_0001.json")

    selected, skipped = select_chunks_for_extraction(
        chunk_files,
        tmp_path,
        limit=2,
        overwrite=False,
    )

    assert skipped == 2
    assert [path.stem for path in selected] == [
        "paper_chunk_0002",
        "paper_chunk_0003",
    ]


def test_failed_or_malformed_outputs_are_retried(tmp_path):
    chunks = []

    for name in ["failed", "malformed", "successful"]:
        chunk = tmp_path / f"{name}.txt"
        chunk.write_text("text", encoding="utf-8")
        chunks.append(chunk)

    write_output(tmp_path / "failed.json", error="temporary API error")
    (tmp_path / "malformed.json").write_text("{", encoding="utf-8")
    write_output(tmp_path / "successful.json")

    selected, skipped = select_chunks_for_extraction(
        chunks,
        tmp_path,
        limit=None,
        overwrite=False,
    )

    assert output_has_error(tmp_path / "failed.json")
    assert output_has_error(tmp_path / "malformed.json")
    assert not output_has_error(tmp_path / "successful.json")
    assert skipped == 1
    assert {path.stem for path in selected} == {"failed", "malformed"}


def test_rag_priority_only_reorders_pending_outputs(tmp_path):
    chunks = []
    texts = {
        "successful": "fatigue stress amplitude high relevance",
        "low": "general additive manufacturing text",
        "high": "fatigue stress amplitude R ratio life data",
    }
    for name, text in texts.items():
        chunk = tmp_path / f"{name}.txt"
        chunk.write_text(text, encoding="utf-8")
        chunks.append(chunk)

    write_output(tmp_path / "successful.json")

    selected, skipped = select_chunks_for_extraction(
        chunks,
        tmp_path,
        limit=1,
        overwrite=False,
        use_rag_priority=True,
        rag_query="fatigue stress amplitude",
    )

    assert skipped == 1
    assert [path.stem for path in selected] == ["high"]


def test_empty_scan_output_is_retried_for_pdf_vision(tmp_path):
    chunk = tmp_path / "scan_chunk_0000.txt"
    chunk.write_text("--- Page 1 ---\n--- Page 2 ---", encoding="utf-8")
    output = tmp_path / "scan_chunk_0000.json"
    write_output(output)

    selected, skipped = select_chunks_for_extraction(
        [chunk],
        tmp_path,
        limit=None,
        overwrite=False,
    )

    assert not chunk_has_usable_text(chunk)
    assert output_needs_pdf_vision(output, chunk)
    assert selected == [chunk]
    assert skipped == 0


def test_api_failure_does_not_overwrite_existing_output(tmp_path):
    output = tmp_path / "existing.json"
    write_output(output)
    original = output.read_text(encoding="utf-8")
    written = write_extraction_result(
        output,
        {"records": [], "_metadata": {"error": "temporary failure"}},
    )
    assert not written
    assert output.read_text(encoding="utf-8") == original


def test_configured_pdf_stem_is_skipped(tmp_path):
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("paper_one\n", encoding="utf-8")
    stems = load_extraction_skip_stems(skip_file)
    assert chunk_is_skipped(
        tmp_path / "paper_one_chunk_0000.txt",
        stems,
    )


def test_postprocess_ignores_orphan_json(tmp_path, monkeypatch):
    active = tmp_path / "active_chunk_0000.json"
    orphan = tmp_path / "orphan_chunk_0000.json"
    payload = {
        "records": [],
        "_metadata": {"source_file": "paper.pdf", "chunk_id": active.stem},
    }
    active.write_text(json.dumps(payload), encoding="utf-8")
    orphan.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "am_mvt.extraction.postprocess.load_active_chunk_manifest",
        lambda: __import__("pandas").DataFrame({"chunk_id": [active.stem]}),
    )

    result = load_llm_json_outputs(tmp_path)

    assert result.empty
