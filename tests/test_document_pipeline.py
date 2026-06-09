from __future__ import annotations

from pathlib import Path

import pandas as pd

from am_mvt.parsing import document_pipeline


def test_active_pipeline_skips_duplicate_pdf_and_archives_orphans(
    tmp_path,
    monkeypatch,
):
    pdf_dir = tmp_path / "data" / "raw" / "pdfs"
    parsed_dir = tmp_path / "data" / "interim" / "parsed_text"
    chunk_dir = tmp_path / "data" / "interim" / "text_chunks"
    output_dir = tmp_path / "data" / "interim" / "llm_outputs"
    for directory in [pdf_dir, parsed_dir, chunk_dir, output_dir]:
        directory.mkdir(parents=True)

    canonical_pdf = pdf_dir / "004_paper.pdf"
    duplicate_pdf = pdf_dir / "039_paper.pdf"
    canonical_pdf.write_bytes(b"canonical")
    duplicate_pdf.write_bytes(b"canonical")
    orphan_chunk = chunk_dir / "039_paper_chunk_0000.txt"
    orphan_output = output_dir / "039_paper_chunk_0000.json"
    orphan_chunk.write_text("old duplicate", encoding="utf-8")
    orphan_output.write_text("{}", encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {
                "local_pdf_filename": canonical_pdf.name,
                "canonical_source_id": "004_paper",
                "content_sha256": "a" * 64,
                "processing_status": "canonical",
                "ready_for_parsing": True,
            },
            {
                "local_pdf_filename": duplicate_pdf.name,
                "canonical_source_id": "004_paper",
                "content_sha256": "a" * 64,
                "processing_status": "duplicate",
                "ready_for_parsing": False,
            },
        ]
    )
    monkeypatch.setattr(
        document_pipeline,
        "get_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    monkeypatch.setattr(
        document_pipeline,
        "save_literature_manifest",
        lambda: (tmp_path / "docs" / "literature_manifest.csv", manifest),
    )
    monkeypatch.setattr(
        document_pipeline,
        "load_skipped_pdf_stems",
        lambda: set(),
    )

    def fake_parse(pdf_path: Path, output_path: Path) -> Path:
        output_path.write_text("A" * 4000, encoding="utf-8")
        return output_path

    monkeypatch.setattr(document_pipeline, "parse_pdf_to_text", fake_parse)
    parsed, chunks, active_manifest, archive_manifest = (
        document_pipeline.parse_active_pdf_documents(
            pdf_dir=pdf_dir,
            parsed_text_dir=parsed_dir,
            chunk_dir=chunk_dir,
            llm_output_dir=output_dir,
        )
    )

    active = pd.read_csv(active_manifest)
    assert parsed == 1
    assert chunks == 2
    assert set(active["source_id"]) == {"004_paper"}
    assert archive_manifest is not None
    assert not orphan_chunk.exists()
    assert not orphan_output.exists()
