from __future__ import annotations

import pandas as pd

from am_mvt.retrieval import evidence_index


def test_evidence_index_returns_manifest_chunks_only(tmp_path, monkeypatch):
    active_chunk = tmp_path / "active_chunk_0000.txt"
    orphan_chunk = tmp_path / "orphan_chunk_0000.txt"
    active_chunk.write_text(
        "Ti-6Al-4V tensile strength and yield strength evidence.",
        encoding="utf-8",
    )
    orphan_chunk.write_text(
        "fatigue stress amplitude data that should not be indexed",
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            {
                "chunk_id": active_chunk.stem,
                "chunk_path": str(active_chunk),
                "chunk_sha256": "a" * 64,
                "source_id": "source_active",
                "source_file": "active.pdf",
            }
        ]
    )
    monkeypatch.setattr(
        evidence_index,
        "load_active_chunk_manifest",
        lambda path=None: manifest,
    )

    index_path = tmp_path / "evidence_index.joblib"
    written = evidence_index.build_evidence_index(index_path=index_path)
    results = evidence_index.query_evidence_index(
        "fatigue stress amplitude",
        index_path=written,
        top_k=5,
    )

    assert written == index_path
    assert [result.chunk_id for result in results] == [active_chunk.stem]
    assert "orphan" not in results[0].evidence_snippet


def test_rank_chunk_paths_uses_rag_scores_and_per_source_limit(tmp_path):
    low = tmp_path / "paper_a_chunk_0000.txt"
    high = tmp_path / "paper_a_chunk_0001.txt"
    other = tmp_path / "paper_b_chunk_0000.txt"
    low.write_text("general additive manufacturing introduction", encoding="utf-8")
    high.write_text("fatigue stress amplitude R ratio life data", encoding="utf-8")
    other.write_text("fatigue stress amplitude for another paper", encoding="utf-8")

    ranked = evidence_index.rank_chunk_paths(
        [low, high, other],
        query="fatigue stress amplitude",
        top_k_per_source=1,
    )

    assert high in ranked
    assert other in ranked
    assert low not in ranked
    assert len(ranked) == 2

