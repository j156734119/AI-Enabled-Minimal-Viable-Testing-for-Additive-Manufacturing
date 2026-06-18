from __future__ import annotations

import json

import pandas as pd

from am_mvt.agent.react_ledger import ReactLedger, record_human_download_boundary


def test_react_ledger_records_auditable_summaries_without_cot(tmp_path):
    ledger = ReactLedger(run_id="test_run", run_dir=tmp_path / "test_run")
    record_human_download_boundary(
        ledger,
        candidate_refs=["doi:10.example/test"],
        observed_pdf_refs=["paper_chunk_0000"],
    )
    ledger.record(
        plan_summary="Retrieve local evidence chunks.",
        action_type="retrieve_local_chunks",
        input_refs=["paper_chunk_0000"],
        observation_summary="One local chunk matched.",
        decision="continue_to_extraction",
        evidence_refs=["paper_chunk_0000"],
    )

    csv_rows = pd.read_csv(ledger.csv_path)
    jsonl_text = ledger.jsonl_path.read_text(encoding="utf-8")
    json_rows = [
        json.loads(line)
        for line in jsonl_text.splitlines()
        if line.strip()
    ]

    assert list(csv_rows["action_type"]) == [
        "human_download_required",
        "local_pdf_observed",
        "retrieve_local_chunks",
    ]
    assert "chain_of_thought" not in csv_rows.columns
    assert "Thought:" not in jsonl_text
    assert json_rows[-1]["decision"] == "continue_to_extraction"

