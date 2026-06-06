from pathlib import Path

import pandas as pd
import pytest

from am_mvt.extraction.audit_records import (
    audit_extracted_records,
    load_approved_record_keys,
)
from am_mvt.extraction.merge_llm_records import append_llm_records_to_master


def make_record(**overrides):
    record = {
        "source_id": "paper_1",
        "source_file": "paper_1.pdf",
        "record_id": "chunk_1_record_0001",
        "evidence_text": "UTS was measured as 950 MPa.",
        "confidence": 0.95,
        "needs_human_check": False,
        "alloy": "Ti-6Al-4V",
        "am_process": "LPBF",
        "uts_MPa": 950,
        "extraction_method": "llm_extraction",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    [
        ({}, "approved"),
        ({"evidence_text": ""}, "human_review_required"),
        ({"confidence": 0.60}, "human_review_required"),
        ({"needs_human_check": True}, "human_review_required"),
        ({"uts_MPa": 5000}, "rejected"),
        (
            {
                "alloy": None,
                "am_process": None,
                "uts_MPa": None,
            },
            "rejected",
        ),
    ],
)
def test_deterministic_audit_status(overrides, expected_status):
    audited = audit_extracted_records(pd.DataFrame([make_record(**overrides)]))
    assert audited.loc[0, "audit_status"] == expected_status


def test_load_approved_record_keys_requires_audit_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="04b_audit_extractions"):
        load_approved_record_keys(tmp_path / "missing.csv")


def test_human_approval_requires_reviewer_metadata(tmp_path):
    audit_path = tmp_path / "audit.csv"
    audited = audit_extracted_records(
        pd.DataFrame([make_record(needs_human_check=True)])
    )
    audited.loc[0, "audit_status"] = "approved"
    audited.loc[0, "audit_method"] = "human_review"
    audited.to_csv(audit_path, index=False)

    with pytest.raises(ValueError, match="reviewed_by"):
        load_approved_record_keys(audit_path)


def test_merge_only_uses_approved_candidate_records(tmp_path):
    master_path = tmp_path / "master.csv"
    candidate_path = tmp_path / "candidates.csv"
    audit_path = tmp_path / "audit.csv"
    output_path = tmp_path / "output.csv"

    pd.DataFrame(
        [
            {
                "source_id": "baseline",
                "record_id": "base_1",
                "alloy": "316L",
                "uts_MPa": 600,
                "extraction_method": "public_dataset",
            }
        ]
    ).to_csv(master_path, index=False)

    candidates = pd.DataFrame(
        [
            make_record(),
            make_record(
                source_id="paper_2",
                record_id="chunk_2_record_0001",
                confidence=0.60,
            ),
        ]
    )
    candidates.to_csv(candidate_path, index=False)

    audited = audit_extracted_records(candidates)
    audited.to_csv(audit_path, index=False)

    append_llm_records_to_master(
        master_path=master_path,
        llm_csv_path=candidate_path,
        audit_path=audit_path,
        output_path=output_path,
        make_backup=False,
    )

    merged = pd.read_csv(output_path, low_memory=False)
    assert set(merged["source_id"]) == {"baseline", "paper_1"}
    approved_row = merged.loc[merged["source_id"].eq("paper_1")].iloc[0]
    assert approved_row["audit_status"] == "approved"
    assert approved_row["audit_method"] == "deterministic"


def test_merge_stops_when_candidate_changed_after_audit(tmp_path):
    master_path = tmp_path / "master.csv"
    candidate_path = tmp_path / "candidates.csv"
    audit_path = tmp_path / "audit.csv"
    output_path = tmp_path / "output.csv"

    pd.DataFrame(
        [{"source_id": "baseline", "record_id": "base_1"}]
    ).to_csv(master_path, index=False)
    candidates = pd.DataFrame([make_record()])
    audit_extracted_records(candidates).to_csv(audit_path, index=False)
    candidates.loc[0, "uts_MPa"] = 1000
    candidates.to_csv(candidate_path, index=False)

    with pytest.raises(ValueError, match="do not match"):
        append_llm_records_to_master(
            master_path=master_path,
            llm_csv_path=candidate_path,
            audit_path=audit_path,
            output_path=output_path,
            make_backup=False,
        )
