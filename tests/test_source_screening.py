import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from am_mvt.ingestion import llm_source_screening
from am_mvt.ingestion.llm_source_screening import (
    MEETING_ONE_JOURNAL_SCOPE,
    build_search_prompt,
    normalise_score,
    select_balanced_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_step01_module():
    script_path = PROJECT_ROOT / "scripts" / "01_search_sources.py"
    spec = importlib.util.spec_from_file_location("step01_search_sources", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalise_score_uses_consistent_zero_to_ten_scale():
    assert normalise_score(0.9) == 9.0
    assert normalise_score(9.0) == 9.0
    assert normalise_score(15.0) == 10.0


def test_balanced_selection_reserves_each_journal():
    rows = []

    for scope in MEETING_ONE_JOURNAL_SCOPE:
        for index in range(3):
            rows.append(
                {
                    "journal": scope.journal,
                    "selection_score": 10 - index,
                }
            )

    selected = select_balanced_candidates(
        pd.DataFrame(rows),
        target_count=16,
        min_per_journal=2,
    )

    assert len(selected) == 16
    assert selected.groupby("journal").size().min() == 2


def test_step01_defaults_to_openai_agent_search(monkeypatch):
    module = load_step01_module()
    calls = []
    output_paths = {
        "interim": "candidate_sources_llm.csv",
        "table": "source_screening_candidates_top50.csv",
        "journal_scope": "source_screening_journal_scope.csv",
    }
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            journals=None,
            merge_existing=False,
            target_count=50,
            per_journal_limit=8,
            min_per_journal=1,
            search_rounds=3,
            year_from=2015,
            year_to=2026,
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        module,
        "run_openai_agent_source_screening",
        lambda **kwargs: (calls.append(kwargs) or pd.DataFrame([{"title": "paper"}]), output_paths),
    )

    module.main()

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["min_per_journal"] == 1
    assert calls[0]["journals"] is None
    assert calls[0]["merge_existing"] is False


def test_step01_cli_defaults_to_one_candidate_per_journal(monkeypatch):
    module = load_step01_module()
    monkeypatch.setattr(sys, "argv", ["01_search_sources.py"])
    assert module.parse_args().min_per_journal == 1


def test_step01_accepts_explicit_journal_subset():
    module = load_step01_module()
    args = module.parse_args(
        ["--journals", "Metals", "Additive Manufacturing"]
    )
    assert args.journals == ["Metals", "Additive Manufacturing"]


def test_step01_accepts_merge_existing():
    module = load_step01_module()
    args = module.parse_args(["--journals", "Metals", "--merge-existing"])
    assert args.journals == ["Metals"]
    assert args.merge_existing is True


def test_explicit_journal_subset_only_calls_requested_journal(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_source_screening, "get_client", lambda: object())
    monkeypatch.setattr(
        llm_source_screening,
        "screen_one_journal",
        lambda **kwargs: (
            calls.append(kwargs["journal_scope"].journal)
            or [
                {
                    "title": "Metal AM tensile data",
                    "journal": kwargs["journal_scope"].journal,
                    "year": 2025,
                    "selection_score": 9,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        llm_source_screening,
        "write_source_screening_outputs",
        lambda df, **kwargs: {
            "interim": "a.csv",
            "table": "b.csv",
            "journal_scope": "c.csv",
        },
    )

    result, _ = llm_source_screening.run_openai_agent_source_screening(
        journals=["Metals"],
        target_count=2,
        search_rounds=1,
    )

    assert calls == ["Metals"]
    assert set(result["journal"]) == {"Metals"}


def test_merge_existing_retains_old_candidates(tmp_path, monkeypatch):
    existing_path = tmp_path / "data" / "interim" / "candidate_sources_llm.csv"
    existing_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "source_id": "old",
                "title": "Existing fatigue paper",
                "journal": "Additive Manufacturing",
                "doi": "10.1000/old",
                "download_status": "downloaded",
            }
        ]
    ).to_csv(existing_path, index=False)
    monkeypatch.setattr(
        llm_source_screening,
        "get_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    new = pd.DataFrame(
        [
            {
                "source_id": "new",
                "title": "New Metals paper",
                "journal": "Metals",
                "doi": "10.1000/new",
                "download_status": "not_downloaded",
            }
        ]
    )

    merged = llm_source_screening.merge_existing_candidates(new)

    assert set(merged["source_id"]) == {"old", "new"}
    assert (
        merged.set_index("source_id").loc["old", "download_status"]
        == "downloaded"
    )


def test_metals_prompt_uses_journal_specific_mdpi_path():
    metals = next(
        scope for scope in MEETING_ONE_JOURNAL_SCOPE
        if scope.journal == "Metals"
    )
    prompt = build_search_prompt(
        metals,
        per_journal_limit=4,
        year_from=2015,
        year_to=2026,
        focus_area="tensile",
    )
    assert "site:mdpi.com/2075-4701" in prompt
    assert "other MDPI journals" in prompt


def test_step01_rejects_removed_legacy_switches():
    module = load_step01_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--llm-web-search"])
    with pytest.raises(SystemExit):
        module.parse_args(["--no-crossref"])


def test_no_valid_api_candidates_preserves_existing_outputs(monkeypatch):
    monkeypatch.setattr(llm_source_screening, "get_client", lambda: object())
    monkeypatch.setattr(
        llm_source_screening,
        "screen_one_journal",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        llm_source_screening,
        "write_source_screening_outputs",
        lambda *args, **kwargs: pytest.fail("empty results must not be written"),
    )

    with pytest.raises(RuntimeError, match="no valid candidates"):
        llm_source_screening.run_openai_agent_source_screening(search_rounds=1)


def test_successful_output_archives_previous_canonical_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        llm_source_screening,
        "get_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    canonical_paths = [
        tmp_path / "data" / "interim" / "candidate_sources_llm.csv",
        tmp_path / "outputs" / "tables" / "source_screening_candidates_top50.csv",
        tmp_path / "outputs" / "tables" / "source_screening_journal_scope.csv",
    ]
    for path in canonical_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old-content", encoding="utf-8")

    new_df = pd.DataFrame(
        [
            {
                "source_id": "new-paper",
                "title": "New paper",
                "journal": "Additive Manufacturing",
            }
        ]
    )
    llm_source_screening.write_source_screening_outputs(
        new_df,
        timestamp="20260608T120000000000Z",
    )

    archive_root = (
        tmp_path / "archive" / "source_search_runs" / "20260608T120000000000Z"
    )
    for canonical_path in canonical_paths:
        archived_path = archive_root / canonical_path.relative_to(tmp_path)
        assert archived_path.read_text(encoding="utf-8") == "old-content"
        assert canonical_path.read_text(encoding="utf-8-sig") != "old-content"


def test_crossref_http_helpers_are_not_present():
    assert not hasattr(llm_source_screening, "search_crossref_journal")
    assert not hasattr(llm_source_screening, "make_crossref_session")
