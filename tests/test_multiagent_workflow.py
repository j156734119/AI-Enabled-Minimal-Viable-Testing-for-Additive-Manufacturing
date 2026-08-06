from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import am_mvt.agent.workflow as workflow
from am_mvt.agent.workflow import (
    ManagerDecision,
    WorkflowOrchestrator,
    WorkflowStage,
    inspect_existing_artifacts,
    validate_model_run,
    validate_processed_views,
)


def fake_project_path(root: Path):
    return lambda *parts: root.joinpath(*parts)


def write_complete_run(run_dir: Path) -> None:
    for relative in (
        workflow.MODEL_RUN_REQUIRED
        + workflow.EXPLANATION_REQUIRED
        + workflow.OOF_REQUIRED
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")


def test_manager_response_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ManagerDecision.model_validate(
            {
                "next_agent": "none",
                "action": "complete",
                "reason_code": "done",
                "required_artifacts": [],
                "blocking_review": False,
                "command": "arbitrary command",
            }
        )


def test_preflight_prefers_explicit_complete_run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "get_path", fake_project_path(tmp_path))
    explicit = tmp_path / "outputs" / "experiments" / "explicit"
    write_complete_run(explicit)

    result = inspect_existing_artifacts(explicit)

    assert result.status == "complete_evidence_run"
    assert Path(result.run_dir) == explicit.resolve()


def test_preflight_can_retrain_only_when_all_processed_views_exist(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(workflow, "get_path", fake_project_path(tmp_path))
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for name in workflow.PROCESSED_VIEWS:
        (processed / name).write_text("source_id,target\ns1,1\n", encoding="utf-8")

    result = inspect_existing_artifacts()

    assert result.status == "processed_views_require_training"
    assert len(result.processed_views) == 4


def test_existing_only_missing_artifacts_stops_without_evidence_agent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(workflow, "get_path", fake_project_path(tmp_path))
    manager_called = False

    def unexpected_manager(*args, **kwargs):
        nonlocal manager_called
        manager_called = True
        raise AssertionError("OpenAI manager must be skipped in offline mode")

    monkeypatch.setattr(workflow.OpenAIWorkflowManager, "decide", unexpected_manager)
    orchestrator = WorkflowOrchestrator(
        run_id="missing_test",
        existing_artifacts_only=True,
        offline=True,
    )
    summary = orchestrator.execute(
        through=WorkflowStage.ARTIFACT_PREFLIGHT,
        use_openai_manager=False,
    )

    assert summary["stage"] == "blocked_missing_existing_artifacts"
    assert summary["evidence_status"] == "skipped_by_user_scope"
    assert not manager_called
    assert (orchestrator.run_dir / "missing_artifacts_report.csv").exists()


def test_illegal_workflow_transition_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "get_path", fake_project_path(tmp_path))
    orchestrator = WorkflowOrchestrator(run_id="transition_test", offline=True)

    with pytest.raises(ValueError, match="Illegal workflow transition"):
        orchestrator.transition(WorkflowStage.COMPLETE)


def test_data_steward_rejects_nonapproved_processed_rows(tmp_path):
    view = tmp_path / "view_model1_uts.csv"
    pd.DataFrame(
        [
            {
                "record_id": "r1",
                "source_id": "s1",
                "audit_status": "human_review_required",
                "uts_MPa": 900,
            }
        ]
    ).to_csv(view, index=False)

    report, valid = validate_processed_views([view])

    assert not valid
    assert report.iloc[0]["nonapproved_rows"] == 1
    assert "contains_nonapproved_records" in report.iloc[0]["issues"]


def test_model_artifact_validation_requires_oof_schema(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(
        [
            {
                "target": "uts_MPa",
                "mode": "process_only",
                "route": "ordinary_regression",
                "candidate": "rf",
            }
        ]
    ).to_csv(tables / "experiment_summary.csv", index=False)
    pd.DataFrame([{"target": "uts_MPa", "y_pred": 900.0}]).to_csv(
        tables / "oof_predictions.csv",
        index=False,
    )

    report, valid = validate_model_run(tmp_path)

    assert not valid
    oof_row = report.loc[report["artifact_type"].eq("oof_predictions")].iloc[0]
    assert "missing_columns" in oof_row["issues"]
