from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from am_mvt.agent.react_ledger import ReactLedger, new_run_id
from am_mvt.config import get_path, load_config, load_project_environment
from am_mvt.utils.artifacts import sha256_file


class WorkflowStage(str, Enum):
    ARTIFACT_PREFLIGHT = "artifact_preflight"
    DATA_READY = "data_ready"
    MODEL_READY = "model_ready"
    MATRIX_READY = "matrix_ready"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETE = "complete"
    BLOCKED_MISSING_ARTIFACTS = "blocked_missing_existing_artifacts"
    FAILED = "failed"


class ManagerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_agent: Literal[
        "evidence_agent",
        "data_steward_agent",
        "modelling_agent",
        "testing_decision_agent",
        "human_reviewer",
        "none",
    ]
    action: Literal[
        "skip_evidence",
        "validate_existing_artifacts",
        "run_modelling",
        "run_explanation",
        "generate_matrix",
        "request_human_review",
        "stop_missing_artifacts",
        "complete",
    ]
    reason_code: str
    required_artifacts: list[str] = Field(default_factory=list)
    blocking_review: bool = False


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: WorkflowStage = WorkflowStage.ARTIFACT_PREFLIGHT
    existing_artifacts_only: bool = True
    requested_run_dir: str = ""
    resolved_run_dir: str = ""
    evidence_status: str = "skipped_by_user_scope"
    audit_status: str = "not_checked"
    modelling_status: str = "not_started"
    matrix_status: str = "not_started"
    blocking_human_reviews: int = 0
    last_action: str = ""
    reason_code: str = ""
    artifact_refs: list[str] = Field(default_factory=list)
    updated_at_utc: str = Field(default_factory=lambda: utc_now())


class ArtifactPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "complete_evidence_run",
        "model_run_requires_explanation",
        "processed_views_require_training",
        "missing",
    ]
    run_dir: str = ""
    available_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    processed_views: list[str] = Field(default_factory=list)


MANAGER_SYSTEM_PROMPT = """
You are the bounded workflow manager for an MSc additive-manufacturing
minimal-viable-testing project. Select only one of the explicitly allowed
workflow actions. Never invent files, approve evidence, alter measurements,
download restricted sources, or return shell commands. Use concise reason
codes and artifact references. The local policy engine is authoritative.
""".strip()

ACTION_AGENT = {
    "skip_evidence": "evidence_agent",
    "validate_existing_artifacts": "data_steward_agent",
    "run_modelling": "modelling_agent",
    "run_explanation": "modelling_agent",
    "generate_matrix": "testing_decision_agent",
    "request_human_review": "human_reviewer",
    "stop_missing_artifacts": "none",
    "complete": "none",
}


MODEL_RUN_REQUIRED = [
    "tables/experiment_summary.csv",
    "model_registry.json",
]

EXPLANATION_REQUIRED = [
    "tables/relationship_evidence.csv",
    "tables/combination_coverage.csv",
    "tables/feature_importance_comparison.csv",
    "tables/grouped_error_analysis.csv",
    "tables/b2_combination_diagnostics.csv",
    "tables/variable_coverage.csv",
]

OOF_REQUIRED = ["tables/oof_predictions.csv"]

PROCESSED_VIEWS = [
    "view_model1_uts.csv",
    "view_model2_sn_fatigue.csv",
    "view_model3_elongation_yield.csv",
    "view_model4_elastic_modulus.csv",
]

VIEW_TARGETS = {
    "view_model1_uts.csv": ["uts_MPa"],
    "view_model2_sn_fatigue.csv": ["log10_fatigue_life_cycles"],
    "view_model3_elongation_yield.csv": [
        "elongation_percent",
        "yield_strength_MPa",
    ],
    "view_model4_elastic_modulus.csv": ["youngs_modulus_GPa"],
}

REQUIRED_WORKFLOW_TARGETS = {
    "uts_MPa",
    "yield_strength_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
    "log10_fatigue_life_cycles",
}

ALLOWED_TRANSITIONS = {
    WorkflowStage.ARTIFACT_PREFLIGHT: {
        WorkflowStage.DATA_READY,
        WorkflowStage.MODEL_READY,
        WorkflowStage.BLOCKED_MISSING_ARTIFACTS,
        WorkflowStage.APPROVAL_REQUIRED,
        WorkflowStage.FAILED,
    },
    WorkflowStage.DATA_READY: {
        WorkflowStage.MODEL_READY,
        WorkflowStage.APPROVAL_REQUIRED,
        WorkflowStage.FAILED,
    },
    WorkflowStage.MODEL_READY: {
        WorkflowStage.MATRIX_READY,
        WorkflowStage.APPROVAL_REQUIRED,
        WorkflowStage.FAILED,
    },
    WorkflowStage.MATRIX_READY: {
        WorkflowStage.COMPLETE,
        WorkflowStage.APPROVAL_REQUIRED,
        WorkflowStage.FAILED,
    },
    WorkflowStage.APPROVAL_REQUIRED: {
        WorkflowStage.DATA_READY,
        WorkflowStage.MODEL_READY,
        WorkflowStage.FAILED,
    },
    WorkflowStage.COMPLETE: set(),
    WorkflowStage.BLOCKED_MISSING_ARTIFACTS: set(),
    WorkflowStage.FAILED: set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_existing(base: Path, paths: list[str]) -> list[str]:
    return [path for path in paths if (base / path).exists()]


def latest_complete_run() -> Path | None:
    root = get_path("outputs", "experiments")
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / required).exists() for required in MODEL_RUN_REQUIRED)
    ]
    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def inspect_existing_artifacts(run_dir: str | Path | None = None) -> ArtifactPreflight:
    requested = Path(run_dir).resolve() if run_dir else None
    views_root = get_path("data", "processed")
    existing_views = [
        str(views_root / name)
        for name in PROCESSED_VIEWS
        if (views_root / name).exists()
    ]

    latest = latest_complete_run()
    run_candidates = []
    if requested is not None and requested.exists():
        run_candidates.append(requested)
    if latest is not None and latest not in run_candidates:
        run_candidates.append(latest)

    for resolved in run_candidates:
        model_available = _relative_existing(resolved, MODEL_RUN_REQUIRED)
        explanation_available = _relative_existing(resolved, EXPLANATION_REQUIRED)
        oof_available = _relative_existing(resolved, OOF_REQUIRED)
        available = model_available + explanation_available + oof_available
        if (
            len(model_available) == len(MODEL_RUN_REQUIRED)
            and len(explanation_available) == len(EXPLANATION_REQUIRED)
            and len(oof_available) == len(OOF_REQUIRED)
        ):
            return ArtifactPreflight(
                status="complete_evidence_run",
                run_dir=str(resolved),
                available_artifacts=available,
                processed_views=existing_views,
            )
        if len(model_available) == len(MODEL_RUN_REQUIRED):
            missing = [
                path
                for path in EXPLANATION_REQUIRED + OOF_REQUIRED
                if not (resolved / path).exists()
            ]
            return ArtifactPreflight(
                status="model_run_requires_explanation",
                run_dir=str(resolved),
                available_artifacts=available,
                missing_artifacts=missing,
                processed_views=existing_views,
            )

    if len(existing_views) == len(PROCESSED_VIEWS):
        return ArtifactPreflight(
            status="processed_views_require_training",
            processed_views=existing_views,
            missing_artifacts=MODEL_RUN_REQUIRED + EXPLANATION_REQUIRED + OOF_REQUIRED,
        )

    missing_views = [
        str(views_root / name)
        for name in PROCESSED_VIEWS
        if not (views_root / name).exists()
    ]
    return ArtifactPreflight(
        status="missing",
        run_dir=str(requested) if requested else "",
        processed_views=existing_views,
        missing_artifacts=MODEL_RUN_REQUIRED
        + EXPLANATION_REQUIRED
        + OOF_REQUIRED
        + missing_views,
    )


def deterministic_manager_decision(preflight: ArtifactPreflight) -> ManagerDecision:
    if preflight.status == "complete_evidence_run":
        return ManagerDecision(
            next_agent="testing_decision_agent",
            action="generate_matrix",
            reason_code="existing_model_evidence_complete",
            required_artifacts=preflight.available_artifacts,
        )
    if preflight.status == "model_run_requires_explanation":
        if "tables/oof_predictions.csv" in preflight.missing_artifacts:
            if len(preflight.processed_views) == len(PROCESSED_VIEWS):
                return ManagerDecision(
                    next_agent="modelling_agent",
                    action="run_modelling",
                    reason_code="oof_evidence_requires_retraining_existing_views",
                    required_artifacts=preflight.processed_views,
                )
            return ManagerDecision(
                next_agent="none",
                action="stop_missing_artifacts",
                reason_code="oof_and_processed_views_missing",
                required_artifacts=preflight.missing_artifacts,
            )
        return ManagerDecision(
            next_agent="modelling_agent",
            action="run_explanation",
            reason_code="model_run_requires_step07",
            required_artifacts=preflight.available_artifacts,
        )
    if preflight.status == "processed_views_require_training":
        return ManagerDecision(
            next_agent="modelling_agent",
            action="run_modelling",
            reason_code="existing_processed_views_ready",
            required_artifacts=preflight.processed_views,
        )
    return ManagerDecision(
        next_agent="none",
        action="stop_missing_artifacts",
        reason_code="no_existing_model_or_processed_views",
        required_artifacts=preflight.missing_artifacts,
    )


def validate_processed_views(paths: list[str | Path]) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    valid = True
    for item in paths:
        path = Path(item)
        issues: list[str] = []
        if not path.exists():
            issues.append("missing_file")
            frame = pd.DataFrame()
        else:
            frame = pd.read_csv(path, low_memory=False)
        expected_targets = VIEW_TARGETS.get(path.name, [])
        missing_targets = [target for target in expected_targets if target not in frame]
        if missing_targets:
            issues.append("missing_targets:" + ",".join(missing_targets))
        if not {"source_id", "source_file", "doi"} & set(frame):
            issues.append("missing_source_provenance")
        if not {"evaluation_group_id", "modelling_group_id", "source_id"} & set(frame):
            issues.append("missing_evaluation_group")
        nonapproved = 0
        if "audit_status" in frame:
            approved = frame["audit_status"].astype("string").eq("approved")
            nonapproved = int((~approved).sum())
            if nonapproved:
                issues.append("contains_nonapproved_records")
        duplicate_records = 0
        if "record_id" in frame:
            duplicate_records = int(frame["record_id"].dropna().duplicated().sum())
            if duplicate_records:
                issues.append("duplicate_record_id")
        file_valid = not issues
        valid &= file_valid
        rows.append(
            {
                "artifact": str(path.resolve()),
                "rows": len(frame),
                "expected_targets": ";".join(expected_targets),
                "nonapproved_rows": nonapproved,
                "duplicate_record_ids": duplicate_records,
                "sha256": sha256_file(path) if path.exists() else "",
                "valid": file_valid,
                "issues": ";".join(issues),
            }
        )
    return pd.DataFrame(rows), valid


def validate_model_run(run_dir: str | Path) -> tuple[pd.DataFrame, bool]:
    run_dir = Path(run_dir)
    checks = {
        "experiment_summary": (
            run_dir / "tables" / "experiment_summary.csv",
            {"target", "mode", "route", "candidate"},
        ),
        "oof_predictions": (
            run_dir / "tables" / "oof_predictions.csv",
            {"target", "mode", "route", "fold", "y_true", "y_pred", "abs_error"},
        ),
    }
    rows: list[dict[str, Any]] = []
    valid = True
    for artifact_type, (path, required_columns) in checks.items():
        issues: list[str] = []
        if not path.exists():
            frame = pd.DataFrame()
            issues.append("missing_file")
        else:
            frame = pd.read_csv(path, low_memory=False)
            missing = sorted(required_columns - set(frame))
            if missing:
                issues.append("missing_columns:" + ",".join(missing))
        if artifact_type == "oof_predictions" and not frame.empty and "target" in frame:
            missing_targets = sorted(
                REQUIRED_WORKFLOW_TARGETS - set(frame["target"].dropna())
            )
            if missing_targets:
                issues.append("missing_targets:" + ",".join(missing_targets))
            if not {"source_id", "source_file", "doi"} & set(frame):
                issues.append("missing_source_provenance")
            if not {
                "evaluation_group_id",
                "modelling_group_id",
                "source_id",
            } & set(frame):
                issues.append("missing_evaluation_group")
        artifact_valid = not issues
        valid &= artifact_valid
        rows.append(
            {
                "artifact_type": artifact_type,
                "artifact": str(path.resolve()),
                "rows": len(frame),
                "sha256": sha256_file(path) if path.exists() else "",
                "valid": artifact_valid,
                "issues": ";".join(issues),
            }
        )
    return pd.DataFrame(rows), valid


class OpenAIWorkflowManager:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        load_project_environment()
        config = load_config()
        self.model = (
            model
            or os.getenv("OPENAI_MODEL")
            or str(config.get("openai", {}).get("model", "gpt-4o-mini"))
        )
        self.client = OpenAI()

    def decide(
        self,
        state: WorkflowState,
        preflight: ArtifactPreflight,
        allowed_actions: list[str],
    ) -> ManagerDecision:
        payload = {
            "workflow_state": state.model_dump(mode="json"),
            "artifact_preflight": preflight.model_dump(mode="json"),
            "allowed_actions": allowed_actions,
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "am_mvt_manager_decision",
                    "strict": True,
                    "schema": ManagerDecision.model_json_schema(),
                }
            },
            temperature=0,
        )
        decision = ManagerDecision.model_validate_json(response.output_text)
        if decision.action not in allowed_actions:
            raise ValueError(f"Manager returned disallowed action: {decision.action}")
        if decision.next_agent != ACTION_AGENT[decision.action]:
            raise ValueError(
                "Manager returned an inconsistent next_agent/action pair: "
                f"{decision.next_agent}/{decision.action}"
            )
        return decision


class WorkflowOrchestrator:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        requested_run_dir: str | Path | None = None,
        existing_artifacts_only: bool = True,
        offline: bool = False,
        resume: bool = False,
    ):
        self.run_id = run_id or new_run_id("workflow")
        self.run_dir = get_path("data", "interim", "agent_runs", self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "workflow_state.json"
        self.ledger = ReactLedger(run_id=self.run_id, run_dir=self.run_dir)
        self.offline = offline
        if resume and self.state_path.exists():
            self.state = WorkflowState.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
        else:
            self.state = WorkflowState(
                run_id=self.run_id,
                existing_artifacts_only=existing_artifacts_only,
                requested_run_dir=str(requested_run_dir or ""),
                evidence_status=(
                    "skipped_by_user_scope"
                    if existing_artifacts_only
                    else "not_started"
                ),
            )
            self._save_state()
            if existing_artifacts_only:
                self.ledger.record(
                    plan_summary="Respect the user scope for the first workflow run.",
                    action_type="skip_evidence",
                    observation_summary="existing_artifacts_only",
                    decision="skipped_by_user_scope",
                )

    def _save_state(self) -> None:
        self.state.updated_at_utc = utc_now()
        self.state_path.write_text(
            self.state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def transition(self, stage: WorkflowStage) -> None:
        if stage == self.state.stage:
            return
        allowed = ALLOWED_TRANSITIONS[self.state.stage]
        if stage not in allowed:
            raise ValueError(
                f"Illegal workflow transition: {self.state.stage.value} -> "
                f"{stage.value}"
            )
        self.state.stage = stage
        self._save_state()

    def _record_decision(
        self,
        decision: ManagerDecision,
        preflight: ArtifactPreflight,
    ) -> None:
        self.ledger.record(
            plan_summary="Route the bounded multi-agent workflow using existing artifacts.",
            action_type=decision.action,
            input_refs=preflight.available_artifacts + preflight.processed_views,
            observation_summary=preflight.status,
            decision=decision.reason_code,
            evidence_refs=decision.required_artifacts,
        )

    def _write_missing_report(self, preflight: ArtifactPreflight) -> Path:
        output = self.run_dir / "missing_artifacts_report.csv"
        rows = [
            {
                "run_id": self.run_id,
                "artifact": artifact,
                "status": "missing",
                "reason": "existing_artifacts_only_preflight",
            }
            for artifact in preflight.missing_artifacts
        ]
        pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
        return output

    def _write_artifact_manifest(self, preflight: ArtifactPreflight) -> Path:
        refs: list[str | Path] = []
        if preflight.run_dir:
            refs.extend(
                Path(preflight.run_dir) / relative
                for relative in preflight.available_artifacts
            )
        refs.extend(preflight.processed_views)
        output = self.run_dir / "artifact_manifest.json"
        output.write_text(
            json.dumps(artifact_manifest(refs), indent=2),
            encoding="utf-8",
        )
        return output

    def preflight(
        self, *, use_openai_manager: bool = True
    ) -> tuple[ArtifactPreflight, ManagerDecision]:
        preflight = inspect_existing_artifacts(self.state.requested_run_dir or None)
        manifest_path = self._write_artifact_manifest(preflight)
        local_decision = deterministic_manager_decision(preflight)
        decision = local_decision
        if use_openai_manager and not self.offline:
            try:
                decision = OpenAIWorkflowManager().decide(
                    self.state,
                    preflight,
                    [local_decision.action],
                )
            except Exception as exc:
                self.ledger.record(
                    plan_summary="Use deterministic routing when the OpenAI manager is unavailable.",
                    action_type="manager_fallback",
                    observation_summary=type(exc).__name__,
                    decision=local_decision.reason_code,
                )
                decision = local_decision
        self._record_decision(decision, preflight)
        self.state.last_action = decision.action
        self.state.reason_code = decision.reason_code
        self.state.artifact_refs = decision.required_artifacts
        self.state.artifact_refs.append(str(manifest_path))
        self.state.resolved_run_dir = preflight.run_dir
        if decision.action == "stop_missing_artifacts":
            report = self._write_missing_report(preflight)
            self.transition(WorkflowStage.BLOCKED_MISSING_ARTIFACTS)
            self.state.artifact_refs.append(str(report))
        elif decision.action == "generate_matrix":
            self.transition(WorkflowStage.MODEL_READY)
            self.state.modelling_status = "existing_evidence_ready"
        elif decision.action == "run_explanation":
            self.transition(WorkflowStage.MODEL_READY)
            self.state.modelling_status = "requires_explanation"
        elif decision.action == "run_modelling":
            self.transition(WorkflowStage.DATA_READY)
            self.state.audit_status = "existing_views_validated"
        self._save_state()
        return preflight, decision

    def execute(
        self,
        *,
        through: WorkflowStage = WorkflowStage.COMPLETE,
        dry_run: bool = False,
        use_openai_manager: bool = True,
    ) -> dict[str, Any]:
        if self.state.stage == WorkflowStage.COMPLETE:
            return self.summary()
        if self.state.stage in {
            WorkflowStage.BLOCKED_MISSING_ARTIFACTS,
            WorkflowStage.APPROVAL_REQUIRED,
            WorkflowStage.FAILED,
        }:
            return self.summary()

        if self.state.stage == WorkflowStage.ARTIFACT_PREFLIGHT:
            preflight, decision = self.preflight(use_openai_manager=use_openai_manager)
        else:
            preflight = inspect_existing_artifacts(
                self.state.resolved_run_dir or self.state.requested_run_dir or None
            )
            decision = deterministic_manager_decision(preflight)
        if (
            dry_run
            or through == WorkflowStage.ARTIFACT_PREFLIGHT
            or self.state.stage == WorkflowStage.BLOCKED_MISSING_ARTIFACTS
        ):
            return self.summary()

        resolved_run = Path(preflight.run_dir) if preflight.run_dir else None
        needs_training = (
            decision.action == "run_modelling"
            or self.state.stage == WorkflowStage.DATA_READY
        )
        if needs_training:
            steward_report, valid = validate_processed_views(preflight.processed_views)
            steward_path = self.run_dir / "data_steward_report.csv"
            steward_report.to_csv(
                steward_path,
                index=False,
                encoding="utf-8-sig",
            )
            self.state.artifact_refs.append(str(steward_path))
            if not valid:
                self.state.reason_code = "existing_processed_views_failed_validation"
                self.state.audit_status = "high_impact_review_required"
                self.state.blocking_human_reviews = int(
                    (~steward_report["valid"]).sum()
                )
                self.transition(WorkflowStage.APPROVAL_REQUIRED)
                return self.summary()
            self.state.audit_status = "approved_records_only_views_validated"
            self._save_state()
            if through == WorkflowStage.DATA_READY:
                return self.summary()

            from am_mvt.modelling.experiment_training import run_experiment_suite

            project_config = load_config().get("multi_agent", {})
            run_name = f"multiagent_{self.run_id}"
            resolved_run = run_experiment_suite(
                run_name=run_name,
                profile=str(project_config.get("modelling_profile", "standard")),
                n_splits=int(project_config.get("cv_folds", 5)),
                mode=str(project_config.get("prediction_mode", "process_only")),
                targets=list(
                    project_config.get(
                        "targets",
                        [
                            "uts_MPa",
                            "yield_strength_MPa",
                            "elongation_percent",
                            "youngs_modulus_GPa",
                            "log10_fatigue_life_cycles",
                        ],
                    )
                ),
            )
            self.state.resolved_run_dir = str(resolved_run)
            self.state.modelling_status = "grouped_oof_models_complete"
            if self.state.stage != WorkflowStage.MODEL_READY:
                self.transition(WorkflowStage.MODEL_READY)

        if self.state.stage == WorkflowStage.MATRIX_READY:
            if through == WorkflowStage.COMPLETE:
                self.transition(WorkflowStage.COMPLETE)
            return self.summary()
        if resolved_run is not None:
            model_report, model_valid = validate_model_run(resolved_run)
            model_report_path = self.run_dir / "model_artifact_validation.csv"
            model_report.to_csv(
                model_report_path,
                index=False,
                encoding="utf-8-sig",
            )
            self.state.artifact_refs.append(str(model_report_path))
            if not model_valid:
                self.state.reason_code = "existing_model_artifacts_failed_validation"
                self.state.audit_status = "high_impact_review_required"
                self.state.blocking_human_reviews = int((~model_report["valid"]).sum())
                self.transition(WorkflowStage.APPROVAL_REQUIRED)
                return self.summary()
        if through == WorkflowStage.MODEL_READY:
            return self.summary()
        if resolved_run is None:
            raise RuntimeError("No experiment run is available for model explanation.")

        required_explanations = [resolved_run / path for path in EXPLANATION_REQUIRED]
        if not all(path.exists() for path in required_explanations):
            from am_mvt.modelling.model_explanation import run_model_explanation

            explanation_outputs = run_model_explanation(
                resolved_run,
                mode="process_only",
            )
            self.state.artifact_refs.extend(
                str(path) for path in explanation_outputs.values()
            )
            self.state.modelling_status = "model_and_step07_evidence_complete"
            self._save_state()

        from am_mvt.optimisation.actionable_matrix import (
            generate_actionable_testing_matrix,
        )

        matrix_outputs = generate_actionable_testing_matrix(resolved_run)
        self.state.artifact_refs.extend(str(path) for path in matrix_outputs.values())
        self.state.matrix_status = (
            "domain_readiness_complete_awaiting_client_targets"
        )
        self.transition(WorkflowStage.MATRIX_READY)

        if through == WorkflowStage.COMPLETE:
            self.transition(WorkflowStage.COMPLETE)
        summary_path = resolved_run / "tables" / "workflow_run_summary.csv"
        self.state.artifact_refs.append(str(summary_path))
        self._save_state()
        pd.DataFrame([self.summary()]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            **self.state.model_dump(mode="json"),
            "state_path": str(self.state_path),
            "ledger_path": str(self.ledger.csv_path),
        }


def artifact_manifest(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for item in paths:
        path = Path(item)
        if path.exists() and path.is_file():
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows
