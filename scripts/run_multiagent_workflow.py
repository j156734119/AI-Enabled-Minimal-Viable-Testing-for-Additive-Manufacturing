"""Run the bounded multi-agent workflow over existing project artifacts."""

from __future__ import annotations

import argparse
import json

from am_mvt.agent.workflow import WorkflowOrchestrator, WorkflowStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate existing approved artifacts, run grouped modelling when "
            "needed, and generate actionable static and fatigue test matrices."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Preferred existing experiment run directory.",
    )
    parser.add_argument(
        "--existing-artifacts-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable literature search and new record extraction (default).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic local manager routing without an API call.",
    )
    parser.add_argument(
        "--through",
        choices=[stage.value for stage in WorkflowStage],
        default=WorkflowStage.COMPLETE.value,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orchestrator = WorkflowOrchestrator(
        run_id=args.run_id,
        requested_run_dir=args.run_dir,
        existing_artifacts_only=args.existing_artifacts_only,
        offline=args.offline,
        resume=args.resume,
    )
    summary = orchestrator.execute(
        through=WorkflowStage(args.through),
        dry_run=args.dry_run,
        use_openai_manager=not args.offline,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if summary["stage"] == "blocked_missing_existing_artifacts" else 0


if __name__ == "__main__":
    raise SystemExit(main())
