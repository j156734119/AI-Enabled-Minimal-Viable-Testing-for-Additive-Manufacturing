from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from am_mvt.config import get_path


LEDGER_COLUMNS = [
    "timestamp_utc",
    "plan_summary",
    "action_type",
    "input_refs",
    "observation_summary",
    "decision",
    "evidence_refs",
]


@dataclass
class ReactLedgerEvent:
    plan_summary: str
    action_type: str
    input_refs: list[str] = field(default_factory=list)
    observation_summary: str = ""
    decision: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        row["input_refs"] = json.dumps(self.input_refs, ensure_ascii=False)
        row["evidence_refs"] = json.dumps(self.evidence_refs, ensure_ascii=False)
        return {column: str(row.get(column, "")) for column in LEDGER_COLUMNS}


def new_run_id(prefix: str = "agent") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


class ReactLedger:
    """
    Persist a ReAct-style action log without recording chain-of-thought.

    The ledger stores auditable summaries of the plan, action, observation, and
    decision. It should not contain hidden reasoning or full model traces.
    """

    def __init__(self, run_id: str | None = None, run_dir: str | Path | None = None):
        self.run_id = run_id or new_run_id()
        self.run_dir = Path(run_dir) if run_dir is not None else get_path(
            "data",
            "interim",
            "agent_runs",
            self.run_id,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "react_ledger.csv"
        self.jsonl_path = self.run_dir / "react_ledger.jsonl"

    def record(
        self,
        *,
        plan_summary: str,
        action_type: str,
        input_refs: list[str] | None = None,
        observation_summary: str = "",
        decision: str = "",
        evidence_refs: list[str] | None = None,
    ) -> ReactLedgerEvent:
        event = ReactLedgerEvent(
            plan_summary=plan_summary,
            action_type=action_type,
            input_refs=input_refs or [],
            observation_summary=observation_summary,
            decision=decision,
            evidence_refs=evidence_refs or [],
        )
        self._append(event)
        return event

    def _append(self, event: ReactLedgerEvent) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=LEDGER_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(event.to_row())

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def record_human_download_boundary(
    ledger: ReactLedger,
    *,
    candidate_refs: list[str],
    observed_pdf_refs: list[str],
) -> None:
    ledger.record(
        plan_summary=(
            "Respect copyright and access-control boundaries while preparing "
            "local evidence for extraction."
        ),
        action_type="human_download_required",
        input_refs=candidate_refs,
        observation_summary=(
            "The system does not download publisher PDFs, use credentials, "
            "or bypass access controls."
        ),
        decision="wait_for_manual_legal_pdf_acquisition",
        evidence_refs=[],
    )
    ledger.record(
        plan_summary="Continue extraction only from local PDFs supplied by the user.",
        action_type="local_pdf_observed",
        input_refs=observed_pdf_refs,
        observation_summary=f"Observed {len(observed_pdf_refs)} local PDF-derived chunks.",
        decision="process_local_evidence_only",
        evidence_refs=observed_pdf_refs,
    )
