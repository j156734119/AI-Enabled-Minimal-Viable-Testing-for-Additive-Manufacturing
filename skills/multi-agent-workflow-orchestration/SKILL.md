---
name: multi-agent-workflow-orchestration
description: Route the bounded Evidence, Data Steward, Modelling, and Testing Decision agents over existing AM project artifacts without allowing arbitrary API actions.
---

# Goal

Coordinate the project skills while keeping deterministic scripts responsible
for validation, modelling, and matrix generation.

# Agent Roles

- Manager: choose one strict-schema action from the local whitelist.
- Evidence Agent: screen and extract literature only outside
  `existing_artifacts_only`; record `skipped_by_user_scope` otherwise.
- Data Steward Agent: validate schema, provenance, audit status, unique keys,
  and file hashes. Only approved literature records may enter modelling.
- Modelling Agent: run process-only grouped CV, OOF predictions, separate
  fatigue routes, intervals, and Step 07 diagnostics.
- Testing Decision Agent: apply traceable condition gates, information value,
  and deterministic budgeted set coverage.

# Safety Boundary

The OpenAI Responses API returns only `next_agent`, `action`, `reason_code`,
`required_artifacts`, and `blocking_review`. The local orchestrator never runs
API-provided shell commands and accepts only the locally permitted action.

# Existing-Artifacts-Only Mode

1. Check the explicit experiment run.
2. Otherwise check the most recent model run.
3. Otherwise validate all four existing processed modelling views.
4. If all are missing, write `missing_artifacts_report.csv` and stop.
5. Never call source screening, PDF extraction, or dataset expansion as a
   fallback.

# Outputs

- `data/interim/agent_runs/<run_id>/workflow_state.json`
- `data/interim/agent_runs/<run_id>/react_ledger.csv`
- `data/interim/agent_runs/<run_id>/data_steward_report.csv`
- `outputs/experiments/<run_name>/tables/workflow_run_summary.csv`

# Command

`python scripts/run_multiagent_workflow.py --existing-artifacts-only --run-dir <run>`
