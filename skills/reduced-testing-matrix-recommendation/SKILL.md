---
name: reduced-testing-matrix-recommendation
description: Translate AM model evidence, coverage, and domain risk into a reduced but representative mechanical testing matrix. Use only after audited data, model comparison, and feature interpretation are available.
---

# Goal

Prioritise a reduced physical testing matrix while preserving validation in
sparse or high-risk regions.

# Preconditions

- Feature relevance, coverage, metrics, and audited sources are available.
- Step 07 relationship and coverage outputs are available.

# Inputs

- Model evidence, coverage risk, and audited domain evidence.

# Procedure

1. Rank `alloy x AM process` domains across all five model targets using row,
   source, group, feature-coverage, and OOF-model evidence.
2. When client property thresholds are absent, stop at domain readiness and
   label any budget plans as evidence-validation only.
3. After client thresholds are supplied, build static tensile conditions from observed medoid records; never create
   a synthetic median process combination.
4. Count one tensile specimen once while treating UTS and yield as primary
   outcomes and elongation and modulus as auxiliary observations.
5. Build the E466-style client example as complete S-N blocks at 80, 95, 110,
   125, and 140 MPa stress amplitude, with three replicates per level and
   separate 0 and 90 degree blocks.
6. Report AFT median life, distribution quantiles, calibrated probabilities of
   reaching 10M and 20M cycles, route agreement, monotonicity, and machine time.
7. Treat 10M-20M cycles as an external plausibility audit only; never use it as
   a forced training target or multiply predictions to match it.
8. Keep EOS-like turned/as-manufactured evidence separate from a client
   machined/stress-relieved condition. Unsupported surface or heat-treatment
   fields make that exact condition not assessable.
9. Apply every OOF, local-error, interval, coverage, OOD, monotonicity, route,
   and source-diversity gate as a separate Boolean column.
10. Score information value and use deterministic greedy set coverage for
   independent static and fatigue budgets.
11. Preserve validation tests for sparse or high-risk regions rather than
   presenting them as reduction opportunities.

# Decision Gates

- Static formal reduction requires a Green domain and all configured gates.
- Weak fatigue models may produce `pilot_validation` or `retain_validation`,
  but never a false `candidate_for_reduction`.
- Final holdout metrics do not determine reduction eligibility.

# Outputs

- `alloy_process_domain_readiness.csv` and `domain_priority_shortlist.csv`
- `client_target_template.csv`
- `client_case_<case_id>_static_<budget>.csv`, fatigue matrix, and case summary
- `condition_evidence_static.csv` and `condition_evidence_fatigue.csv`
- Static plans for 24, 36, and 48 specimens.
- Fatigue plans for 30, 45, and 60 specimens.
- `matrix_summary.csv` and `matrix_change_log.csv`
- Fatigue protocol audit, regime summary, route comparison, threshold
  calibration, external benchmark check, and model-promotion audit.

All files are written under `outputs/experiments/<run_name>/tables/`. Exact
condition evidence is a downstream audit, not the primary readiness result.

# Validation

- Require reason, supporting features, model evidence, coverage risk,
  confidence level, and validation need for every recommendation.

# Stop Conditions

When evidence is insufficient, output blockers and next validation tests. Stop
only when required existing artifacts are absent.

# Commands

`python scripts/08_generate_testing_matrix.py --run-dir outputs/experiments/<run_name>`

Client-targeted pilot case:

`python scripts/generate_client_target_case.py --run-dir outputs/experiments/<run_name> --target-file examples/client_target_alsi10mg_lpbf.csv`

Legacy comparison:

`python scripts/08_generate_testing_matrix.py --run-dir outputs/experiments/<run_name> --legacy`
