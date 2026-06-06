---
name: dataset-integration
description: Integrate public datasets and strictly approved literature records into the AM master modelling dataset. Use when building or updating the evidence-grounded dataset after extraction audit.
---

# Goal

Build the master modelling dataset without admitting unaudited literature data.

# Preconditions

- Baseline public datasets are available.
- Literature candidates have a valid extraction audit.
- Read `../references/workflow-contracts.md`.

# Inputs

- Public/open baseline datasets.
- `data/interim/llm_extracted_records.csv`
- `data/interim/llm_extraction_audit_review.csv`

# Procedure

1. Build the baseline master dataset.
2. Select literature records whose audit status is `approved`.
3. Join approvals to immutable candidate values by `source_id` and `record_id`.
4. Recalculate engineered features and deduplicate.
5. Write merge counts and evidence coverage.

# Decision Gates

- Stop when the audit is missing or invalid.
- Exclude `human_review_required` and `rejected` records.
- Do not silently overwrite unrelated or user-validated records.

# Outputs

- `data/processed/master_modelling_dataset.csv`
- `data/processed/llm_merge_summary.csv`

# Validation

- Reconcile rows before and after merge.
- Require unique source and record keys.
- Preserve provenance, evidence, confidence, and human-check fields.

# Stop Conditions

Stop on malformed approvals, duplicate audit keys, or unavailable master data.

# Commands

`python scripts/05_build_dataset.py`

`python scripts/05b_merge_llm_into_master.py`
