---
name: extraction-audit
description: Audit LLM-derived AM candidate records and assign strict modelling eligibility decisions. Use after evidence-grounded extraction and before dataset integration or model training.
---

# Goal

Assign `approved`, `human_review_required`, or `rejected` without changing the
candidate evidence values.

# Preconditions

- `data/interim/llm_extracted_records.csv` exists.
- Read `../references/workflow-contracts.md` for statuses and required fields.

# Inputs

- Candidate extraction table.
- Parsed chunk or source PDF for optional human review.

# Procedure

1. Check source, record, and evidence identifiers.
2. Validate confidence, human-check flag, useful data, and numeric ranges.
3. Reject unusable or physically invalid records.
4. Route incomplete or uncertain records to human review.
5. Preserve explicit human decisions with reviewer metadata.

# Decision Gates

- Automatically approve only records passing every deterministic requirement.
- Never infer missing evidence during audit.
- Never approve a manually corrected record without an explicit saved decision.

# Outputs

- `data/interim/llm_extraction_audit_review.csv`

# Validation

- Require unique `source_id` plus `record_id`.
- Restrict statuses to the three contract values.
- Keep reasons for every decision.

# Stop Conditions

Stop dataset integration when the audit file is absent, malformed, duplicated,
or contains an unknown status.

# Commands

`python scripts/04b_audit_extractions.py`
