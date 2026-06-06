# Workflow Contracts

## Stage Status

| Skill | Status |
|---|---|
| source-screening | operational |
| pdf-provenance | operational |
| evidence-grounded-extraction | operational |
| extraction-audit | operational |
| dataset-integration | operational |
| modelling-view-generation | operational |
| model-comparison | operational |
| feature-relevance-relationship-discovery | partially implemented |
| reduced-testing-matrix-recommendation | specification only |

## Canonical Outputs

| Stage | Output |
|---|---|
| Source screening | `data/interim/candidate_sources_llm.csv` |
| PDF provenance | `outputs/tables/source_provenance_audit.csv` |
| PDF manifest | `docs/literature_manifest.csv` |
| Candidate extraction | `data/interim/llm_extracted_records.csv` |
| Extraction audit | `data/interim/llm_extraction_audit_review.csv` |
| Master modelling data | `data/processed/master_modelling_dataset.csv` |
| Merge summary | `data/processed/llm_merge_summary.csv` |
| Modelling views | `data/processed/view_model*.csv` |
| Model evidence | `outputs/tables/project_*.csv` |
| Testing matrix | `outputs/tables/reduced_testing_matrix.csv` |

## Evidence Contract

Literature-derived candidates preserve `source_id`, `source_title`, `doi`,
`source_file`, `page_or_section`, `evidence_text`, `confidence`,
`needs_human_check`, and `record_id` when available. Missing values remain
blank or null.

## Audit Contract

Allowed `audit_status` values:

- `approved`
- `human_review_required`
- `rejected`

Automatic approval requires complete source/evidence keys, confidence of at
least 0.70, `needs_human_check=false`, useful AM or mechanical data, and valid
numeric ranges. Only approved keys enter the master modelling dataset.

Human decisions must preserve `audit_method`, `reviewed_by`, and `reviewed_at`.
