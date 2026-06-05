# Dataset Integration Skill

## Purpose
Merge audited records into the project master dataset without contaminating modelling views with weak evidence.

## Inputs
- Baseline public/open datasets
- `data/interim/llm_extracted_records.csv`
- Audit status fields

## Outputs
- `data/processed/master_dataset.csv`
- `data/processed/modelling_dataset.csv`
- Merge/audit summary tables

## Rules
- Do not merge candidate rows that have no useful AM/mechanical testing information.
- Preserve source and audit fields after merging.
- Keep `needs_human_check` available for downstream filtering.
- Do not silently overwrite user-validated rows.

## Validation checks
- Count rows before and after merge.
- Count rows with evidence text and confidence.
- Check source_id and record_id deduplication.

## Related scripts
- `scripts/05_build_dataset.py`
- `scripts/05b_merge_llm_into_master.py`
