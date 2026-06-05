# Extraction Audit Skill

## Purpose
Audit LLM-derived candidate records before they are merged into the master dataset.

## Inputs
- `data/interim/llm_extracted_records.csv`
- `data/interim/llm_extraction_audit.csv`
- Parsed text chunks or source PDFs when manual review is needed

## Outputs
- `data/interim/llm_extraction_audit_review.csv`

## Rules
- Prioritise traceability over quantity.
- Do not approve records without source and evidence metadata.
- Flag numeric values that are not visible in, or directly supported by, the evidence text.
- Keep uncertain records as `needs_human_check=true`.

## Validation checks
- Check for missing `source_file`, `evidence_text`, `confidence`, and `needs_human_check`.
- Check whether expected units match project schema units.
- Check duplicate `source_id` + `record_id` pairs.

## Related scripts
- `scripts/04_extract_with_llm.py`
- `scripts/05b_merge_llm_into_master.py`
