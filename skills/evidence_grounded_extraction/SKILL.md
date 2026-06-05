# Evidence-Grounded Extraction Skill

## Purpose
Extract candidate structured AM mechanical testing records from parsed text chunks while retaining source evidence.

## Inputs
- Parsed text chunks under `data/interim/text_chunks/`
- PDF/source metadata where available
- Project extraction schema

## Outputs
- `data/interim/llm_extracted_records.csv`
- `data/interim/llm_extraction_audit.csv`

## Rules
- Extract only values explicitly stated in text, tables, or figure captions.
- Do not infer missing numerical values from domain knowledge.
- Leave unclear values blank/null.
- Keep concise `evidence_text` for each candidate record.
- Preserve `source_file`, `page_or_section`, `doi` or `source_title` where available.
- Set `needs_human_check=true` when units, table alignment, source identity, or field mapping are uncertain.

## Validation checks
- Candidate records without evidence text should not be treated as verified records.
- Records with confidence below 0.70 should require human checking.
- Unit conversions should be checked before merging into modelling views.

## Related scripts
- `scripts/03_parse_documents.py`
- `scripts/04_extract_with_llm.py`
- `scripts/05b_merge_llm_into_master.py`
