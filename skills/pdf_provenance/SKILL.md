# PDF Provenance Skill

## Purpose
Link PDFs manually provided by the user to source metadata before parsing or extraction.

## Inputs
- `data/raw/pdfs/*.pdf`
- `data/interim/candidate_sources.csv`

## Outputs
- `data/interim/pdf_inventory.csv`

Recommended columns: `source_id`, `local_pdf_filename`, `source_title`, `doi`, `journal`, `year`, `pdf_sha256`, `match_status`, `manual_check_required`, `notes`.

## Rules
- Do not download restricted PDFs automatically.
- Do not assume a PDF's DOI or title without evidence from metadata, filename, first page, or user input.
- Use `manual_check_required=true` when matching is uncertain.

## Validation checks
- Every parsed PDF should have a file hash.
- Every extracted PDF should have a source_id or be explicitly flagged as unmatched.

## Related scripts
- `scripts/02_download_open_files.py`
- `scripts/03_parse_documents.py`
