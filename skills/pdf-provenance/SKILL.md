---
name: pdf-provenance
description: Match lawfully obtained local PDFs to source metadata and prepare provenance records before parsing. Use when inventorying, renaming, verifying, or preparing research PDFs.
---

# Goal

Link each local PDF to defensible source metadata before parsing or extraction.

# Preconditions

- PDFs were obtained manually through lawful routes.
- Candidate metadata is available when possible.
- Read `../references/workflow-contracts.md` for canonical outputs.

# Inputs

- `data/raw/pdfs/inbox/*.pdf`
- Candidate-source tables from source screening.

# Procedure

1. Inspect filename, embedded metadata, first-page title, and DOI evidence.
2. Preview title normalisation before moving any file.
3. Match PDFs to candidate metadata and export the literature manifest.
4. Flag uncertain identity for human review.

# Decision Gates

- Do not automate publisher or subscription downloads.
- Do not assert title or DOI without file or metadata evidence.
- Keep unmatched files out of extraction until reviewed.

# Outputs

- `outputs/tables/source_provenance_audit.csv`
- `outputs/tables/local_pdf_inventory.csv`
- `outputs/tables/pdf_title_normalisation_plan.csv`
- `docs/literature_manifest.csv`

# Validation

- Require a stable local filename and source identifier before extraction.
- Preserve match method, score, and human-check status.

# Stop Conditions

Stop before parsing when identity is unresolved or the file is unreadable.

# Commands

`python scripts/02b_prepare_pdfs.py`

`python scripts/02b_prepare_pdfs.py --apply`

`python scripts/02c_export_literature_manifest.py`
