# Source Screening Skill

## Purpose
Screen user-approved additive manufacturing journals, public datasets, and supplementary materials for candidate sources relevant to mechanical properties and minimal viable testing.

## Inputs
- User-approved journal list and date range.
- Search keywords for metal AM, tensile properties, fatigue life, process parameters, defects, porosity, surface condition, and post-processing.

## Outputs
- `data/interim/candidate_sources.csv`

Recommended columns: `source_id`, `title`, `journal`, `year`, `doi`, `url`, `pdf_url`, `access_type`, `priority_tier`, `relevance_reason`, `local_pdf_filename`, `download_status`, `notes`.

## Rules
- Do not use or request university credentials, passwords, cookies, or institutional tokens.
- Do not mark paywalled material as downloaded or usable unless the user manually provides it through lawful access.
- Prefer open-access papers, open datasets, and public supplementary files.
- Mark uncertain access as `manual_download_required`.

## Validation checks
- Each candidate must have a title and at least one locator: DOI, URL, or dataset link.
- Each row must include a relevance reason.
- Sources outside the user-approved scope should be excluded or clearly marked as out-of-scope.

## Related scripts
- `scripts/01_search_sources.py`
- `scripts/02_download_open_files.py`
