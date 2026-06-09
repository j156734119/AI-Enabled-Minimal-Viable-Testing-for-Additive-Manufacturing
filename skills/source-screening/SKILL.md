---
name: source-screening
description: Screen and rank lawful candidate literature and datasets for metal additive-manufacturing mechanical testing. Use when searching approved journals, public datasets, or supplementary materials before PDF collection.
---

# Goal

Produce a traceable candidate-source list for agent-assisted minimal viable
testing research without claiming access to unavailable full text.

# Preconditions

- Use only the approved journal scope, date range, and search focus.
- Read `../references/workflow-contracts.md` when output paths or fields matter.

# Inputs

- Approved journals and date range.
- Metal AM tensile, fatigue, process, defect, surface, and post-processing terms.

# Procedure

1. Search every approved journal before final ranking.
2. Prefer original studies likely to contain extractable numerical data.
3. Score relevance, data richness, impact, and selection on a 0-10 scale.
4. Deduplicate by DOI, then by normalised title.
5. Reserve journal coverage when suitable candidates exist.

# Decision Gates

- Exclude sources outside the approved journal scope.
- Never use or request credentials, cookies, VPN sessions, or access tokens.
- Mark restricted sources as `manual_download_required` or
  `university_subscription`.
- Do not claim a PDF is downloaded unless a lawful local file exists.

# Outputs

- `data/interim/candidate_sources_llm.csv`
- `outputs/tables/source_screening_candidates_top50.csv`
- `outputs/tables/source_screening_journal_scope.csv`

# Validation

- Require a title and at least one DOI, URL, or dataset locator.
- Require a relevance reason and access classification.
- Confirm score bounds and journal-scope coverage.

# Stop Conditions

Stop and record the failure when the approved scope is unavailable, a search
cannot be completed, or a result cannot be tied to a source locator.

# Commands

`python scripts/01_search_sources.py`
