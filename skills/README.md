# Skill-Based Agent Workflow

This folder defines the bounded skills used by the project agents. The skills are intended to make the workflow auditable and repeatable, not to let an agent freely search, download, or infer data.

| Skill | Main purpose | Typical script connection | OpenAI API use |
|---|---|---|---|
| Source screening | Find and rank candidate AM literature/data sources within user-approved journals and datasets. | `scripts/01_search_sources.py` | Optional if an LLM is used to classify abstracts; otherwise no. |
| PDF provenance | Match user-provided PDFs to source metadata and calculate file provenance. | `scripts/02_download_open_files.py`, `scripts/03_parse_documents.py` | No. |
| Evidence-grounded extraction | Extract explicit AM experimental records from parsed text chunks. | `scripts/04_extract_with_llm.py` | Yes. |
| Extraction audit | Check whether candidate records retain source/evidence/confidence flags. | `scripts/04_extract_with_llm.py`, `scripts/05b_merge_llm_into_master.py` | Optional for LLM-assisted audit comments; deterministic checks should run first. |
| Dataset integration | Merge accepted candidate records into the master dataset. | `scripts/05_build_dataset.py`, `scripts/05b_merge_llm_into_master.py` | No. |
| Modelling view generation | Produce task-specific modelling views. | `scripts/06_train_models.py` | No. |
| Model comparison | Train multiple tabular regression models for robustness checks. | `scripts/06_train_models.py` | No. |
| Feature relevance and relationship discovery | Identify important variables and relationship candidates after training. | `scripts/07_explain_models.py` | No by default. |
| Reduced testing matrix recommendation | Convert evidence, coverage, and model signals into a reduced testing matrix. | `scripts/08_generate_testing_matrix.py` | Optional for narrative drafting only; recommendations should be rule/evidence based. |

## Core rule
Every literature-derived record must be traceable back to source metadata and evidence text. Agent outputs are candidate data until audited.
