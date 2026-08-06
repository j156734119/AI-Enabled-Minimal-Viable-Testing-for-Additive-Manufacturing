# AI-Enabled Minimal Viable Testing for Additive Manufacturing

This repository contains a research framework for:

**AI-Enabled Minimal Viable Testing for Additive Manufacturing**

The project investigates how publicly available secondary data can support an AI-assisted framework for reduced but representative mechanical testing in metal additive manufacturing.

## Project Scope

This project uses publicly available secondary data only, including:

- Open datasets
- Academic journal articles
- Review papers
- Supplementary materials
- Literature-derived structured datasets

The project does not involve:

- Human participants
- Interviews
- Questionnaires
- Confidential organisational data

## Core Research Logic

The project studies relationships between material, process, defect, surface, post-processing and testing variables, and numerical mechanical outcomes.

The main logic is:

```text
material / process / defect / testing variables
→ mechanical outcomes
→ feature importance and sensitivity analysis
→ reduced but representative testing strategy
```

## Example input variables:

Alloy type
AM process
Build orientation
Surface condition
Processing parameters
Porosity metrics
Defect type
Process signatures

## Example output variables:

- Ultimate tensile strength
- Yield strength
- Elongation
- Fatigue life in cycles
- Log-transformed fatigue life
- Young's / elastic modulus

Failure mode, fracture origin, hardness, and residual stress are retained only
as future extensions. The reviewed data do not provide sufficiently consistent
structured coverage for these variables, so they are not current modelling
targets or features.

## Current Modelling Tasks

The current early-stage milestone is to complete the workflow up to model training for four dissertation-aligned tasks:

```text
Model 1: UTS prediction
Model 2: S-N fatigue life prediction
Model 3: elongation / yield response prediction
Model 4: Young's / elastic modulus prediction
```

`process_only` is the dissertation's main result and excludes previously
measured mechanical properties. `reduced_testing` may use selected measured
properties and is reported as an auxiliary or diagnostic result.

The default `balanced` profile runs only `process_only` with five-fold
GroupKFold. It compares mean, median, alloy-family median, an L2-regularised
SGD linear baseline, Random Forest, XGBoost, one lightweight CatBoost
configuration, and a three-layer MLP neural-network baseline. The
CPU-oriented `fast` profile uses three folds and omits the MLP. The
optional `standard` profile retains four conservative CatBoost configurations,
five-fold CV, and either or both prediction modes.

Fatigue additionally compares ordinary failure-only log-life regression,
hierarchical Basquin, Basquin plus a CatBoost residual correction, and
XGBoost-AFT. AFT retains runouts as right-censored observations rather than
treating the runout threshold as a failure life.

## Repository Structure
```text
data/
    raw/          original public data and downloaded open files
    interim/      parsed text, text chunks, and extraction outputs
    processed/    cleaned structured datasets

src/am_mvt/
    ingestion/    metadata search and open data loading
    parsing/      PDF/text/table parsing
    extraction/   NLP-assisted structured extraction
    cleaning/     unit conversion and schema validation
    modelling/    regression/classification models
    optimisation/ testing matrix reduction
    utils/        shared utilities

scripts/
    executable workflow steps

tests/
    automated tests

outputs/
    figures, tables, models, and reports
```

## Setup

Create and activate a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
pip install -e .
```

Create a local environment file:

```
copy .env.example .env
```

Then fill in the real values in .env.

## Workflow

Run the project step by step:

```
python scripts/01_search_sources.py          # OpenAI Responses API agent web search
python scripts/02_download_open_files.py     # prepare folders and provenance audit tables
python scripts/02b_prepare_pdfs.py            # preview PDF title normalisation
python scripts/02b_prepare_pdfs.py --apply    # move renamed PDFs into data/raw/pdfs
python scripts/02c_export_literature_manifest.py  # export GitHub-safe article list
python scripts/03_parse_documents.py
python scripts/03b_build_evidence_index.py   # optional local-only RAG index over active chunks
python scripts/04_extract_with_llm.py --limit 0  # extract all pending chunks; successful JSON is skipped
# or rebuild the combined CSV from existing active JSON without API calls:
python scripts/04_extract_with_llm.py --combine-only
python scripts/04b_audit_extractions.py          # deterministic admission audit
python scripts/05_build_dataset.py
python scripts/05b_merge_llm_into_master.py      # merges approved records only
python scripts/06_train_models.py --run-name balanced_v2
python scripts/07_explain_models.py --run-dir outputs/experiments/balanced_v2
python scripts/08_generate_testing_matrix.py --run-dir outputs/experiments/balanced_v2
```

Run the bounded multi-agent workflow without searching for literature or
adding records:

```text
python scripts/run_multiagent_workflow.py --existing-artifacts-only --run-dir outputs/experiments/balanced_v2
```

Use `--offline` for deterministic local manager routing, `--dry-run` for
artifact preflight only, `--resume --run-id <id>` to continue an unfinished
run, and `--through <stage>` to stop after a named workflow stage. When no
existing run or complete processed views can be found, the workflow writes a
missing-artifact report and stops; it does not start literature search.

Step 01 always uses the OpenAI Responses API with the `web_search` tool:

```
python scripts/01_search_sources.py --target-count 50 --per-journal-limit 8 --min-per-journal 1 --search-rounds 3
```

The source-screening agent searches every approved journal before ranking and
reserves journal coverage when suitable candidates are available. It produces
candidate source tables for manual PDF collection. It does not call Crossref,
automate publisher downloads, or use publisher logins, cookies, VPNs, or
institutional access. If the API returns no valid candidates, existing Step 01
outputs remain unchanged. A successful new search archives the previous
canonical CSV files under `archive/source_search_runs/<utc_timestamp>/`.
The Metals search uses the journal-specific `mdpi.com/2075-4701` path and a
focused replenishment request if the first pass returns no valid Metals paper.

Historical datasets, models, experiment outputs, and generated meeting
documents can be reviewed and moved into the local ignored archive with:

```text
python scripts/archive_legacy_artifacts.py --keep-experiment balanced_v2
python scripts/archive_legacy_artifacts.py --keep-experiment balanced_v2 --apply
```

Repeat `--keep-experiment` to retain multiple canonical runs. Add
`--include-documents`, `--include-document-code`, and
`--include-local-working-artifacts` to archive generated Word/PDF files,
meeting-document helper scripts, and temporary document workspaces. The first
command is a dry run. Archive names use UTC timestamps by default, and every
moved file is recorded in `archive_manifest.csv` with its source path, archive
path, size, modification time, and SHA-256.

Place newly downloaded PDFs in:

```text
data/raw/pdfs/inbox/
```

Step 02b reads PDF metadata, first-page title text, and DOI evidence. It
prioritises exact DOI/candidate-title matches and prepares compact filenames
that remain stable for text chunk generation, such as:

```text
001_addma_2019_316l_fatigue_orientation_surface_roughness.pdf
```

Run it without `--apply` first and review:

```text
outputs/tables/pdf_title_normalisation_plan.csv
```

Only the `--apply` run moves renamed PDFs from the inbox into
`data/raw/pdfs/`, where Step 03 will parse them.

After the formal PDF folder is ready, export the reproducibility manifest:

```bash
python scripts/02c_export_literature_manifest.py
```

This scans every PDF directly under `data/raw/pdfs/` and writes:

```text
docs/literature_manifest.csv
```

The manifest lists article title, journal, year, DOI, source links, local
standardised filename, content SHA-256, canonical source, duplicate status,
verification status, and parsing readiness. Step 03 parses canonical PDFs only,
uses the repository skip list, and writes
`data/interim/active_chunk_manifest.csv`. Steps 04 and 04b ignore JSON outputs
that are not in that active manifest. The literature manifest contains
metadata only and does not include or upload the PDF files.

For a training-only run, stop after:

```
python scripts/06_train_models.py --run-name balanced_v2
```

Step 06 also writes row-aligned OOF predictions with source, evaluation group,
fold, condition fields, prediction error, and interval evidence. Step 07
calculates holdout permutation importance, grouped error analysis,
feature and combination coverage, limited sensitivity scans, SHAP explanations,
one diagnostic B2 combination holdout per target, and a conservative
relationship evidence table. Step 08 first ranks alloy-process domains across
all five targets and writes a client-target template. Until client thresholds
are supplied, budget files are explicitly evidence-validation plans rather
than target-driven reduction matrices. UTS and yield remain the primary static
decision targets while elongation and modulus are auxiliary tensile outputs.
It produces independent 24/36/48 static specimen plans and 30/45/60 fatigue
specimen plans. Fatigue uses complete five-level, three-replicate S-N blocks;
weak models produce validation plans rather than false reduction claims.
Sparse defect, surface, and heat-treatment regions are marked as requiring
validation rather than recommended for test elimination. Use `--legacy` to
retain the former Step 08 output for dissertation comparison.

The default command is equivalent to:

```text
python scripts/06_train_models.py --run-name balanced_v2 --profile balanced --mode process_only --cv-folds 5
```

The balanced profile uses 160 Random Forest trees, 240 XGBoost estimators, one
lightweight CatBoost, and an MLP with hidden layers 128, 64, and 32. The MLP is
an additional comparison model and is not forced to become the selected model.
It is preceded by an isolated NumPy/BLAS probe so a broken Windows numerical
runtime cannot terminate the five formal target-training jobs.

Run the auxiliary reduced-testing mode separately when needed:

```text
python scripts/06_train_models.py --run-name reduced_fast_v1 --profile fast --mode reduced_testing
```

Run the original full comparison explicitly:

```text
python scripts/06_train_models.py --run-name full_v1 --profile standard --mode all
```

Step 06 uses a DOI-first, dataset-ID and source-ID fallback group split. It
reserves a 15% final holdout and uses five-fold GroupKFold on the remaining 85%
for the default balanced profile. Ordinary models are selected primarily by
grouped-CV out-of-fold R-squared, with RMSE and MAE as tie-breakers. The final
holdout is evaluated once. Candidate features must have at least 20 non-missing
rows and 1% coverage
in the task frame, preventing extremely sparse fields from destabilising fold
preprocessing. Each run is isolated under:

```text
outputs/experiments/<run_name>/
```

The run contains configuration, fold and summary metrics, Basquin parameters,
physical monotonicity checks, model artifacts, and a model registry. Existing
legacy metrics and models are not overwritten.

Step 04 normally extracts parsed text chunks. When a PDF has no usable text
layer, it sends the original PDF as a Responses API `input_file`, allowing a
vision-capable model to inspect page images. Existing successful text
extractions remain incremental; prior empty text-only results for scanned PDFs
are eligible for one PDF-vision retry.

### Human-in-the-loop ReAct and local-only RAG

The literature workflow is semi-automated by design:

```text
Agent screens public candidate literature
-> researcher manually obtains PDFs through lawful routes
-> local PDFs are parsed and chunked
-> local evidence chunks may be retrieved with lightweight TF-IDF RAG
-> LLM extraction writes candidate records with source/page evidence
-> rule-based audit and human review decide admission
```

The system does not automate publisher PDF downloads, use credentials, cookies,
VPN sessions, or institutional access tokens. PDF acquisition is an explicit
human-in-the-loop action. The ReAct-style ledger records auditable
`action_type`, `observation_summary`, `decision`, and evidence references under
`data/interim/agent_runs/<run_id>/`; it does not store hidden chain-of-thought.

Build the optional local evidence index after Step 03:

```text
python scripts/03b_build_evidence_index.py
```

Query local evidence chunks for demonstrations or manual checks:

```text
python scripts/query_evidence_rag.py "Ti-6Al-4V fatigue stress amplitude data" --top-k 5
```

Use RAG priority for low-cost Step 04 trials or newly added PDFs:

```text
python scripts/04_extract_with_llm.py --use-rag-priority --rag-top-k-per-source 3 --limit 20
```

RAG priority only reorders or limits the pending local chunks selected for
extraction. It does not replace the active manifest, source evidence, audit
gate, or manual review.

After training, batch-predict proposed experiment scenarios with:

```text
python scripts/06b_predict_scenarios.py --run-dir outputs/experiments/balanced_v2 --input examples/prediction_scenarios_template.csv --output outputs/experiments/balanced_v2/scenario_predictions.csv --mode all
```

The output reports 90% out-of-fold conformal intervals for ordinary and
Basquin routes. AFT reports a censor-aware point estimate and is kept separate;
the three fatigue routes are not automatically averaged.

## Evidence Audit Gate

OpenAI extraction produces candidate evidence, not verified modelling data.
After Step 04, run:

```text
python scripts/04b_audit_extractions.py
```

This writes:

```text
data/interim/llm_extraction_audit_review.csv
```

Each record receives `approved`, `human_review_required`, or `rejected`.
Step 05b stops if this audit is absent or malformed and admits only approved
`source_id` plus `record_id` keys. Candidate values remain in
`data/interim/llm_extracted_records.csv`; the audit file supplies decisions,
not replacement evidence values.

The project skills under `skills/` are dual-use task specifications. Codex uses
their YAML metadata for discovery, while OpenAI API callers inject only the
skill body into the system prompt.

Step 04 is incremental by default: existing successful JSON outputs are
skipped, while chunks without output and prior API-error outputs are selected
for extraction. `--limit` applies to that pending set, and `--overwrite` is
required to deliberately rerun successful extractions.

## Important Notes

Raw PDFs, confidential files, API keys, and large intermediate files should not be committed to GitHub.

The OpenAI API is used only as an assistive NLP tool for candidate information extraction from lawful public sources. Extracted records must be validated before being used for analysis.

## Licence

This repository is released under the MIT License for the code written by the author. External datasets and papers remain under their original licences.
