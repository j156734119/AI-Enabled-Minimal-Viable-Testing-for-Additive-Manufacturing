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
Residual stress indicators
Process signatures

## Example output variables:

- Ultimate tensile strength
- Yield strength
- Elongation
- Fatigue life in cycles
- Log-transformed fatigue life
- Young's / elastic modulus
- Hardness

Failure mode is retained only as a possible future extension. The reviewed
public datasets do not provide consistent structured labels, so it is not used
as a current modelling target or feature.

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

The default CPU-oriented `fast` profile runs only `process_only` with
three-fold GroupKFold. It compares mean, median, alloy-family median, Ridge,
Random Forest, XGBoost, and one lightweight CatBoost configuration. The
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
python scripts/04_extract_with_llm.py --limit 0  # existing JSON outputs are skipped
python scripts/04b_audit_extractions.py          # deterministic admission audit
python scripts/05_build_dataset.py
python scripts/05b_merge_llm_into_master.py      # merges approved records only
python scripts/06_train_models.py --run-name cpu_fast_v1
python scripts/07_explain_models.py --run-dir outputs/experiments/cpu_fast_v1
python scripts/08_generate_testing_matrix.py --run-dir outputs/experiments/cpu_fast_v1
```

Step 01 always uses the OpenAI Responses API with the `web_search` tool:

```
python scripts/01_search_sources.py --target-count 50 --per-journal-limit 8 --min-per-journal 4 --search-rounds 3
```

The source-screening agent searches every approved journal before ranking and
reserves journal coverage when suitable candidates are available. It produces
candidate source tables for manual PDF collection. It does not call Crossref,
automate publisher downloads, or use publisher logins, cookies, VPNs, or
institutional access. If the API returns no valid candidates, existing Step 01
outputs remain unchanged. A successful new search archives the previous
canonical CSV files under `archive/source_search_runs/<utc_timestamp>/`.

Historical datasets, models, and experiment outputs can be reviewed and moved
into the local ignored archive with:

```text
python scripts/archive_legacy_artifacts.py
python scripts/archive_legacy_artifacts.py --apply
```

The first command is a dry run. The applied archive writes
`archive/legacy_20260608/archive_manifest.csv` with source paths, archive paths,
file sizes, modification times, and SHA-256 hashes. The current
`outputs/experiments/cpu_fast_v1/` run is retained in place.

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
standardised filename, verification status, and parsing readiness. It contains
metadata only and does not include or upload the PDF files.

For the current early milestone, stop after:

```
python scripts/06_train_models.py --run-name cpu_fast_v1
```

Step 07 calculates holdout permutation importance, grouped error analysis,
feature coverage, limited sensitivity scans, and a conservative relationship
evidence table. Step 08 converts that evidence into a reduced but
representative testing matrix. Sparse defect, surface, and heat-treatment
regions are marked as requiring validation rather than recommended for test
elimination.

The default command is equivalent to:

```text
python scripts/06_train_models.py --run-name cpu_fast_v1 --profile fast --mode process_only --cv-folds 3
```

The fast profile uses 120 Random Forest trees, 200 XGBoost estimators, and one
CatBoost model with depth 6, at most 300 iterations, and 30-round early
stopping. The Basquin residual branch reuses this lightweight CatBoost.

Run the auxiliary reduced-testing mode separately when needed:

```text
python scripts/06_train_models.py --run-name reduced_fast_v1 --profile fast --mode reduced_testing
```

Run the original full comparison explicitly:

```text
python scripts/06_train_models.py --run-name full_v1 --profile standard --mode all
```

Step 06 uses a DOI-first, dataset-ID fallback group split. It reserves a 20%
final holdout and selects ordinary models using grouped-CV mean MAE: three
folds for `fast` and five folds for `standard`. The final holdout is evaluated
once. Each run is isolated under:

```text
outputs/experiments/<run_name>/
```

The run contains configuration, fold and summary metrics, Basquin parameters,
physical monotonicity checks, model artifacts, and a model registry. Existing
legacy metrics and models are not overwritten.

After training, batch-predict proposed experiment scenarios with:

```text
python scripts/06b_predict_scenarios.py --run-dir outputs/experiments/cpu_fast_v1 --input examples/prediction_scenarios_template.csv --output outputs/experiments/cpu_fast_v1/scenario_predictions.csv --mode all
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

## Important Notes

Raw PDFs, confidential files, API keys, and large intermediate files should not be committed to GitHub.

The OpenAI API is used only as an assistive NLP tool for candidate information extraction from lawful public sources. Extracted records must be validated before being used for analysis.

## Licence

This repository is released under the MIT License for the code written by the author. External datasets and papers remain under their original licences.
