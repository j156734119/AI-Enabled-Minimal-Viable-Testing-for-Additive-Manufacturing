# AI-Enabled Minimal Viable Testing for Additive Manufacturing

This repository contains the Python framework for Fangxing Lin's MSc dissertation project:

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

The project studies relationships between:

```text
process / material / defect variables
        ↓
mechanical outcomes
        ↓
reduced but representative testing strategy
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

Example output variables:

Tensile properties
Fatigue properties
Mechanical outcomes
Failure modes

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

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
pip install -e .

Create a local environment file:

copy .env.example .env

Then fill in the real values in .env.

Workflow

Run the project step by step:

python scripts/01_search_sources.py
python scripts/02_download_open_files.py
python scripts/03_parse_documents.py
python scripts/04_extract_with_llm.py
python scripts/05_build_dataset.py
python scripts/06_train_models.py
python scripts/07_explain_models.py
python scripts/08_generate_testing_matrix.py
Important Notes

Raw PDFs, confidential files, API keys, and large intermediate files should not be committed to GitHub.

The OpenAI API is used only as an assistive NLP tool for candidate information extraction from lawful public sources. Extracted records must be validated before being used for analysis.

Licence

This repository is released under the MIT License for the code written by the author. External datasets and papers remain under their original licences.


---

# 7. `src/am_mvt/__init__.py`

```python
"""
am_mvt

Python package for the dissertation project:
AI-Enabled Minimal Viable Testing for Additive Manufacturing.
"""

__version__ = "0.1.0"