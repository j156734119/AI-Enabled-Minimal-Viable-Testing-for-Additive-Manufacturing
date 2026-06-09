---
name: reduced-testing-matrix-recommendation
description: Translate AM model evidence, coverage, and domain risk into a reduced but representative mechanical testing matrix. Use only after audited data, model comparison, and feature interpretation are available.
---

# Goal

Prioritise a reduced physical testing matrix while preserving validation in
sparse or high-risk regions.

# Preconditions

- Feature relevance, coverage, metrics, and audited sources are available.
- Step 07 relationship and coverage outputs are available.

# Inputs

- Model evidence, coverage risk, and audited domain evidence.

# Procedure

1. Combine feature relevance with coverage and model quality.
2. Increase testing priority when an important region has weak coverage.
3. Treat fatigue, defects, porosity, surface, stress amplitude, and R-ratio
   conservatively.
4. Assign confidence and validation requirements.
5. Preserve validation tests for sparse defect, surface, and heat-treatment
   regions rather than presenting them as reduction opportunities.

# Decision Gates

- Never recommend eliminating tests in sparse or high-risk regions.
- Lower confidence when model performance is weak.

# Outputs

- `outputs/tables/reduced_testing_matrix.csv`
- Optional report under `outputs/reports/`

# Validation

- Require reason, supporting features, model evidence, coverage risk,
  confidence level, and validation need for every recommendation.

# Stop Conditions

Stop recommendation when evidence or coverage is insufficient.

# Commands

`python scripts/08_generate_testing_matrix.py`
