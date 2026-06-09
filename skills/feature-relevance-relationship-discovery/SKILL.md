---
name: feature-relevance-relationship-discovery
description: Interpret AM model associations, feature relevance, physical sanity checks, and coverage risk conservatively. Use after model comparison when preparing evidence for testing prioritisation.
---

# Goal

Identify supported variable relationships without making unsupported causal
claims.

# Preconditions

- Model metrics, pipelines, modelling views, and audited literature are available.
- A completed Step 06 run with ordinary model artifacts is available.

# Inputs

- Step 06 metrics, model artifacts, and modelling views.

# Procedure

1. Calculate holdout permutation importance on original input features.
2. Report errors by alloy family, process, orientation, and surface condition.
3. Calculate feature coverage and limited sensitivity scans.
4. Check direction against physical sanity summaries.
5. Separate model support from literature support.

# Decision Gates

- Use `associated with`, `related to`, and `may influence`.
- Lower confidence for sparse, unstable, or conflicting evidence.

# Outputs

- Ranked variables, relationship candidates, coverage-risk notes, grouped
  errors, sensitivity tables, and figures.

# Validation

- Label the evidence type and coverage for every reported relationship.

# Stop Conditions

Stop interpretation when model performance or coverage cannot support ranking.

# Commands

`python scripts/07_explain_models.py`
