---
name: feature-relevance-relationship-discovery
description: Interpret AM model associations, feature relevance, physical sanity checks, and coverage risk conservatively. Use after model comparison when preparing evidence for testing prioritisation.
---

# Goal

Identify supported variable relationships without making unsupported causal
claims.

# Preconditions

- Model metrics, pipelines, modelling views, and audited literature are available.
- This stage is partially implemented; `scripts/07_explain_models.py` remains a
  placeholder for advanced explanation outputs.

# Inputs

- Metrics, feature importance, modelling views, and coverage summaries.

# Procedure

1. Compare feature signals across supported models.
2. Check direction against physical sanity summaries.
3. Record data coverage and source diversity.
4. Separate model support from literature support.

# Decision Gates

- Use `associated with`, `related to`, and `may influence`.
- Lower confidence for sparse, unstable, or conflicting evidence.

# Outputs

- Ranked variables, relationship candidates, and coverage-risk notes.

# Validation

- Label the evidence type and coverage for every reported relationship.

# Stop Conditions

Stop interpretation when model performance or coverage cannot support ranking.

# Commands

`python scripts/07_explain_models.py`
