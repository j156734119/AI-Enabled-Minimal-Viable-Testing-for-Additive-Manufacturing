---
name: model-comparison
description: Train and compare compact reproducible tabular regression baselines for approved AM modelling views. Use when evaluating Dummy, Ridge, Random Forest, and XGBoost across dissertation targets.
---

# Goal

Compare learned models against trivial and linear baselines without turning the
dissertation into a broad prediction competition.

# Preconditions

- Task-specific modelling views exist.
- Read `../references/workflow-contracts.md`.

# Inputs

- Four approved modelling views.

# Procedure

1. Use fixed group-aware splits and random seeds.
2. Train Dummy mean, Ridge, Random Forest, and XGBoost.
3. Record metrics, failures, model files, and feature importance.
4. Compare MAE and RMSE against Dummy.

# Decision Gates

- Require minimum rows and usable features.
- Record model failures instead of silently dropping them.
- Avoid expensive hyperparameter searches by default.

# Outputs

- Model metrics, best-model summary, training audit, errors, feature importance.
- `outputs/models/*.joblib`

# Validation

- Report train/test rows, groups, sources, MAE, RMSE, R2, and split method.
- Describe R2 as variance explained, not classification accuracy.

# Stop Conditions

Stop a target when minimum rows or usable features are unavailable.

# Commands

`python scripts/06_train_models.py`
