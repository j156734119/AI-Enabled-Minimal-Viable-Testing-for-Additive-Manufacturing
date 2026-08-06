---
name: model-comparison
description: Train grouped, leakage-controlled AM regression benchmarks plus censor-aware and physics-anchored fatigue routes.
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

1. Use dataset-ID-first groups for fatigue S-N curves and DOI-first groups for
   static targets, with source and record identifiers as fallbacks.
2. Default to the CPU-oriented `fast` profile: `process_only`, three-fold
   GroupKFold, and one lightweight CatBoost candidate.
3. Use `standard --mode all` only when the full five-fold, dual-mode,
   four-CatBoost comparison is required.
4. Compare Dummy mean/median, alloy-family median, Ridge, Random Forest,
   XGBoost, and conservative CatBoost candidates using mean CV MAE.
5. Route fatigue by protocol: conventional force-controlled records at no more
   than 200 Hz are the ASTM E466-style route; records at 1000 Hz or above are
   ultrasonic VHCF; missing or intermediate frequencies are isolated.
6. Keep ordinary regression and hierarchical Basquin as failure-only
   diagnostics. Use monotonic XGBoost-AFT with explicit right-censored runouts
   as the formal high-cycle decision route.
7. Compare AFT normal, logistic, and extreme distributions at configured
   scales using grouped OOF AFT loss, with C-index as a secondary metric.
8. Do not train an ASTM E606-style model without total, plastic, and elastic
   strain-amplitude data; report the route as not assessable.
9. Generate 90% OOF conformal intervals for ordinary and Basquin routes.
10. Evaluate the final holdout once and retain physical monotonicity checks.

# Decision Gates

- Require minimum rows and usable features.
- Record model failures instead of silently dropping them.
- Avoid expensive hyperparameter searches by default.

# Outputs

- Run configuration, CV metrics, final holdout metrics, model registry,
  Basquin parameters, physical checks, and model artifacts.
- `outputs/experiments/<run_name>/`

# Validation

- Report CV mean/std and final test MAE, RMSE, R2, and split method.
- Report AFT negative log-likelihood and Harrell C-index.
- Verify unknown runout states never enter AFT as observed failures.
- Verify 20 kHz data never enter the conventional E466-style model.
- Verify negative Basquin slopes and monotonic non-increasing stress scans.
- Describe R2 as variance explained, not classification accuracy.

# Stop Conditions

Stop a target when minimum rows or usable features are unavailable.

# Commands

Fast default:

`python scripts/06_train_models.py --run-name <name>`

Full comparison:

`python scripts/06_train_models.py --run-name <name> --profile standard --mode all`
