---
name: modelling-view-generation
description: Build provenance-preserving task-specific regression views from the approved AM master dataset. Use before model comparison for UTS, S-N fatigue, elongation and yield, or Young's modulus.
---

# Goal

Create four reproducible modelling views from approved master data.

# Preconditions

- `data/processed/master_modelling_dataset.csv` exists.
- Read `../references/workflow-contracts.md`.

# Inputs

- Approved master modelling dataset.

# Procedure

1. Build separate UTS, S-N fatigue, elongation/yield, and modulus views.
2. Keep provenance and modelling group identifiers.
3. Limit repeated S-N points per dataset and apply equal-dataset weighting.
4. Keep records from one paper or experiment in the same split group.
5. Exclude failure mode and fracture origin from every modelling view.

# Decision Gates

- Exclude rows without the relevant target.
- Prevent repeated fatigue joins from dominating static-property views.
- Keep failure-mode classification as future work.

# Outputs

- Four `data/processed/view_model*.csv` files.
- `data/processed/model_view_summary.csv`

# Validation

- Report rows, source counts, group counts, and target coverage.
- Require usable features and targets for each non-empty view.

# Stop Conditions

Stop a task when its target or independent-group coverage is insufficient.

# Commands

`python scripts/06_train_models.py`
