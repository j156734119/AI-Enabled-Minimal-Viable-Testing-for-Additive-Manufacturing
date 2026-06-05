# Modelling View Generation Skill

## Purpose
Build task-specific modelling views from the master dataset.

## Inputs
- `data/processed/master_dataset.csv`

## Outputs
- `data/processed/view_model1_static.csv`
- `data/processed/view_model2_sn_fatigue.csv`
- View summary outputs

## Rules
- Keep provenance columns wherever possible.
- Model 1 should focus on static/tensile targets.
- Model 2 should focus on S-N fatigue-life targets.
- Avoid duplicate fatigue curve joins dominating static-property views.

## Validation checks
- Report row counts, source counts, and non-missing target counts for each view.
- Confirm each view contains usable feature columns and target columns.

## Related scripts
- `scripts/06_train_models.py`
- `src/am_mvt/modelling/build_views.py`
