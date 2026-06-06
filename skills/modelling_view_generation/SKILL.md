# Modelling View Generation Skill

## Purpose
Build task-specific modelling views from the master dataset.

## Inputs
- `data/processed/master_modelling_dataset.csv`

## Outputs
- `data/processed/view_model1_uts.csv`
- `data/processed/view_model2_sn_fatigue.csv`
- `data/processed/view_model3_elongation_yield.csv`
- `data/processed/view_model4_elastic_modulus.csv`
- View summary outputs

## Rules
- Keep provenance columns wherever possible.
- Model 1 should focus on UTS.
- Model 2 should focus on S-N fatigue life.
- Model 3 should focus on elongation and yield strength.
- Model 4 should focus on Young's modulus.
- Avoid duplicate fatigue curve joins dominating static-property views.
- Keep records from the same paper or experimental dataset in the same train/test split.

## Validation checks
- Report row counts, source counts, and non-missing target counts for each view.
- Confirm each view contains usable feature columns and target columns.

## Related scripts
- `scripts/06_train_models.py`
- `src/am_mvt/modelling/build_views.py`
