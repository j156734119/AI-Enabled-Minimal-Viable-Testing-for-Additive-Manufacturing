# Model Comparison Skill

## Purpose
Train a compact set of tabular regression models so the dissertation can compare a trivial baseline, a regularised linear model, bagging, and gradient boosting.

## Inputs
- `data/processed/view_model1_uts.csv`
- `data/processed/view_model2_sn_fatigue.csv`
- `data/processed/view_model3_elongation_yield.csv`
- `data/processed/view_model4_elastic_modulus.csv`

## Outputs
- `outputs/tables/project_regression_model_metrics.csv`
- `outputs/tables/project_feature_importance.csv`
- `outputs/tables/project_training_errors.csv`
- `outputs/tables/project_best_model_summary.csv`
- `outputs/tables/project_training_data_audit.csv`
- `outputs/models/*.joblib`

## Candidate model families
- Dummy mean baseline: confirms whether learned models beat a trivial predictor.
- Ridge: conservative regularised linear baseline.
- Random Forest: nonlinear bagging model for mixed tabular data.
- XGBoost: nonlinear boosting model for structured process-property data.

## Rules
- Keep random seeds fixed.
- Prefer moderate model sizes before any expensive hyperparameter search.
- Keep model metrics comparable by using the same split for all models per target.

## Validation checks
- Each target should report train/test row counts, split method, MAE, RMSE, R2, improvement over Dummy, and model path.
- Do not describe R2 as classification accuracy.
- If a model fails for a target, record the error rather than failing silently.

## Related scripts
- `scripts/06_train_models.py`
- `src/am_mvt/modelling/train_regression.py`
