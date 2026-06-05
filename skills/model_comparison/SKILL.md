# Model Comparison Skill

## Purpose
Train a controlled set of tabular regression models so the dissertation can compare simple baselines, linear regularised models, kernel models, bagging ensembles, and gradient boosting methods.

## Inputs
- `data/processed/view_model1_static.csv`
- `data/processed/view_model2_sn_fatigue.csv`

## Outputs
- `outputs/tables/project_regression_model_metrics.csv`
- `outputs/tables/project_feature_importance.csv`
- `outputs/tables/project_training_errors.csv`
- `outputs/models/*.joblib`

## Candidate model families
- Dummy mean baseline: confirms whether learned models beat a trivial predictor.
- Ridge and ElasticNet: robust linear baselines for small, noisy tabular datasets.
- Support Vector Regression: useful for small/medium datasets with nonlinear relationships.
- Random Forest and Extra Trees: robust bagging ensembles for mixed tabular data and feature interactions.
- Gradient Boosting, HistGradientBoosting, and XGBoost: strong nonlinear tabular baselines for process-property relationships.

## Rules
- Keep random seeds fixed.
- Prefer moderate model sizes before any expensive hyperparameter search.
- Keep model metrics comparable by using the same split for all models per target.

## Validation checks
- Each target should report train/test row counts, split method, MAE, RMSE, R2, and model path.
- If a model fails for a target, record the error rather than failing silently.

## Related scripts
- `scripts/06_train_models.py`
- `src/am_mvt/modelling/train_regression.py`
