# Reduced Testing Matrix Recommendation Skill

## Purpose
Translate model evidence, coverage evidence, and domain risk rules into a reduced but representative mechanical testing matrix.

## Inputs
- Feature relevance outputs
- Coverage summaries
- Model metrics
- Audited source metadata

## Outputs
- `outputs/tables/reduced_testing_matrix.csv`
- Optional narrative report under `outputs/reports/`

Recommended columns: `priority`, `alloy_family`, `am_process`, `build_orientation`, `surface_condition`, `test_type`, `target_property`, `recommended_test_condition`, `reason`, `supporting_features`, `model_evidence`, `coverage_risk`, `confidence_level`, `needs_validation`.

## Rules
- High feature importance plus sufficient coverage can support inclusion in the reduced matrix.
- High feature importance plus weak coverage should usually increase testing priority rather than justify test reduction.
- Fatigue recommendations should treat porosity, defect type, surface condition, stress amplitude, and R-ratio conservatively.
- Low model performance should lower confidence and increase validation requirements.

## Validation checks
- Every recommendation should have a reason, supporting features, model evidence, coverage risk, and confidence level.
- Do not recommend eliminating tests in sparse or high-risk data regions.

## Related scripts
- `scripts/08_generate_testing_matrix.py`
