# Feature Relevance and Relationship Discovery Skill

## Purpose
Identify variables and variable groups that may be important for AM mechanical outcomes, while separating model association from causal claims.

## Inputs
- Model metrics and trained pipelines
- Modelling views
- Feature importance or permutation importance outputs when available
- Domain notes from audited literature

## Outputs
- Ranked important variables
- Candidate relationship summaries
- Coverage-risk notes for variables and groups

## Rules
- Use conservative language: `associated with`, `related to`, or `may influence`.
- Do not make direct causal claims from secondary data alone.
- Mark weak or sparse evidence as requiring validation.
- Use results to support reduced but representative testing, not to replace physical validation.

## Validation checks
- Check whether each relationship has model support, literature/domain support, or both.
- Check whether coverage is sufficient before treating a variable as reliable for recommendation.

## Related scripts
- `scripts/07_explain_models.py`
