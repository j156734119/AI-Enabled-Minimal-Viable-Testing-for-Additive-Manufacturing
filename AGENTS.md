# Project Agent Instructions

These instructions apply to the whole repository.

## Project positioning
- Treat this as an MSc dissertation codebase for **AI-Enabled Minimal Viable Testing for Additive Manufacturing**.
- Keep the research contribution focused on an **agent-assisted, skill-based, evidence-grounded workflow** that supports a **reduced but representative mechanical testing strategy**.
- Do not reframe the project as a generic mechanical-property prediction competition.

## Source and legal/ethics boundary
- Do not request, store, script, or automate use of university credentials, publisher credentials, passwords, cookies, session tokens, VPN sessions, or institutional access tokens.
- `.env` may contain OpenAI configuration only, for example `OPENAI_API_KEY` and `OPENAI_MODEL`.
- Agents may screen candidate literature and record metadata, but the user must manually obtain PDFs or supplementary files through lawful routes.
- Prefer open-access papers, public datasets, and publicly available supplementary materials.
- If a source appears to require institutional subscription access, mark it as `manual_download_required` or `university_subscription` rather than attempting automated download.

## Evidence-grounded data rule
- Do not claim an extracted record comes from a paper unless source metadata and evidence are retained.
- Each literature-derived candidate record should preserve, where available: `source_id`, `source_title`, `doi`, `source_file`, `page_or_section`, `evidence_text`, `confidence`, and `needs_human_check`.
- Do not infer missing numerical values from domain knowledge. Leave missing values blank/null.
- Treat LLM extraction as candidate data until it is audited or human-checked.

## Skill-based workflow
Use the repository skills as bounded task specifications:
1. Source screening
2. PDF provenance
3. Evidence-grounded extraction
4. Extraction audit
5. Dataset integration
6. Modelling view generation
7. Model comparison
8. Feature relevance and relationship discovery
9. Reduced testing matrix recommendation

Skills describe rules and outputs; scripts provide reproducible execution. Keep those two layers aligned when changing the workflow.

## Modelling guidance
- Use multiple tabular-regression baselines because the dataset combines numerical process parameters, categorical AM descriptors, small/medium sample sizes, and possible nonlinear interactions.
- Keep baseline models in the training pipeline so dissertation results can separate real predictive signal from trivial mean/linear baselines.
- Prefer robust, reproducible defaults over heavy hyperparameter searches unless explicitly requested.

## Writing and terminology
- Use conservative language: "associated with", "related to", "may influence", and "supports prioritisation".
- Avoid unsupported causal claims such as "causes" unless the source evidence is experimental and directly supports causality.
- Say "agent-assisted" or "semi-automated" rather than "fully automatic" for literature search, extraction, and curation.
