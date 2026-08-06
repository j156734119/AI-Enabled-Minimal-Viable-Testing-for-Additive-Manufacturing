# Skill-Based Agent Workflow

These standard project skills are usable by Codex and as bounded OpenAI API
task specifications. API callers load only the `SKILL.md` body; Codex uses the
frontmatter and `agents/openai.yaml` for discovery.

| Skill | Status | OpenAI API |
|---|---|---|
| multi-agent-workflow-orchestration | operational | Manager routing only |
| source-screening | operational | Yes |
| pdf-provenance | operational | No |
| evidence-grounded-extraction | operational | Yes |
| extraction-audit | operational | No; deterministic first |
| dataset-integration | operational | No |
| modelling-view-generation | operational | No |
| model-comparison | operational | No |
| feature-relevance-relationship-discovery | partially implemented | No |
| reduced-testing-matrix-recommendation | operational | No |

Shared paths, evidence fields, and audit statuses are defined in
`references/workflow-contracts.md`.

Every literature-derived record remains candidate evidence until audited. Only
records with `audit_status=approved` may enter modelling data.
