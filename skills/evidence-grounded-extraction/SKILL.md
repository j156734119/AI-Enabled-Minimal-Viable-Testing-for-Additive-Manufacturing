---
name: evidence-grounded-extraction
description: Extract candidate structured metal AM mechanical-testing records while retaining direct evidence and uncertainty. Use when converting parsed PDF text chunks into auditable candidate records with the OpenAI API.
---

# Goal

Extract candidate experimental records from the supplied paper chunk without
inventing, completing, or borrowing data.

# Preconditions

- Process only the supplied source chunk.
- Treat every output as candidate data until extraction audit.
- Read `../references/workflow-contracts.md` when field semantics matter.

# Inputs

- Source PDF filename and chunk identifier.
- Parsed text from `data/interim/text_chunks/`.
- The strict extraction JSON Schema supplied by the caller.

# Procedure

1. Extract only original experimental values explicitly visible in the chunk.
2. Represent one experimental condition and related result per record.
3. Preserve material, AM process, build orientation, surface condition, heat
   treatment, defects, tensile properties, fatigue properties, and loading
   conditions when explicitly stated.
4. Pair S-N stress and fatigue-life values only when their alignment is clear.
5. Include a concise supporting `evidence_text`.
6. Use confidence 0.90-1.00 for clear tables, 0.70-0.89 for clear prose, and
   0.50-0.69 for uncertain candidates requiring review.

# Decision Gates

- Return null for missing or unclear values.
- Do not infer numbers, units, table alignment, or values from domain knowledge.
- Do not extract review statements describing another paper.
- Set `needs_human_check=true` for uncertain units, mappings, identities, or
  table alignment.

# Outputs

- Candidate JSON matching the caller's strict Schema.
- `data/interim/llm_extracted_records.csv`
- `data/interim/llm_extraction_audit.csv`

# Validation

- Use W, mm/s, micrometres, J/mm^3, MPa, GPa, percent, cycles, Hz, and Celsius
  in their corresponding Schema fields.
- Keep stress amplitude and fatigue life paired.
- Return an empty records array when no supported record exists.

# Stop Conditions

Stop extraction for the chunk when source identity is unavailable, the chunk is
unreadable, or no values can be tied to direct evidence.

# Commands

`python scripts/04_extract_with_llm.py --limit 0`
