# Agent API Use and Cost Planning

This project uses agents as bounded workflow helpers. Most agents do not need the OpenAI API; they can operate with deterministic Python scripts and local files.

## Agents/skills that may call OpenAI

| Agent or skill | OpenAI API needed? | Why |
|---|---:|---|
| Source screening | Optional | An LLM can classify abstracts or relevance notes, but deterministic metadata search can run without it. |
| PDF provenance | No | File matching, hashes, and inventory checks are deterministic. |
| Evidence-grounded extraction | Yes | The LLM extracts candidate structured records from parsed PDF chunks. |
| Extraction audit | Optional | Deterministic completeness checks should run first; an LLM can optionally draft audit comments. |
| Dataset integration | No | Schema mapping and merging are deterministic. |
| Modelling view generation | No | View generation is deterministic. |
| Model comparison | No | scikit-learn/XGBoost training is local. |
| Feature relevance and relationship discovery | No by default | Model-side analysis should be deterministic; an LLM may help draft dissertation narrative only. |
| Reduced testing matrix recommendation | Optional | Rule-based generation should be deterministic; an LLM may help draft explanations after evidence tables exist. |

## Cost estimation method

Estimated cost = `(input_tokens / 1,000,000 * input_price) + (output_tokens / 1,000,000 * output_price)`.

As a planning assumption, one parsed PDF text chunk may use about 3,500-6,000 input tokens and 500-1,000 output tokens depending on chunk length and number of extracted records.

## Practical extraction scenarios

| Scenario | Approximate workload | Approximate tokens | Example low-cost model estimate |
|---|---|---|---|
| Pilot | 5 PDFs × 8 chunks = 40 calls | 200k input + 30k output | Low single-digit USD with a mini/nano model. |
| Small dissertation batch | 20 PDFs × 10 chunks = 200 calls | 1.0M input + 150k output | Usually a few USD with a mini/nano model; higher with flagship models. |
| Larger batch | 50 PDFs × 12 chunks = 600 calls | 3.6M input + 450k output | Still manageable with mini/nano models, but validate on a pilot first. |

## Cost control rules

- Run a 3-5 PDF pilot first and inspect `data/interim/llm_extraction_audit.csv` before scaling.
- Prefer cheaper models for source screening and first-pass extraction.
- Use a stronger model only for difficult table chunks or audit disagreements.
- Limit chunk counts during testing with `python scripts/04_extract_with_llm.py --limit 20`.
- Never submit full PDFs when parsed chunks are sufficient.
- Never include credentials, cookies, or subscription-only access tokens in prompts or files.
