# Experiment Status

This document summarizes the public experiment state included in this artifact, excluding the paper itself and excluding controlled-access attack documents or injected search contexts.

## Main Corpus

- 44 high-stakes recommendation queries across four domains: health, finance, legal, and consumer IT/product.
- Public task definitions are in `tasks/{health,finance,legal,product}.yaml`.
- Public query clusters are in `data/search_cache/generated_queries.yaml`.
- Synthetic attack documents and raw search-cache contents are intentionally omitted from this public artifact.

## Public Attack Conditions

Attack-condition labels are listed in `data/attack_modes/attack_modes.json`. The paper describes a five-mode taxonomy; the public main matrix contains six attack conditions because Mode 1A is split into instruction-payload and factual-payload variants.

- `baseline`
- `1A`
- `1A_fact`
- `1B`
- `2A_institutional`
- `2B`
- `3`

## Completed Public Result Sets

- `exp1_main_eval`: 13-backend main evaluation on the 308-case matrix.
- `exp2_eval`: pipeline-sensitivity evaluation on the Gemini backend, using source counts N=1,2,3 and N=3 rank-position variants.
- `prompt_defense_eval`: explicit prompt-defense comparison for Gemini, DeepSeek-V4-Flash, and MiniMax-M2.7.
- `openclaw_eval`: deployment-default/OpenClaw scaffold comparison for Gemini, DeepSeek-V4-Flash, and MiniMax-M2.7.
- `skillinstall_2b_matched`: auxiliary agent-skill summary. Auxiliary query clusters are included in `data/search_cache/generated_queries.yaml`; the OpenClaw 26-case cross-mode probe includes sanitized per-case query/result details, while other auxiliary per-case injected contexts are kept out of the public package.

## File Types

- Aggregate metrics are under `results/aggregate_metrics/`.
- Per-case judge labels are under `results/judge_labels/`.
- Sanitized model outputs are under `results/model_outputs_sanitized/` as JSONL files.

The sanitized output files preserve final model answers and high-level metadata while removing retrieved result bodies, injected pages, raw `search_calls.response` payloads, and reasoning traces.
