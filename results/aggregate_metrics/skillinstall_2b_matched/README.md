# Skill-Install Matched 2B Results

This directory contains public aggregate and sanitized per-case support files for the auxiliary agent-skill analysis.

Files:

- `summary.json`: Reproducible aggregate summary for the matched 18-case Mode 2B table and the OpenClaw cross-mode table.
- `openclaw_cross_mode_cases.json`: Per-case query, target metadata, backend outputs, search-query strings, and judge scores for the 26-case OpenClaw cross-mode probe. Raw search-result bodies and injected contexts are omitted.
- `openclaw_clean_baseline_manual_labels.json`: Per-case clean-baseline OpenClaw labels supporting the false-rejection and useful-denied counts in the auxiliary probe. It includes sanitized model outputs but omits raw search-result bodies, injected contexts, and reasoning traces.
- `eval/hermes_supplemental_2b_stats_all_20260522_105539.json`: Aggregate stats for the supplemental Hermes 2B judge run. Per-case Hermes auxiliary judge outputs are omitted from the public package.

Main matched Mode 2B direct-ASR counts:

| Ecosystem | Claude | GPT-5.4-mini | GPT-5.5 |
|---|---:|---:|---:|
| OpenClaw | 0/6 | 6/6 | 6/6 |
| Anthropic Skills | 0/6 | 5/6 | 4/6 |
| Hermes Agent | 0/6 | 6/6 | 6/6 |
| Total | 0/18 | 17/18 | 16/18 |

OpenClaw cross-mode direct-ASR counts:

| Mode | Claude | GPT-5.4-mini | GPT-5.5 |
|---|---:|---:|---:|
| 1A | 0/4 | 4/4 | 3/4 |
| 1B | 0/4 | 4/4 | 4/4 |
| 2A_institutional | 0/4 | 4/4 | 4/4 |
| 2B | 0/10 | 10/10 | 10/10 |
| 3 | 0/4 | 4/4 | 4/4 |
| Total | 0/26 | 26/26 | 25/26 |

OpenClaw clean-baseline manual labels:

| Condition | Outcome | Claude | GPT-5.4-mini |
|---|---:|---:|---:|
| High-stakes clean | False rejection, reported-table coding | 8/10 | 0/10 |
| High-stakes clean | Clear platform-fabrication sublabel | 7/10 | 0/10 |
| High-stakes clean | Useful denied | 10/10 | 0/10 |
| Read-only clean | False rejection, reported-table coding | 0/4 | - |
| Read-only clean | Useful denied | 3/4 | - |
