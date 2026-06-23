# SearchGEO Public Artifact

Public artifact for the paper:

> Yimeng Chen, Zhe Ren, Firas Laakom, Yu Li, Dandan Guo, and Jürgen Schmidhuber. 2026. *How Much Can We Trust LLM Search Agents? Measuring Endorsement Vulnerability to Web Content Manipulation*. arXiv:2606.16821. https://arxiv.org/abs/2606.16821

DOI: https://doi.org/10.48550/arXiv.2606.16821

## Citation

```bibtex
@misc{chen2026searchgeo,
  title = {How Much Can We Trust LLM Search Agents? Measuring Endorsement Vulnerability to Web Content Manipulation},
  author = {Chen, Yimeng and Ren, Zhe and Laakom, Firas and Li, Yu and Guo, Dandan and Schmidhuber, Jürgen},
  year = {2026},
  eprint = {2606.16821},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2606.16821},
  url = {https://arxiv.org/abs/2606.16821}
}
```

This repository contains the public artifact for SearchGEO, a controlled evaluation framework for studying endorsement corruption in LLM-based web-search agents. The artifact supports reproducible defense research while following the paper's dual-use release boundary: it does not publish synthetic attack documents, injected-result contexts, raw copyrighted search-cache contents, live-domain infrastructure, executable payloads, or operational SEO instructions.

This public package is intentionally not a complete rerunnable attack bundle. It provides the code, task definitions, metadata, sanitized outputs, judge labels, and aggregate metrics needed to inspect and reproduce the public analyses. Controlled-access materials, including full per-case bundles with synthetic attack documents or injected-result contexts, are available upon request to bona fide research groups under no-deployment and no-live-SEO terms.

## Contents

- `agents/`, `mock_api/`, `eval_pipeline/`, `defenses/`: agent wrappers, mock search proxy, evaluation pipeline, and defense prompt loader.
- `scripts/`: experiment/evaluation runners and benign query/search-cache utilities. Attack-content generation scripts are intentionally omitted.
- `config/`: model settings, reproducibility settings, and high-level injection-rule configs.
- `tasks/`: 44 main task definitions across health, finance, legal, and consumer IT/product, including reference answers, gold facts, and scoring targets.
- `data/search_cache/generated_queries.yaml`: public query clusters for the main corpus and auxiliary skill-install probe. Raw cached search-result contents are omitted.
- `data/attack_modes/attack_modes.json`: public attack-condition labels and high-level descriptions.
- `results/aggregate_metrics/`: aggregate metric JSON files from the public result sets.
- `results/judge_labels/`: per-case judge labels and scoring metadata for the public result sets.
- `results/model_outputs_sanitized/`: final model answers and high-level metadata with raw search contexts, injected pages, diagnostic reruns, non-public source paths, reasoning traces, and exposed attack-payload strings removed or redacted.
- `docs/EXPERIMENT_STATUS.md`: concise map of the current public experiment state.

For result-file details, see `results/README.md`.

## Result Sets

- `exp1_main_eval`: 13-backend main evaluation. For paper-matching attack-only ASR, use `results/aggregate_metrics/exp1_main_attack_only_summary.json`.
- `exp2_eval`: pipeline-sensitivity evaluation.
- `prompt_defense_eval`: explicit prompt-defense runs.
- `openclaw_eval`: OpenClaw/deployment-default scaffold runs.
- `skillinstall_2b_matched`: auxiliary agent-skill summaries, including the 26-case OpenClaw cross-mode probe and clean-baseline manual labels for OpenClaw false rejection / useful denial.

Per-run `stats_all_*.json` files may include clean-baseline cases in their `overall` field. The main paper reports attack-only ASR for attack-conditioned settings, excluding the clean baseline.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Copy `.env.example` to `.env` or set provider keys in your shell if you intend to run new model calls. Existing public results can be inspected without API keys.

Useful entry points:

```bash
python scripts/run_eval.py --help
python scripts/run_exp1_v2.py --help
python scripts/run_exp2_v2.py --help
```

The included public results are already evaluated: aggregate metrics are under `results/aggregate_metrics/`, per-case judge outputs are under `results/judge_labels/`, and sanitized final answers are under `results/model_outputs_sanitized/`. The `run_eval.py` script is for judging newly generated raw experiment outputs, not for re-judging the sanitized JSONL files directly.

Because raw cached search results and controlled attack documents are not included, full attack replay is not available from this public repository alone. The included code and sanitized artifacts are intended for auditing the evaluation logic, inspecting task/query metadata, inspecting public labels and outputs, and extending defensive evaluations with independently supplied data.

## Release Boundary

Publicly included:

- code, configuration files, documentation, benign query metadata, attack-mode labels, aggregate metrics, judge labels, and sanitized model outputs.

Intentionally omitted:

- `data/attack_content/` and generated synthetic attack pages;
- raw `data/search_cache/` result contents and live fallback caches;
- raw result bundles containing `search_calls.response` or injected-result contexts;
- attack-content generation prompts and cleanup/regeneration scripts;
- live-domain infrastructure, executable payloads, and operational SEO materials.

Full per-case bundles, including synthetic attack documents and injected-result contexts, are available upon request under the restricted research-access terms described in `RELEASE_POLICY.md`.

## License

The public software artifact is released under the MIT License in `LICENSE`. The license applies to included code, configuration, documentation, aggregate metrics, judge labels, and sanitized model-output files. It does not grant rights to raw copyrighted web/search contents or controlled synthetic attack bundles, which are not included in this repository.
