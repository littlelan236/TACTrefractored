# Implementation status

Updated: 2026-10-14. This describes implemented capabilities and saved products,
not a claim that any job is currently running.

## Implemented

- Strict experiment resolution and immutable run configuration/provenance.
- Native vLLM baseline and registry/EasySteer steering backends, model-bound axes.
- Shared task ContainerFactory, dependency/bootstrap policy, images and timeouts.
- Agent rollouts, fresh infrastructure retries, resume and shared-task coordination.
- Scoped SWE-bench 4.1.0 harness evaluation and trajectory usage/reporting.
- DeepSeek trajectory annotation implementation, state handoff and script loop cutoff.
- Serving-tokenizer enrichment, annotated-prefix hidden extraction and verification.
- TACT mean-difference, LDA contrast and Fisher axis fitting capabilities.
- Canonical workflow fitting TACT/LDA and launching TACT CAP/GATE, plus LDA queue.
- Optional action-cycle rollout intervention, off by default.
- Project and pinned vendor regression tests; recorded historical backend validation.

## Canonical products

- Qwen3.5-9B baseline: 258 trajectories; split 127 train / 64 val / 67 eval.
- Final train/val labels: 191 tasks and 15,560 labels.
- Enrichment: 15,559 recoverable labeled states, one unavailable state.
- Hidden: 191 NPZ files, verification complete; TACT and LDA bundles exist.
- Original TACT and LDA CAP/GATE: 67 rollout results in each arm.
- Additional TACT/LDA additive runs exist; paused loop-intervention trials are separate.

See [baseline workflow](BASELINE-WORKFLOW.md) for paths, commands and stage scope.
Harness failure counts and resolve rates must be read from the chosen evaluation
summary; rollout completion does not guarantee grading completion or resolution.

## Current boundaries

- Research preparation requires canonical trajectories and annotations beforehand;
  `--stage all` does not perform baseline rollout or API annotation.
- LDA queue SOURCE/TARGET IDs are configured in script source, not CLI flags.
- Enrichment replaces existing hidden and prepared-label products.
- EasySteer architecture and scheduling boundaries are documented in
  [EASYSTEER-BACKEND.md](EASYSTEER-BACKEND.md).
- Synthetic graph/mechanism tests establish implementation behavior, not benchmark gains.
- The 2026-10-14 script cleanup retained five pipeline/setup scripts. Historical
  smoke, dataset-preparation and analysis scripts are no longer current entries.
- `experiments/` was empty at inventory; do not assume historical config/statistics
  paths exist. Actual formal run data remains under `runs/`.

Registry and EasySteer validation reports retain dated evidence from earlier
sessions; their historic test counts do not constitute a new test run today.
