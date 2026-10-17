# Baseline preparation and TACT/LDA steering

Updated: 2026-10-14. Run commands from the project root. This guide describes the
existing code and artifacts; updating documentation does not launch an experiment.

## Canonical inputs

The canonical baseline is Qwen3.5-9B, sourced from
`baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012`.

| Input | Location | Membership |
|---|---|---|
| Original trajectories | `baseline/traj/<task>.traj.json` | 258 tasks |
| Final annotations | `baseline/annotation/<task>.jsonl` | 127 train + 64 val |
| Annotation audit records | `baseline/annotation/jobs/` | Extra records, not extra tasks |
| Partition configuration | `baseline/config/splits.json` | 127 axis-train / 64 axis-val / 67 eval |
| Baseline configuration snapshot | `baseline/config/resolved-config.json` | Exact baseline model and runtime |
| Membership/source integrity | `baseline/manifest.json` | Sources and hashes |

Trajectories are flat files, not partition subdirectories. The split file and
manifest define membership. The 191 final annotation files contain 15,560 labels;
recursive JSONL counts also include audit copies. Labels may cover only the
annotated prefix after the long-loop cutoff.

`configs/experiment.yaml` is a general default and currently names
Qwen3.6-35B-A3B. It is not the configuration of this canonical Qwen3.5-9B baseline.
Axis bundles are bound to the checkpoint/tokenizer; do not exchange models while
keeping an old bundle.

## Baseline rollout and harness

Baseline execution is implemented by the package CLI, not an extra script.
Prepare a new raw experiment configuration with the intended model, dataset,
`serving.backend=native_vllm` and `steering.strategy=none`, then use a new run ID:

```bash
.venvs/runner/bin/python -m tact --config configs/NEW_BASELINE.yaml config check
.venvs/runner/bin/python -m tact --config configs/NEW_BASELINE.yaml run --run-id NEW_BASELINE_ID
```

`NEW_*` names are placeholders. `run` performs rollout, harness evaluation and
reporting. A rollout-only run can use `--no-evaluate`. New runs write to `runs/`;
they do not automatically replace the canonical `baseline/` archive.

Resume a rollout or score an existing run separately:

```bash
.venvs/runner/bin/python -m tact rollout --run-id EXISTING_RUN_ID --resume-run
.venvs/runner/bin/python -m tact evaluate --run-id EXISTING_RUN_ID --eval-id NEW_EVAL_ID
.venvs/runner/bin/python -m tact report --run-id EXISTING_RUN_ID
```

Existing-run commands use the immutable `resolved-config.json`; config overrides
require a new run. An evaluation ID must not already exist. Evaluation outputs
are `runs/<run>/evaluation/<eval-id>/{status,progress,summary}.json`, with harness
and environment logs below the same directory.

`Submitted` is the rollout exit status, not proof that tests pass. `resolved`
comes from harness outcomes. The denominator includes missing patches.
`complete=false` can indicate harness failures even when all task outcomes have
been recorded. Summary existence or an old PID file does not establish a live job.

## Annotation prerequisite

Before the research workflow, populate canonical trajectories, final annotations,
split configuration and manifest. The retained workflow does not generate labels
or automatically assemble a new canonical baseline.
Annotation implementation remains in `src/tact/research/deepseek_annotation/`.

For newly scheduled annotation runs, the agreed limit is 3 retained loop steps.
Historical annotations may contain prefixes made under earlier limits. Pass the
intended cap explicitly; module defaults are not evidence of the historical
annotation setting. This is separate from rollout loop intervention and from
the statistical definition of terminal single-step loops.

## Enrichment, hidden extraction and axes

```bash
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage preflight
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage enrichment
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage hidden
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage axes
```

Preflight checks all 191 annotated tasks, contiguous label prefixes and trajectory
hashes. Enrichment re-executes with `.venvs/serving/bin/python` so tokenization
matches serving. It reconstructs full request prefixes, verifies prompt token
counts and server IDs where available, then adds reasoning and the think-close
boundary. Missing/incomplete states are recorded, not fabricated.

**Enrichment replaces `baseline/hidden/` and `baseline/prepared-labels/`.** Run it
only when intentionally regenerating preparation data. `--stage all` also starts
with this replacement; it is not a resume or progress-inspection command.

| Product | Location |
|---|---|
| Enrichment completion record | `baseline/enrichment/completed.json` |
| Decision inputs | `baseline/hidden/<task>.decisions.json` |
| Hidden tensors / extraction records | `baseline/hidden/<task>.npz`, `<task>.extraction.json` |
| Hidden membership / verification | `baseline/hidden/manifest.json`, `verification.json` |
| Fit labels | `baseline/prepared-labels/labels/<task>.jsonl` and manifest |
| Preparation configuration | `baseline/config/workflow-research.json` |
| TACT / LDA bundles | `baseline/axes/tact/`, `baseline/axes/lda/` |

The current completed enrichment records 15,559 recoverable states from 15,560
labels, with one unavailable state. There are 191 hidden NPZ files and hidden
verification is complete. These are historical product counts, not live progress.

Fit uses axis-train, and axis-val selects sites; eval is excluded. Keep
high/medium-confidence labels and exclude step zero during fit. Methods are
`tact_mean_difference` and `lda_contrast`. No unannotated loop tail is extended.

## TACT CAP/GATE

```bash
.venvs/runner/bin/python scripts/tact_baseline_workflow.py --stage steering --experiment-id NEW_TACT_EXPERIMENT_ID
```

This stage consumes the preparation configuration and TACT bundle. It runs both
67-task eval arms, then harness and reporting for each. Configs/controller/logs
go to `experiments/<id>/`, and run artifacts to `runs/<id>-{cap,gated}/`.

| Setting | CAP | GATE |
|---|---|---|
| Bundle | `baseline/axes/tact` | `baseline/axes/tact` |
| GPUs / port | 0-3 / 8130 | 4-7 / 8131 |
| Steering | k=2 | k=2, lambda=0.5 |
| Rollout workers / max_num_seqs | 12 / 12 | 12 / 12 |
| Harness workers | 12 | 12 |

Both arms use EasySteer, TP=4, prefix caching, gpu_memory_utilization=0.5,
async_scheduling=false and `all_generated_think_close`. The final sampled close
token must be consumed by a later forward to apply steering.

The complete preparation-to-TACT sequence is:

```bash
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage all --experiment-id NEW_TACT_EXPERIMENT_ID
```

It runs enrichment -> hidden -> axes -> TACT steering/harness, **not baseline
rollout, annotation or LDA steering**. Separate processes release extraction CUDA
memory before serving. Preparation status is in `baseline/workflow/status.json`;
steering status is in the experiment directory. Existing run IDs are not replaced.

## LDA CAP/GATE queue

`scripts/queue_lda_after_tact.py` retains the original `SOURCE` and `TARGET`
experiment IDs in its source. For a new pair, set both to the intended new IDs
before running; the CLI has no experiment-ID arguments.

```bash
.venvs/runner/bin/python scripts/queue_lda_after_tact.py --check
.venvs/runner/bin/python scripts/queue_lda_after_tact.py
```

`--check` validates and writes configs/plan/lock files. The queue copies TACT arm
configs from `experiments/SOURCE/{cap,gated}.json`, changing only the axis to LDA,
rollout workers to 15 and max_num_seqs to 15. Harness workers remain 12. It waits
for both TACT evaluation processes to reach a terminal state and summaries to
cover 67 outcomes. It permits terminal harness failures; this is not a guarantee
that every patch was successfully graded. Ports must be released before launch.

`--resume` resumes existing LDA rollouts but does not make an already-existing
`evaluation/candidate/` overwriteable. Use the package `evaluate` command with a
new eval ID for rescoring. The queue's original experiment IDs also collide with
completed original runs if launched without changing them.

At the 2026-10-14 inventory, `experiments/` was empty. Old configs/controller and
statistics paths documented in historical reports were absent. Existing `runs/`
and `baseline/` remain; restore the intended source configs or generate a new
TACT experiment before using the queue. Absence does not establish where files
were moved or whether an external backup exists.

## Retained scripts and historical runs

See [scripts index](../scripts/README.md) and [workspace layout](WORKSPACE-LAYOUT.md).
The original formal run IDs are:

- Baseline: `baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012` (258 results).
- Baseline harness: `baseline-eval-harness-20261012` (67 tasks).
- TACT: `tact-easysteer-eval-20261013-{cap,gated}` (67 results each).
- LDA: `lda-easysteer-eval-20261013-{cap,gated}` (67 results each).

Also retain additive and paused loop-intervention runs as separate experiments.
Run status files and research completion records are not continuously refreshed
process monitors. Read actual task results for counts and inspect processes when
checking liveness. The configuration in each run is the authoritative record.
