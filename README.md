# tactRefractor

Independent SWE-bench-only TACT reproduction project. The current implementation contains the unified configuration, task container factory, rollout runner, official harness adapter, trajectory-based research pipeline, and native vLLM baseline backend.

The registry vLLM steering backend is implemented for the audited Qwen3.5 dense/MoE
runtime. The EasySteer backend now uses the pinned native steering API and graph
kernel extension. Registry use, limitations and validation are
documented in [REGISTRY-BACKEND.md](docs/REGISTRY-BACKEND.md).

EasySteer use and its explicit all-occurrences close-token semantics are documented
in [EASYSTEER-BACKEND.md](docs/EASYSTEER-BACKEND.md), with test evidence and performance
limits in [EASYSTEER-VALIDATION.md](docs/EASYSTEER-VALIDATION.md).

## Quick checks

All settings for new runs, including generation and agent templates, are explicit
in [configs/experiment.yaml](configs/experiment.yaml). See
[configuration semantics](docs/CONFIGURATION.md) for precedence, immutable run
snapshots, timeout scopes, and backend capability constraints.

```bash
.venvs/runner/bin/tact config check
.venvs/runner/bin/tact assets-check --deep
.venvs/runner/bin/tact environment-check --run-id env-check
.venvs/runner/bin/tact run --run-id baseline-smoke
```

All task Docker containers are created through `tact.environments.ContainerFactory`. Rollout and harness use the same image identity, dependency policy, wheelhouse, bootstrap script, locale, and timeout resolver. Each task attempt and environment probe is retained under `runs/<run>/tasks/<id>/attempts/<attempt>/`.

The default `.traj.json` is the authoritative rollout artifact. A separate capture model is not used. Stored responses, reasoning, tool calls, usage and provenance are consumed by `TrajectoryReader` and decision-point extraction.

Model-specific hidden-state axes must be generated for the exact checkpoint, tokenizer, template, layer count and hidden size. An axis bundle cannot be used with a different model fingerprint.

## Three-method point clouds

`tact fit --all-methods` fits TACT mean difference, LDA contrast and Fisher on
the same verified train/validation inputs and saves labeled projections. `tact plot`
exports their point clouds and comparison as PNG/PDF/SVG, CSV/NPZ, and an HTML
gallery. See [POINT-CLOUDS.md](docs/POINT-CLOUDS.md) for commands, data requirements
and the preserved historical figures. OT is red, OA green, and CAL purple.
## Registry steering

The native vLLM registry backend uses the same runner and configuration entry.
See [registry usage and implementation](docs/REGISTRY-BACKEND.md) and
[validation results](docs/REGISTRY-VALIDATION.md). Actual TACT experiments require
a fitted, model-bound AxisBundle; synthetic mechanism fixtures are for testing.


## Source and runtime assets

Project source includes configurations, dependency inputs and snapshots
under `locks/`, regression tests, the five retained pipeline/setup scripts, and the
actual EasySteer Python runtime overlay with upstream licenses. See
[vendor sources](vendor/README.md) for pinned upstream revisions and packaging.
Run dependency installation commands from the repository root; editable entries
in the serving snapshots use relative project paths.

Model weights, wheel/image caches, generated trajectories, axes, reports,
virtual environments, and local credentials remain outside Git. The SWE task
manifest and split/exclusion metadata are source inputs; the requirements-fetch
cache is generated locally. Runtime assets must be provisioned separately
before the quick checks above.

## SWE-bench Verified-70

A held-out 70-task cohort (10 functional categories, 7 tasks each) from the official Verified-500 release is available. Selection criteria, per-task difficulty and run commands are in [the dataset guide](data/swebench/verified70/README.md). It inherits the canonical runtime configuration through a dataset-only overlay.

Qwen27B official sampling profile and parameter rationale: [docs/QWEN27-RECOMMENDED-PARAMETERS.md](docs/QWEN27-RECOMMENDED-PARAMETERS.md).

## Canonical baseline and current workflow (2026-10-14)

The canonical baseline uses Qwen3.5-9B, sourced from
`baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012`. It contains 258
trajectories: 127 axis-train / 64 axis-val / 67 eval. Final annotations cover all
191 train/val tasks. Hidden extraction and TACT/LDA bundles are complete.

Use `baseline/traj/<task>.traj.json`, `baseline/annotation/<task>.jsonl`,
`baseline/config/splits.json` and `baseline/manifest.json`. The canonical
configuration is `baseline/config/resolved-config.json`; the general default
`configs/experiment.yaml` currently selects a different, Qwen3.6-35B-A3B model.

See [the complete workflow guide](docs/BASELINE-WORKFLOW.md) for baseline
rollout/harness, annotation prerequisites, enrichment, hidden extraction, axes,
TACT steering and LDA queue. [Scripts index](scripts/README.md) lists the five
retained scripts; [workspace layout](docs/WORKSPACE-LAYOUT.md) explains retention.

Original TACT/LDA CAP/GATE runs each contain 67 results under
`runs/{tact,lda}-easysteer-eval-20261013-{cap,gated}/`. Their saved configurations
and chosen evaluation summaries are authoritative. The old 27B baseline is a
historical run, not the current canonical baseline.

At inventory, `experiments/` was empty. Historical analysis/config/controller
paths may therefore be absent even though formal run data remains. Restore the
required source configs before relaunching the LDA queue. Updating these docs
does not start or resume jobs.

Split history is documented in the [repartition report](reports/repartition-20261011/README.md).
