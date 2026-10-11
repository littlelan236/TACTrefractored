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
# Registry steering

The native vLLM registry backend uses the same runner and configuration entry.
See [registry usage and implementation](docs/REGISTRY-BACKEND.md) and
[validation results](docs/REGISTRY-VALIDATION.md). Actual TACT experiments require
a fitted, model-bound AxisBundle; synthetic mechanism fixtures are for testing.


## Files kept in Git

The repository includes project source, configurations, dependency inputs and
snapshots under `locks/`, regression tests, setup/experiment scripts, and the
actual EasySteer Python runtime overlay with upstream licenses. See
[vendor sources](vendor/README.md) for pinned upstream revisions and packaging.
Run dependency installation commands from the repository root; editable entries
in the serving snapshots use relative project paths.

Model weights, wheel/image caches, generated trajectories, axes, reports,
virtual environments, and local credentials remain outside Git. The SWE task
manifest and split/exclusion metadata are versioned; the requirements-fetch
cache is generated locally. Runtime assets must be provisioned separately
before the quick checks above.

## SWE-bench Verified-70

A held-out 70-task cohort (10 functional categories, 7 tasks each) from the official Verified-500 release is available. Selection criteria, per-task difficulty and run commands are in [the dataset guide](data/swebench/verified70/README.md). It inherits the canonical runtime configuration through a dataset-only overlay.

Qwen27B official sampling profile and parameter rationale: [docs/QWEN27-RECOMMENDED-PARAMETERS.md](docs/QWEN27-RECOMMENDED-PARAMETERS.md).

## Active dataset and Qwen-27B baseline (2026-10-11)

The current partition is 127 axis-train / 64 axis-val / 67 eval. Verified70 gold failures are excluded. Previous 71 eval tasks were redistributed 47/24 with functional-type stratification at a 2:1 ratio. See [migration report](reports/repartition-20261011/README.md).

The all-task baseline run is `baseline-qwen27b-all258-20261011`, using the immutable run snapshot with Qwen3.6-27B, native vLLM, GPUs 4–7, TP=4 and port 8027. Live status: `runs/baseline-qwen27b-all258-20261011/status.json`; task trajectories: `runs/baseline-qwen27b-all258-20261011/tasks/`.
