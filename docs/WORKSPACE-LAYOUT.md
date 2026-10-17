# Workspace layout and retention

Updated: 2026-10-14 after scripts cleanup.

| Path | Role | Retention |
|---|---|---|
| `src/`, `packages/`, `vendor/` | Project implementation and pinned native overlays | Keep source and licenses |
| `configs/`, `resources/`, `locks/`, `tests/` | Runtime definitions, dependencies and regression tests | Keep |
| `docs/`, `README.md`, `scripts/README.md` | Current interfaces and usage | Keep; dated evidence remains historical |
| `data/` | Benchmark inputs, splits/exclusions, model profiles | Keep authoritative data; fetch cache is generated |
| `baseline/` | Canonical 258 trajectories, 191 annotations, hidden tensors, axes and baseline harness | Keep |
| `runs/` | Immutable configs, task attempts/results, predictions, harness and runtime evidence | Keep formal runs; archive old trials with provenance |
| `experiments/` | Experiment launch/control/config/statistics products | Empty at inventory; historical paths may be absent |
| `state/task-groups/` | Shared task coordination and result reuse | Archive with its baseline runs |
| `artifacts/` | Historical labels, annotation audits, older and synthetic axes | Archive; label duplicates do not imply whole-directory duplicates |
| `imports/` | Migration inputs, manifests and prior snapshots | Keep; asset-migration manifest is used by assets-check |
| `reports/`, `logs/` | Research/validation evidence and launch logs | Preserve final reports and relevant raw evidence |
| `tmp/` | Supplementary annotation inputs, repair scripts and audit working files | Review audit contents before clearing |
| `assets/` | Model weights, images, dependency wheels and compilation cache | Keep required assets; compiler cache is regenerable |
| `.venvs/` | Runner/analysis/serving/EasySteer environments | Rebuildable with pinned inputs but operationally required |

`state/task-groups/qwen35-9b-baseline258-20261012/` has per-task lock, identity,
owner and result records. Result records point to source-run trajectories and
hashes; reuse validates and copies the source attempt. Forward/reverse coordinated
baseline runs share a cohort and are not independent repeated samples. Process
locks are released on exit; lock files and stored PIDs do not imply active jobs.

The scripts directory now contains exactly five necessary scripts and its README:
the baseline research/TACT workflow, LDA queue and three EasySteer setup/runtime
scripts. One-off statistics, smoke/performance drivers, cycle-replay research
scripts, Verified70 preparation utilities, old backup and Python cache were
removed on 2026-10-14. Package implementation, vendor tests, source data and
existing run products were not removed by this script cleanup.

Older reports may name deleted scripts as historical evidence. Such names are
not current executable instructions. The package CLI and remaining scripts are
documented in [BASELINE-WORKFLOW.md](BASELINE-WORKFLOW.md).

Baseline trajectory and annotation files are flat under their canonical roots.
Use `baseline/config/splits.json` and `baseline/manifest.json` for membership.
Do not count annotation audit JSONL copies as additional labeled tasks.

Models and compiler caches dominate disk use; deleting tiny analysis scripts
improves organization but saves little space. Original trajectories/annotations
are not reproducible by simply rerunning a stochastic rollout or API annotator.
Hidden tensors can be recomputed, but require significant GPU work. Preserve
model fingerprint, immutable config, selected attempt, trajectory hash, evaluation
ID and task-set definitions alongside any summarized table.


## Git source snapshot policy (2026-10-17)

Local retention and Git tracking are separate. Keep operational data on the server,
but do not commit model assets, virtual environments, trajectories, axes, runs,
reports, logs, cache directories or scratch repairs. Downloaded benchmark record
copies (including instances JSON, source snapshots and parquet) are also local
inputs; their existing Git history is not rewritten. A fresh clone must provision
these inputs separately, using the recorded dataset revision/selection metadata.

Git retains baseline split/exclusion inputs, benchmark selections and environment
repair recipes. Under experiments, track controller scripts, README files and
CAP/GATE input configs; omit runtime queue state, PID/lock files, comparison and
validation dumps. Preserve vendor Python overlays and dependency locks as source.
