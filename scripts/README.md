# Required pipeline scripts

Only scripts needed for baseline preparation and TACT/LDA steering remain here.

| Entry | Purpose |
|---|---|
| tact_baseline_workflow.py | Baseline preflight, enrichment, hidden extraction, TACT/LDA axes, TACT CAP/GATE rollout and harness |
| queue_lda_after_tact.py | Wait for TACT evaluations, then LDA CAP/GATE rollout and harness; supports --resume |
| setup_easysteer.sh | Install the pinned EasySteer environment |
| install_easysteer_overlay.py | Install the pinned native runtime overlay; called by setup |
| verify_easysteer_runtime.py | Runtime integrity check; called by setup and the EasySteer backend |

Run commands from the project root. Baseline rollout and harness are package commands, not separate scripts:

```bash
.venvs/runner/bin/python -m tact --config configs/experiment.yaml run --run-id NEW_BASELINE_ID
.venvs/runner/bin/python -m tact rollout --run-id EXISTING_BASELINE_ID --resume-run
.venvs/runner/bin/python -m tact evaluate --run-id EXISTING_RUN_ID --eval-id NEW_EVAL_ID
```

Choose the desired baseline dataset/model/sampling in the config before a new run. `run` includes rollout, harness and reporting. Existing-run commands use their immutable resolved config.

Research preparation consumes the canonical `baseline/traj`, `baseline/annotation` and `baseline/config` data. Annotation and canonical baseline assembly must already be complete; the workflow does not create them.

```bash
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage preflight
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage enrichment
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage hidden
.venvs/analysis/bin/python scripts/tact_baseline_workflow.py --stage axes
.venvs/runner/bin/python scripts/tact_baseline_workflow.py --stage steering --experiment-id NEW_TACT_EXPERIMENT_ID
```

Enrichment re-executes with the serving tokenizer and replaces existing hidden/prepared-label inputs. Do not run it merely to inspect progress. `--stage all` runs enrichment -> hidden -> axes -> TACT steering/harness; it does not launch a new baseline rollout.

The LDA queue retains the original SOURCE/TARGET experiment IDs inside the script. For a new experiment, update those IDs first. It needs the TACT arm configs in `experiments/SOURCE/` and its evaluation results in `runs/`; the previously emptied experiments directory must be restored or regenerated. Its `--check` validates/writes the plan and config files, and is not a read-only command.

Baseline preparation products remain in `baseline/`; steering and evaluation products are stored in `experiments/` and `runs/`.

## Maintained documentation

See [full baseline/TACT/LDA workflow](../docs/BASELINE-WORKFLOW.md) for canonical
paths, exact stage scope, prerequisites, original configurations and queue limits.
See [workspace layout](../docs/WORKSPACE-LAYOUT.md) for runtime state and retention.

On 2026-10-14, 30 nonessential files and the Python cache were removed from this
directory. Historical analysis, replay, speed/smoke, Verified70 provisioning and
the old queue backup are no longer runnable entries. Dataset/annotation package
code, native vendor tests, assets and formal run products remain unchanged.
