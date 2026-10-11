# Unified experiment configuration

`configs/experiment.yaml` is the single editable configuration for new runs.
It explicitly contains all values requested on 2026-10-11, plus the existing
implementation settings and the exact agent prompts/response templates.

```bash
.venvs/runner/bin/tact config show
.venvs/runner/bin/tact config check
.venvs/runner/bin/tact run --run-id new-baseline
```

The precedence is explicit `--set dotted.key=value` > a supplied `--config`
overlay > `configs/experiment.yaml`. A partial overlay inherits this same
canonical file, rather than unrelated Python defaults. Duplicate YAML keys and
unknown fields fail validation. `config check/show` report the base file,
overrides, environment overrides (none), and any schema-defaulted fields. The
canonical configuration has no schema-defaulted fields.

Generation parameters, GPU IDs/TP, task selection, dependency pins, concurrency,
timeouts, research parameters and prompt templates all come from this file.
Requests and serving arguments are generated from the same resolved snapshot.
The snapshot contains templates, so editing the live YAML cannot change an
existing run's prompt halfway through evaluation or analysis.

The default values include temperature 0.6, seed 0, port 8000, GPUs 0–7, TP=8,
eight rollout/harness workers, max_tokens=81920, and container lifetime 28800.
Container lifetime and model-request timeout have independent clocks. Each
container-side operation must fit within the declared lifetime; a host-side
request timeout is not added to bootstrap to silently increase that lifetime.
Consequently an exceptionally long request can outlive its container: the
runner records this as an infrastructure failure and applies the configured
fresh-attempt retry policy. The configured values are not auto-enlarged.

`pipeline_wall_seconds=0` disables the overall limit. A positive value is one
shared deadline across rollout, evaluation and reporting, not a fresh budget
per stage. Standalone stages use the same value for that invocation.

Additional implementation settings are explicit in the same YAML:

- `serving.async_scheduling: null` keeps the native backend default; false
  explicitly disables async scheduling. Registry/EasySteer require synchronous
  scheduling and reject true.
- GPU graph and steering constraints remain backend capabilities. Native
  serving only accepts NONE. Registry uses first-generated-close semantics;
  EasySteer currently requires an explicit `all_generated_think_close` setting
  when selected. The resolver never changes the requested trigger silently.
- `templates` contains system/instance prompts and observation/format-error
  templates. Operational agent budgets remain in `agent`.
- `environment.requirements_cache`, metadata-request and server-shutdown
  budgets are configurable instead of being private runner constants.

Existing runs continue to use their own immutable `resolved-config.json`.
`resources/legacy-agent.yaml` is only for snapshots created before templates
were inlined; its hash must match the historical snapshot. It is not read by
new runs and must not be edited as a live configuration.

The dated speed20 generation script now inherits the canonical file for new
preparations. Its explicit cohort/GPU/backend overrides remain experiment
choices. Existing result files are not rewritten. Standalone synthetic/backend
smoke and fixed-load benchmark scripts intentionally have their own CLI fixture
inputs; they are not production rollout/harness entry points.
