# Configuration unification audit — 2026-10-11

Remote project: `/home/wangtianyi/tactRefractor`.

## Changes and findings

- The user's requested 61 fields were checked individually against both the
  canonical YAML and its resolved configuration. All values match exactly.
- All schema fields, including implementation extensions, are explicit in
  `configs/experiment.yaml`; its `sources.schema_defaults` list is empty.
- The default selection resolves to 71 held-out eval instances. The default
  serving endpoint is port 8000 with GPUs 0–7 and TP=8; temperature is 0.6,
  seed is 0, and max_tokens remains 81920 as requested.
- System/instance prompts and response formatting were moved into `templates`
  in the canonical YAML. New run snapshots contain those strings and their
  content hash; rollout/model configuration consumers do not read another YAML.
- The former `configs/agent.yaml` is preserved byte-for-byte as
  `resources/legacy-agent.yaml` solely for old immutable snapshots. All 17
  existing old snapshots successfully load their original template by hash.
- Partial config overlays inherit the canonical YAML; explicit CLI overrides
  have highest precedence. Duplicate YAML settings fail validation. Existing
  runs continue to use their own frozen snapshots.
- The previous bootstrap-plus-request-timeout validation incorrectly rejected
  the requested 28800-second lifetime. Container-side budgets are now checked
  individually; model request and container lifecycle clocks remain independent.
- `pipeline_wall_seconds` now governs one overall pipeline deadline, including
  all stages. Zero disables it. Deadline expiry returns 124 and is recorded as
  `pipeline_timeout`.
- Harness requirements-cache location, metadata-fetch timeout, server shutdown
  budgets and native async scheduling are explicit implementation extensions
  in the same file. Rollout environment serialization now reflects configured
  Docker executable and startup/pull budget.
- The dated speed20 preparation script now obtains generation/base settings
  from the canonical file rather than `DualBench/configs/runtime.json`. Existing
  experiment configs, trajectories and results were not rewritten.

## Verification

- Full project suite: 36 tests passed (before the final shared-container
  consumer regression was added).
- Additional shared-container config regression: verified separately.
- Ruff over project source and tests: passed.
- `git diff --check`: passed.
- Default `tact config check`: valid, 71 tasks, no hidden schema defaults.
- Behavioral checks cover actual request kwargs and server CLI compilation,
  CLI/overlay precedence, frozen prompt consumers, environment-override
  isolation, duplicate-key rejection, timeout validation, shared pipeline
  deadlines, and container policy/spec consumption.

Machine-readable evidence and resolved configuration are stored locally on the
server in `reports/config-unification-20261011/`; generated reports remain
ignored by Git.

## Scope and remaining backend constraints

No GPU model server, Docker workload or full SWE evaluation was launched for
this change. This audit verifies configuration values and consumer behavior;
it does not certify GPU performance or resolve-rate outcomes.

Registry's first-close and EasySteer's all-generated-close requirements remain
explicit backend capabilities. Selecting an incompatible trigger fails rather
than changing the requested first-close setting automatically. Numerical
steering implementations were not changed as part of configuration unification.

Standalone synthetic/backend smoke and fixed-load benchmark scripts retain
their deliberate fixture CLI inputs. They do not define defaults for production
rollout, harness or research commands. Historical snapshots deliberately retain
their original settings.
