# Verified-200 NAS cohort

Prepared on 2026-10-15 from the pinned official SWE-bench Verified-500 test
snapshot (revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`). The 200 task IDs
have no overlap with any of the existing 266 instances. The existing canonical
258-task baseline, splits, model/generation settings and retained workflow
scripts are not changed.

## Selection and difficulty

Official human repair-time difficulty counts are 87 `<15 min fix`, 87
`15 min - 1 hour`, and 26 `1-4 hours`. There are no unused `>4 hours` tasks:
all three official tasks in that band were already selected previously. Only
26 unused `1-4 hours` tasks exist. Exact four-band uniformity is therefore
impossible without repeating existing tasks or inventing difficulty labels.

Functional labels extend the existing project taxonomy and are assigned by
changed source subsystem. They are not official SWE-bench categories. Selection
minimizes category count imbalance subject to available candidates and the
difficulty quotas, then balances difficulty inside categories; ties use a fixed
SHA256 seed. No model performance or gold-test outcome determines membership.

| Category | Selected | Available unused |
|---|---:|---:|
| ORM / SQL / migrations | 22 | 110 |
| Web / HTTP / routing / security | 22 | 84 |
| Symbolic mathematics | 22 | 48 |
| Scientific arrays / coordinates / units | 21 | 24 |
| ML algorithms / estimator API | 22 | 24 |
| Plotting / rendering / interaction | 22 | 29 |
| Documentation / parsing / references | 22 | 37 |
| Test framework / fixtures / logging | 11 | 11 |
| Static analysis / types / CLI configuration | 14 | 14 |
| Serialization / file formats / output | 22 | 29 |

Every new task is held-out `eval`. This cohort must not be used for axis fitting
or hyperparameter selection. `catalog.json`, `catalog.csv` and `selection.json`
record task titles, changed paths, official difficulty and selection provenance.

## Storage and traffic

The NAS root is `/VM/wangtianyi/tactRefractor/verified200-20261015/`.

* `dataset/`: the selected benchmark definitions and catalogs.
* `images/`: immutable per-task image identity and layer descriptors.
* `blobs/sha256/`: compressed OCI layers and configs, deduplicated by SHA256.
* `original-images/`, `rootless/`: original identities and any audited ownership
  normalization required by rootless Docker. Original blobs remain intact.
* `download-plan.json`, `download-status.json`, `download-events.jsonl`:
  required bytes, completion and received payload accounting.

The original 200 images contain 286.695 GB of compressed layers when counted
per image. Deduplication reduces this to **83.739 GB (77.988 GiB)** across 342
unique layers. Reusing matching compressed blobs already cached locally further
reduces external download traffic; the actual savings and received payload are
recorded in the final provisioning report. This is payload accounting, excluding
TLS/HTTP headers and unrelated traffic from other jobs. NAS writes and reads are
separate LAN traffic. Rootless derived layers add NAS storage but no Internet
download traffic.

Files are downloaded directly to the NAS, with resumption and SHA256 checks.
Public Docker Hub manifests are pinned by digest; the public `mirror.gcr.io`
cache can supply the exact same digest when Hub returns rate limiting. No
Docker daemon restart, `data-root` relocation or broad image prune occurs.

## Existing CLI and local cache

Use this independent dataset-only overlay:

```bash
.venvs/runner/bin/tact --config data/swebench/verified200-nas/benchmark.yaml config check
.venvs/runner/bin/tact --config data/swebench/verified200-nas/benchmark.yaml nas-images check
.venvs/runner/bin/tact --config data/swebench/verified200-nas/benchmark.yaml nas-images status
.venvs/runner/bin/tact --config data/swebench/verified200-nas/benchmark.yaml nas-images load --limit 8
.venvs/runner/bin/tact --config data/swebench/verified200-nas/benchmark.yaml run --run-id NEW_NAS_RUN_ID
```

`nas-images check --deep` also hashes the stored blobs; normal check verifies
manifest hashes, blob presence and sizes. The downloader already checks every
original layer's SHA256. Do not run the cohort while download/rootless preparation
is incomplete. Use the completion and validation records to determine readiness.

The overlay inherits `configs/experiment.yaml`, including its current model.
For a Qwen3.5-9B steering comparison, use the intended model/runtime settings and
that model's fitted bundle explicitly; this overlay does not silently select the
canonical baseline model or change steering parameters.

`ContainerFactory` locks NAS image IDs without loading all 200 images during
preflight. Before creating each task container, it imports that image through the
existing Docker socket. It streams an OCI archive directly from NAS blobs to
Docker 29's containerd image importer; there is no temporary local tar. Rollout,
resume, environment checks and the official harness use the same factory.

Only newly imported NAS image IDs are enrolled in the local cache journal at
`~/.local/share/tact-nas-image-cache/`. Existing images are never enrolled or
automatically deleted. The cache has a conservative 160 GiB logical image limit
and keeps 100 GiB of local filesystem headroom; image-size accounting may count
shared layers multiple times. It uses one per-daemon inter-process lock, held
through container creation. Eviction skips any image referenced by a running or
stopped container and never uses forced removal. If space cannot be made within
these rules, loading fails explicitly; reduce new-cohort concurrency or free
space deliberately. Cache loss does not change pinned image identities.

The original datasets do not contain `nas_image` references, so their image
lookup and container settings retain the original local path. Existing command
lines, resolved configs and image IDs remain valid.

## Provisioning and validation

The maintained provisioning entry is `scripts/prepare_verified200_nas.py`.
Selection uses `.venvs/analysis/bin/python` and SciPy; subsequent stages use
`.venvs/runner/bin/python`. Selection refuses to overwrite an existing cohort.
Manifest preparation produces the NAS references; download resumes completed
layers and partial files. `seed-local` copies exact existing compressed blobs
without modifying Docker storage. `normalize-rootless --wait` audits Matplotlib
repository build layers and publishes the final references after completion.

```bash
.venvs/analysis/bin/python scripts/prepare_verified200_nas.py select
.venvs/runner/bin/python scripts/prepare_verified200_nas.py manifests
.venvs/runner/bin/python scripts/prepare_verified200_nas.py seed-local
.venvs/runner/bin/python scripts/prepare_verified200_nas.py download --workers 12
.venvs/runner/bin/python scripts/prepare_verified200_nas.py normalize-rootless
```

Do not rerun `manifests` after publishing normalized identities: it is a
provisioning stage, not a read-only progress command. Progress is available in
NAS status files and `logs/nas200/`. The original file/config preservation
snapshot and source backups are in `tmp/nas200-backup-20261015/`.

Regression tests exercise local-only compatibility, immutable identities,
corrupt/incomplete blobs, cache ownership and in-use protection, OCI archive
contents, and ownership-only rootless transformations. Live import/container and
gold-smoke outcomes are recorded separately. A smoke does not establish that all
200 tasks pass gold evaluation or that model steering improves benchmark scores.

## Automatic gold evaluation after download exit

`scripts/gold_after_nas_download.py` is armed for the current downloader. It waits
on a Linux pidfd process-exit event, without polling or timer sleeps. Normal
completion, failure or signal termination triggers available rootless preparation
and the full fixed 200-task gold evaluation. Missing image data is recorded as
infrastructure failure; it is not a model result. Preparation does not wait for
missing layers. Duplicate gold run IDs are refused.

The scheduled run is `verified200-nas-gold-20261015`, with 4 harness workers.
This override affects only the new run. Controller state and logs are under
`logs/nas200/gold-after-download/`; the result summary will be
`runs/verified200-nas-gold-20261015/evaluation/gold/summary.json`.

## Gold retry with explicit Clash metadata proxy

The initial full gold evaluation stopped before testing because its metadata
HTTP requests used direct connections, which reset on this server. A live
GitHub request through the existing Clash proxy returned HTTP 200; changing
nodes was not required by that evidence.

`scripts/run_nas200_gold_proxy.py` runs an isolated 200-task metadata preflight,
then evaluates all 200 gold patches under the original immutable run config.
Only GitHub raw/jsDelivr GET requests explicitly use `http://127.0.0.1:7890`.
Docker socket requests and container traffic retain their existing behavior.
Real HTTP 404/410 responses are preserved and cached for official fallback-path
lookup. Metadata failures are not replaced with fabricated environment files.

Run ID: `verified200-nas-gold-20261015`; new eval ID: `gold-proxy-20261015`.
Logs: `logs/nas200/gold-proxy-20261015/`.
Results: `runs/verified200-nas-gold-20261015/evaluation/gold-proxy-20261015/`.
The launcher blocks on child processes; it does not poll. Original failed gold
artifacts, scripts, datasets and configuration remain unchanged.

## Task environment repairs

See [the repair record](NAS200-ENVIRONMENT-REPAIRS.md) for the 17 unsuccessful
initial gold tasks, scoped offline dependency/environment repairs, and the
provenance-aware combined evaluation report.
