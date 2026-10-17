# NAS Verified200 environment repairs — 2026-10-15

The initial complete attempt processed 200 tasks: 183 resolved, 4 unresolved,
12 environment failures and 1 timeout. Original artifacts remain under
`runs/verified200-nas-gold-20261015/evaluation/gold-proxy-20261015/`.

Repairs are declared on individual dataset rows using `environment_overrides`.
`ContainerFactory.policy_for()` validates these against the existing environment
schema and uses them for container creation, bootstrap, commands, probes and the
official evaluation script. Rollout and harness share this factory and managed
container. Rows without overrides retain the previous environment policy.

| Tasks | Cause | Repair |
| --- | --- | --- |
| 11 Django tasks, 16454–17087 | Python 3.11 image has NumPy 2; original offline wheelhouse has no compatible NumPy <2 | Dedicated `assets/wheels-nas200/`, including NumPy 1.26.4 cp311 |
| Django 7530 | Python 3.5 `json.loads` rejects bytes from `pip list` | Decode subprocess output as UTF-8 before JSON parsing |
| Astropy 7606 | pytest 3.3.1 substitutes `unit0` for the callable-generated empty parameter ID; official ID is `[]` | pytest 4.6.11, pytest-remotedata 0.3.2 and offline dependencies; execute the same tests with their intended IDs |
| Requests 1921 | Inherited Clash proxy changes HTTP/TLS behavior of httpbin tests | Clear upper/lower-case HTTP/HTTPS/ALL proxy variables; pin pytest 4.6.11 for legacy raises(code_string) calls |
| sklearn 14710 | 300-second timeout during histogram gradient boosting tests | Limit OpenMP/BLAS threads to 1; task-specific 900-second evaluation budget |
| Sphinx 9711 | Test passes but compact, colored tox/pytest output has no parseable node ID | Request verbose, uncolored pytest output and pass its options through tox |
| SymPy 23262 | SciPy missing for two integration/lambdify regression tests | Install SciPy 1.13.1 from the dedicated offline wheelhouse |

The dedicated wheelhouse initially links the original wheels without modifying
their contents, and adds the required wheels. SHA256 identities and per-task
overrides are recorded in
`data/swebench/verified200-nas/environment-repairs.json`, mirrored to the NAS
dataset directory. Official task fields, splits, patches, test lists and OCI
image identities are unchanged. Dependency repairs happen during container
bootstrap, so they also apply to future rollout containers.

Re-evaluate the initial 17 unsuccessful tasks in a fresh evaluation directory:

```bash
.venvs/runner/bin/python -u scripts/repair_nas200_gold.py \
  --run-id verified200-nas-repair-20261015 --eval-id gold-repair-v1
```

The script blocks on evaluation completion. It never polls. Its combined
200-task summary records the source evaluation for each task and combines the
183 original successes with the repair attempt. It does not rewrite the
original evaluation's results. Logs and combined results are under
`logs/nas200/verified200-nas-repair-20261015/gold-repair-v1/`.

Regression validation after the shared code changes: **93 passed, 3 skipped**.
Live repair results are recorded separately; a successful unit suite does not
establish that all gold tasks pass.

## Follow-up validation

The first repair attempt resolved 16/17 tasks, bringing the aggregate to 199/200.
Astropy required an additional pytest-remotedata 0.3.2 pin because the original
plugin called a marker API removed in pytest 4.6. Its 98-test parameter-ID smoke
then passed without source changes. Requests also receives pytest 4.6.11 to
support its legacy `pytest.raises(exception, code_string)` usage; the previously
failing extra test passed in isolation.

The final two-task evaluation uses `verified200-nas-repair-v2-20261015`, eval
`gold-repair-v2`. It builds on the previous combined summary using:

```bash
.venvs/runner/bin/python -u scripts/repair_nas200_gold.py \
  --run-id verified200-nas-repair-v2-20261015 --eval-id gold-repair-v2 \
  --previous-combined logs/nas200/verified200-nas-repair-20261015/gold-repair-v1/combined-summary.json \
  --also psf__requests-1921
```

Live role checks for Django 7530 and SymPy 23262 passed in all three roles:
baseline, steering and harness. Each task had identical container-spec,
environment-policy and installed-package hashes across roles. See
`reports/verified200-nas/environment-repair-20261015/rolecheck/summary.json`.

Preservation checks confirm all 470 original images/tags remain and the original
configuration, benchmark data/splits/exclusions and baseline scripts retain their
pre-provisioning SHA256 hashes.

## Final result

All 200 selected gold tasks are resolved and graded. This is an aggregate with
per-task provenance: 183 original successes, 15 successes from repair-v1 and
2 successes from repair-v2. The 17 originally unsuccessful tasks have all passed
with their declared environment repairs. Astropy's final output has 242 passed;
Requests has 119 passed. No gold/test patch, assertion, official test list or
image identity was changed. The original wheelhouse was also rechecked against
the initial evaluation's full wheel hash inventory and is identical.

Final aggregate: `reports/verified200-nas/environment-repair-20261015/summary.json`.
Each task points to its own original or repair evaluation summary; original
failed reports are retained. This validates gold patch execution, not model
rollout accuracy or steering effectiveness.
