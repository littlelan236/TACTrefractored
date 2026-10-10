from __future__ import annotations

from pathlib import Path

from tact.artifacts import hash_json, write_json
from tact.config import select_instances
from tact.environments.containers import ContainerFactory, TaskContext


def check_environments(cfg: dict, run: Path) -> dict:
    instances, _ = select_instances(cfg)
    factory = ContainerFactory(cfg)
    reports = {}
    try:
        write_json(run / "images.json", factory.lock_images(instances))
        for row in instances:
            iid = row["instance_id"]
            records = []
            for role in ["baseline", "steering", "harness"]:
                context = TaskContext(run.name, iid, "environment-check", role,
                                      run / "environment-check" / iid / role)
                container = factory.create(row, context)
                try:
                    container.start()
                    probe = factory.probe(container, "verified-environment.json")
                    timeout = factory.execute(container, "sleep 5", 1)
                    if timeout["returncode"] not in [124, 137]:
                        raise AssertionError("Container command timeout did not fire")
                    healthy = factory.execute(container, "python -c 'print(1)'", 10)
                    if healthy["returncode"] != 0:
                        raise AssertionError("Tool timeout corrupted task container")
                    records.append({"role": role, "spec_hash": hash_json(container.spec),
                                    "policy_hash": probe["policy_hash"],
                                    "packages_hash": probe["packages_hash"], "timeout_verified": True})
                finally:
                    factory.close(container)
            for key in ["spec_hash", "policy_hash", "packages_hash"]:
                if len({r[key] for r in records}) != 1:
                    raise AssertionError(f"Environment differs between roles for {iid}: {key}")
            reports[iid] = records
        value = {"complete": True, "task_count": len(reports), "tasks": reports,
                 "steering_note": "Environment role only; no steering serving backend is implemented"}
        write_json(run / "environment-check/summary.json", value)
        return value
    finally:
        factory.cleanup_owned(run.name)
