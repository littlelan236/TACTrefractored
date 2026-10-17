from __future__ import annotations

import concurrent.futures
import fcntl
import os
import shutil
import time
from pathlib import Path

from tact.artifacts import hash_json, identifier, now, read_json, sha256, write_json

TERMINAL = {"Submitted", "LimitsExceeded", "ContextWindowExceededError", "RepeatedFormatError", "TimeExceeded"}


def experiment_identity(cfg):
    serving = cfg["serving"]
    return hash_json({
        "model": cfg["model_descriptor"]["model_fingerprint"], "generation": cfg["generation"],
        "agent": cfg["agent"], "templates": cfg.get("templates"), "steering": cfg["steering"],
        "environment": cfg["environment"], "timeouts": cfg["timeouts"],
        "context": cfg["model"]["max_model_len"], "dtype": cfg["model"]["dtype"],
        "backend": serving["backend"], "tp": serving["tensor_parallel_size"],
        "tool_parser": serving["tool_parser"], "reasoning_parser": serving["reasoning_parser"],
    })


class Claim:
    def __init__(self, handle):
        self.handle = handle

    def close(self):
        self.handle.close()


class TaskCoordinator:
    """Explicit shared cohort, Linux process locks and provenance-preserving copies."""

    def __init__(self, cfg, run, images):
        self.cfg, self.run, self.images = cfg, Path(run), images
        group = identifier(cfg["execution"]["coordination_group"])
        self.directory = Path(cfg["project_root"]) / "state/task-groups" / group
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "group.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = experiment_identity(cfg)
            meta = self.directory / "group.json"
            if meta.exists() and read_json(meta)["experiment_identity"] != expected:
                raise ValueError("Coordination group has incompatible model/sampling/environment/budgets")
            if not meta.exists():
                write_json(meta, {"experiment_identity": expected, "created_at": now(),
                                 "note": "Collaborative cohort; results are not independent samples."})

    def claim(self, row):
        iid = identifier(row["instance_id"])
        handle = (self.directory / (iid + ".lock")).open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        expected = {"instance_hash": hash_json(row), "image_id": self.images[iid]["image_id"]}
        identity = self.directory / (iid + ".identity.json")
        if identity.exists() and read_json(identity) != expected:
            handle.close()
            raise ValueError("Shared task dataset/image mismatch: " + iid)
        write_json(identity, expected)
        write_json(self.directory / (iid + ".owner.json"),
                   {"pid": os.getpid(), "run": self.run.name, "claimed_at": now()})
        return Claim(handle)

    def reuse(self, row):
        iid = row["instance_id"]
        local = self.run / "tasks" / iid / "result.json"
        if local.exists() and read_json(local)["exit_status"] in TERMINAL:
            self.publish(row, read_json(local))
            return read_json(local)
        record = self.directory / (iid + ".result.json")
        if not record.exists():
            return None
        shared = read_json(record)
        source = Path(shared["source_run"])
        source_task = source / "tasks" / iid
        source_result = shared["result"]
        attempt = source_task / "attempts" / identifier(source_result["attempt_id"])
        trajectory = attempt / (iid + ".traj.json")
        if sha256(trajectory) != shared["trajectory_sha256"]:
            raise ValueError("Shared completed trajectory changed: " + iid)
        name = "reused-" + hash_json([str(source), source_result["attempt_id"]])[:20]
        target = local.parent / "attempts" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(".copy-" + name)
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(attempt, temporary)
            temporary.rename(target)
        result = {**source_result, "attempt_id": name,
                  "reused_from": {"run_id": source.name, "attempt_id": source_result["attempt_id"],
                                  "trajectory_sha256": shared["trajectory_sha256"], "copied_at": now()}}
        write_json(local, result)
        print(iid + ": reused terminal result from " + source.name, flush=True)
        return result

    def publish(self, row, result):
        if result["exit_status"] not in TERMINAL:
            return
        iid = row["instance_id"]
        record = self.directory / (iid + ".result.json")
        if record.exists():
            return
        trajectory = self.run / "tasks" / iid / "attempts" / result["attempt_id"] / (iid + ".traj.json")
        write_json(record, {"source_run": str(self.run), "result": result,
                           "trajectory_sha256": sha256(trajectory), "published_at": now()})


def coordinated_batch(cfg, run, batch, factory, process_task):
    coordinator = TaskCoordinator(cfg, run, factory.images)
    pending = list(batch)
    results, running = [], {}

    def execute(row, claim):
        try:
            result = coordinator.reuse(row)
            if result is None:
                result = process_task(cfg, run, row, factory)
                coordinator.publish(row, result)
            return result
        finally:
            claim.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["execution"]["rollout_workers"]) as pool:
        while pending or running:
            for row in pending[:]:
                if len(running) >= cfg["execution"]["rollout_workers"]:
                    break
                claim = coordinator.claim(row)
                if claim is None:
                    continue
                pending.remove(row)
                try:
                    future = pool.submit(execute, row, claim)
                except BaseException:
                    claim.close()
                    raise
                running[future] = row
            if running:
                done, _ = concurrent.futures.wait(running, timeout=1,
                                                 return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    del running[future]
            elif pending:
                time.sleep(1)
    return results
