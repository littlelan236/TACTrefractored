from __future__ import annotations

import concurrent.futures
import importlib.metadata
import logging
import traceback
from pathlib import Path

from tact.artifacts import identifier, now, read_json, update_status, write_json
from tact.backends import model_config, process_manager
from tact.config import agent_config, select_instances
from tact.environments.containers import ContainerFactory, TaskContext

TERMINAL = {"Submitted", "LimitsExceeded", "ContextWindowExceededError", "RepeatedFormatError", "TimeExceeded"}
INFRASTRUCTURE = {"ContainerCorrupted", "EnvironmentSetupFailed", "APIError", "Timeout", "ReadTimeout",
                  "ConnectError", "APIConnectionError", "DockerException", "NotFound"}


def prepare_run(cfg: dict, run_id: str) -> Path:
    run = Path(cfg["project_root"]) / "runs" / identifier(run_id)
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}; use rollout --resume-run or a new ID")
    run.mkdir(parents=True)
    write_json(run / "resolved-config.json", cfg)
    write_json(run / "selection.json", {"instance_ids": cfg["selected_ids"]})
    write_json(run / "provenance.json", {"created_at": now(), "project_version": "0.1.0",
        "model": cfg["model_descriptor"], "config_hash": cfg["config_hash"],
        "packages": {p: importlib.metadata.version(p) for p in
                     ["mini-swe-agent", "swebench", "litellm", "docker"]}})
    update_status(run / "status.json", "prepared")
    return run


def atomic_agent_class():
    # Only persistence changes; query/action/control-flow remain upstream.
    from minisweagent.agents.default import DefaultAgent

    class DurableAgent(DefaultAgent):
        def save(self, path, *extra_dicts):
            data = self.serialize(*extra_dicts)
            if path:
                write_json(Path(path), data)
            return data

    return DurableAgent


def process_task(cfg: dict, run: Path, instance: dict, factory: ContainerFactory) -> dict:
    from minisweagent.models.litellm_model import LitellmModel

    from tact.environments.rollout_adapter import RolloutEnvironment

    iid = identifier(instance["instance_id"])
    task = run / "tasks" / iid
    result_path = task / "result.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["exit_status"] in TERMINAL:
            return previous
    attempts = task / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts.iterdir())
    if existing:
        for p in existing:
            if not (p / "attempt.json").exists():
                write_json(p / "attempt.json", {"exit_status": "Interrupted", "attempt_id": p.name})
    offset = len(existing)
    remaining = max(0, cfg["execution"]["infrastructure_retry_limit"] + 1 - offset)
    if not remaining:
        if result_path.exists():
            return read_json(result_path)
        raise RuntimeError(f"No retry budget remains for interrupted task {iid}")
    agent_settings = agent_config(cfg)
    for index in range(offset, offset + remaining):
        attempt_id = f"attempt-{index:03d}"
        output = attempts / attempt_id
        output.mkdir()
        traj = output / f"{iid}.traj.json"
        context = TaskContext(run.name, iid, attempt_id, "rollout", output)
        started, agent, env = now(), None, None
        status, submission, error = "pending", "", None
        try:
            env = RolloutEnvironment(factory, instance, context)
            agent = atomic_agent_class()(LitellmModel(**model_config(cfg)), env,
                                          **agent_settings, output_path=traj)
            info = agent.run(instance["problem_statement"])
            status, submission = info.get("exit_status", "Unknown"), info.get("submission", "")
        except Exception as exc:
            logging.getLogger("tact.rollout").exception("Task attempt failed: %s/%s", iid, attempt_id)
            status = type(exc).__name__
            error = {"exception": str(exc), "traceback": traceback.format_exc()}
            (output / "runner.log").write_text(error["traceback"])
        finally:
            if agent is not None:
                agent.save(traj, {"instance_id": iid, "info": {"exit_status": status,
                           "submission": submission, "attempt_id": attempt_id,
                           "tact": {"run_id": run.name, "config_hash": cfg["config_hash"]},
                           **(error or {})}})
            if env is not None:
                env.close()
            result = {"instance_id": iid, "attempt_id": attempt_id, "exit_status": status,
                      "submission": submission, "started_at": started, "finished_at": now(),
                      "error": error, "infrastructure_failure": status in INFRASTRUCTURE}
            write_json(output / "attempt.json", result)
            write_json(result_path, result)
        print(f"{iid}: {attempt_id} {status}", flush=True)
        if status not in INFRASTRUCTURE:
            return result
    return result


def generate_predictions(run: Path, cfg: dict) -> list[dict]:
    rows = []
    for iid in cfg["selected_ids"]:
        p = run / "tasks" / iid / "result.json"
        if not p.is_file():
            continue
        result = read_json(p)
        rows.append({"instance_id": iid, "model_name_or_path": cfg["model"]["served_name"],
                     "model_patch": result.get("submission", "")})
    write_json(run / "predictions.json", rows)
    return rows


def run_rollout(cfg: dict, run: Path) -> dict:
    rows, _ = select_instances(cfg)
    factory = ContainerFactory(cfg, read_json(run / "images.json") if (run / "images.json").exists() else None)
    update_status(run / "status.json", "preflight")
    try:
        write_json(run / "images.json", factory.lock_images(rows))
        with process_manager(cfg, run / "serving") as backend:
            update_status(run / "status.json", "rollout")
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["execution"]["rollout_workers"]) as pool:
                futures = [pool.submit(process_task, cfg, run, row, factory) for row in rows]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            backend.validate_run_evidence()
        generate_predictions(run, cfg)
        failures = [r["instance_id"] for r in results if r["exit_status"] not in TERMINAL]
        update_status(run / "status.json", "rollout_failed" if failures else "rollout_complete",
                       task_count=len(results), failed_tasks=failures)
        return {"complete": not failures, "task_count": len(results), "failed_tasks": failures}
    except BaseException as exc:
        update_status(run / "status.json", "rollout_failed", error=str(exc))
        raise
    finally:
        factory.cleanup_owned(run.name)
