from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from tact.artifacts import update_status, write_json
from tact.config import child_environment


def run_stage(cfg: dict, run: Path, stage: str, extra: list[str] = (),
              *, wall_deadline: float | None = None) -> int:
    root = Path(cfg["project_root"])
    role = "analysis" if stage in {"decision-points", "latent", "fit", "plot"} else "runner"
    command = [str(root / f".venvs/{role}/bin/python"), "-m", "tact.worker",
               stage, "--run", str(run), *extra]
    wall_budget = cfg["timeouts"]["pipeline_wall_seconds"]
    if wall_deadline is None and wall_budget:
        wall_deadline = time.monotonic() + wall_budget
    remaining = None if wall_deadline is None else wall_deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        update_status(run / "status.json", "pipeline_timeout", failed_stage=stage)
        return 124
    output = run / "logs"
    output.mkdir(exist_ok=True)
    with (output / f"{stage}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=child_environment(cfg),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / f"{stage}-process.json", {"pid": process.pid, "command": command})
        try:
            return process.wait(timeout=remaining)
        except BaseException as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=cfg["timeouts"].get("server_shutdown_seconds", 30))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                update_status(run / "status.json", "pipeline_timeout", failed_stage=stage)
                return 124
            update_status(run / "status.json", "interrupted", failed_stage=stage)
            raise


def pipeline(cfg: dict, run: Path, *, evaluate: bool = True) -> bool:
    budget = cfg["timeouts"]["pipeline_wall_seconds"]
    deadline = time.monotonic() + budget if budget else None
    if run_stage(cfg, run, "rollout", wall_deadline=deadline):
        return False
    if evaluate and run_stage(cfg, run, "evaluate", ["--eval-id", "candidate"], wall_deadline=deadline):
        return False
    return run_stage(cfg, run, "report", wall_deadline=deadline) == 0
