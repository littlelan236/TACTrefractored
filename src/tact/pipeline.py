from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from tact.artifacts import update_status, write_json
from tact.config import child_environment


def run_stage(cfg: dict, run: Path, stage: str, extra: list[str] = ()) -> int:
    root = Path(cfg["project_root"])
    role = "analysis" if stage in {"decision-points", "latent", "fit", "plot"} else "runner"
    command = [str(root / f".venvs/{role}/bin/python"), "-m", "tact.worker",
               stage, "--run", str(run), *extra]
    deadline = cfg["timeouts"]["pipeline_wall_seconds"]
    output = run / "logs"
    output.mkdir(exist_ok=True)
    with (output / f"{stage}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=child_environment(cfg),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / f"{stage}-process.json", {"pid": process.pid, "command": command})
        try:
            return process.wait(timeout=deadline or None)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            update_status(run / "status.json", "interrupted", failed_stage=stage)
            raise


def pipeline(cfg: dict, run: Path, *, evaluate: bool = True) -> bool:
    if run_stage(cfg, run, "rollout"):
        return False
    if evaluate and run_stage(cfg, run, "evaluate", ["--eval-id", "candidate"]):
        return False
    return run_stage(cfg, run, "report") == 0
