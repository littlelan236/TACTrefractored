"""Start Verified-200 gold evaluation on downloader exit, using Linux pidfd events.

Normal completion, nonzero exit and signal termination all wake the same event.
There are no timed sleeps, status polling, or process-list polling in this script.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STORE = Path("/VM/wangtianyi/tactRefractor/verified200-20261015")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def wait_pid_exit(pid, expected_script="prepare_verified200_nas.py"):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return "already_exited"
    try:
        if expected_script:
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                return "already_exited"
            # A zombie has an empty cmdline and an already-readable pidfd.
            if any(command) and not (
                any(Path(os.fsdecode(arg)).name == expected_script for arg in command if arg)
                and b"download" in command
            ):
                raise ValueError("PID does not identify the intended NAS downloader")
        select.select([fd], [], [])
        return "process_exit_event"
    finally:
        os.close(fd)


def run_logged(command, path):
    with path.open("ab") as log:
        return subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def publish_references():
    """Keep finished derivations usable even if another preparation step failed."""
    path = ROOT / "data/swebench/verified200-nas/instances.json"
    rows = json.loads(path.read_text())
    if len(rows) != 200:
        raise ValueError("Expected the fixed 200-task cohort")
    unavailable = []
    for row in rows:
        p = STORE / "images" / (row["instance_id"] + ".json")
        record = json.loads(p.read_text())
        row["image_name"] = record["requested"]
        row["nas_image"]["manifest_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        if any(not (STORE / "blobs/sha256" / item["digest"][7:]).is_file()
               or (STORE / "blobs/sha256" / item["digest"][7:]).stat().st_size != item["size"]
               for item in [record["config"], *record["layers"]]):
            unavailable.append(row["instance_id"])
    write(path, rows)
    write(STORE / "dataset/instances.json", rows)
    return unavailable


def run_pipeline(pid, run_id, workers):
    output = ROOT / "logs/nas200/gold-after-download"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    with (output / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "runs" / run_id).exists():
            raise FileExistsError(f"Run already exists: {run_id}; automatic duplicate evaluation refused")
        status = {"stage": "waiting_for_download_exit", "download_pid": pid, "run_id": run_id,
                  "harness_workers": workers, "trigger": "Linux pidfd exit event; no polling"}
        def update(stage, **fields):
            status.update(stage=stage, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **fields)
            write(status_path, status)
        update("waiting_for_download_exit")
        event = wait_pid_exit(pid)
        download = STORE / "download-status.json"
        try:
            download_state = json.loads(download.read_text())
        except (OSError, ValueError):
            download_state = {"complete": False, "status_unavailable": True}
        update("preparing_gold", exit_event=event, download_complete=bool(download_state.get("complete")),
               download_state=download_state)
        python = str(ROOT / ".venvs/runner/bin/python")
        try:
            prep_code = run_logged([python, "scripts/prepare_verified200_nas.py", "normalize-rootless", "--skip-missing"],
                                   output / "preparation.log")
        except Exception as exc:
            prep_code = -1
            update("preparing_gold", preparation_exception=str(exc))
        try:
            missing = publish_references()
            publication_error = None
        except Exception as exc:
            missing = None
            publication_error = str(exc)
        # Run the full fixed cohort even on download/preparation failure. Missing
        # image data is infrastructure failure, never a model-resolution result.
        command = [python, "-m", "tact", "--config", "data/swebench/verified200-nas/benchmark.yaml",
                   "--set", f"execution.harness_workers={workers}", "gold", "--run-id", run_id]
        update("gold_running", preparation_exit_code=prep_code, unavailable_image_tasks=missing,
               publication_error=publication_error, gold_command=command)
        code = run_logged(command, output / "gold.log")
        summary = ROOT / "runs" / run_id / "evaluation/gold/summary.json"
        update("gold_finished" if code == 0 else "gold_failed", gold_exit_code=code,
               summary_path=str(summary), summary_exists=summary.exists())
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    parser.add_argument("--run-id", default="verified200-nas-gold-20261015")
    parser.add_argument("--harness-workers", type=int, default=4)
    args = parser.parse_args()
    if args.download_pid <= 0 or args.harness_workers <= 0 or "/" in args.run_id or "\\" in args.run_id or args.run_id in {"", ".", ".."}:
        parser.error("Invalid PID, workers, or run ID")
    raise SystemExit(run_pipeline(args.download_pid, args.run_id, args.harness_workers))


if __name__ == "__main__":
    main()
