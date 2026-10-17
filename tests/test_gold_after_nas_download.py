import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def controller():
    path = Path(__file__).resolve().parents[1] / "scripts/gold_after_nas_download.py"
    spec = importlib.util.spec_from_file_location("gold_after_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("exit_code", [0, 7])
def test_pidfd_wakes_on_success_and_abnormal_exit(exit_code):
    module = controller()
    child = subprocess.Popen([sys.executable, "-c", f"import sys;sys.stdin.read(1);sys.exit({exit_code})"], stdin=subprocess.PIPE)
    try:
        child.stdin.write(b"x"); child.stdin.close()
        assert module.wait_pid_exit(child.pid, None) in {"process_exit_event", "already_exited"}
        assert child.wait() == exit_code
    finally:
        if child.returncode is None:
            child.kill(); child.wait()


@pytest.mark.parametrize("download_complete,prep_code", [(True, 0), (False, 1)])
def test_full_gold_starts_after_download_even_when_preparation_fails(tmp_path, monkeypatch, download_complete, prep_code):
    module = controller()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    store = tmp_path / "nas"
    store.mkdir()
    monkeypatch.setattr(module, "STORE", store)
    (store / "download-status.json").write_text(json.dumps({"complete": download_complete}))
    events = []
    monkeypatch.setattr(module, "wait_pid_exit", lambda pid: events.append("exit") or "process_exit_event")
    monkeypatch.setattr(module, "publish_references", lambda: [] if download_complete else ["unavailable-task"])
    def run(command, path):
        events.append(command)
        return prep_code if "normalize-rootless" in command else 0
    monkeypatch.setattr(module, "run_logged", run)
    assert module.run_pipeline(123, "gold-test", 4) == 0
    assert events[0] == "exit"
    assert "normalize-rootless" in events[1]
    assert "gold" in events[2] and "execution.harness_workers=4" in events[2]
    assert not any("benchmark.selection=" in argument for argument in events[2])
    status = json.loads((tmp_path / "logs/nas200/gold-after-download/status.json").read_text())
    assert status["download_complete"] == download_complete
    assert status["preparation_exit_code"] == prep_code
    assert status["stage"] == "gold_finished"


def test_duplicate_gold_run_is_refused(tmp_path, monkeypatch):
    module = controller()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "runs/already-exists").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.run_pipeline(123, "already-exists", 4)
