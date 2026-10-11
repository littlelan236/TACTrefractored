"""The cache wrapper must preserve official alternate requirements-path lookup."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tact.evaluation.runner import install_harness_adapter


@pytest.mark.parametrize("missing_status", [404, 410])
def test_missing_candidate_continues_official_requirements_search(monkeypatch, tmp_path, missing_status):
    from swebench.harness import run_evaluation as official
    from swebench.harness.test_spec import python as sb_python
    from swebench.harness.test_spec.test_spec import TestSpec

    repo, commit = "test/metadata", "frozen-commit"
    prefix = "https://raw.githubusercontent.com/" + repo + "/" + commit + "/"
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.url = url
        response.status_code = missing_status if url.endswith("missing.txt") else 200
        response._content = b"not found" if response.status_code != 200 else b"numpy==1.20\n"
        return response

    # Register original attributes for restoration: the adapter is process-scoped.
    for attr in ("build_container", "cleanup_container", "RUN_EVALUATION_LOG_DIR", "exec_run_with_timeout"):
        monkeypatch.setattr(official, attr, getattr(official, attr))
    monkeypatch.setattr(TestSpec, "eval_script", TestSpec.eval_script)
    monkeypatch.setattr(sb_python.requests, "get", fetch)
    monkeypatch.setitem(sb_python.MAP_REPO_TO_REQS_PATHS, repo, ["missing.txt", "requirements.txt"])
    cfg = {"environment": {"requirements_cache": str(tmp_path / "cache")},
           "timeouts": {"metadata_request_seconds": 1}, "project_root": str(tmp_path)}
    factory = SimpleNamespace(policy=SimpleNamespace(wrap_eval=lambda script: script))
    install_harness_adapter(cfg, factory, [], tmp_path / "preflight")
    sb_python.get_requirements_by_commit.cache_clear()
    try:
        assert sb_python.get_requirements_by_commit(repo, commit).strip() == "numpy==1.20"
        assert calls == [prefix + "missing.txt", prefix + "requirements.txt"]
        assert len(list(Path(cfg["environment"]["requirements_cache"]).glob("*.txt"))) == 1
    finally:
        sb_python.get_requirements_by_commit.cache_clear()
