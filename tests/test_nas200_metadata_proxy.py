import importlib.util
from pathlib import Path

import pytest
import requests


@pytest.mark.parametrize("status", [200, 404, 410])
def test_metadata_proxy_is_scoped_and_caches_real_missing_responses(tmp_path, monkeypatch, status):
    path = Path(__file__).resolve().parents[1] / "scripts/run_nas200_gold_proxy.py"
    spec = importlib.util.spec_from_file_location("nas_gold_proxy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def original(session, method, url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = status
        response._content = b"exact metadata"
        response.url = url
        return response
    monkeypatch.setattr(requests.sessions.Session, "request", original)
    module.install_metadata_proxy("http://127.0.0.1:7890", tmp_path)
    url = "https://raw.githubusercontent.com/repo/name/commit/environment.yml"
    session = requests.Session()
    assert session.get(url).status_code == status
    assert session.get(url).content == b"exact metadata"
    assert len(calls) == 1
    assert calls[0][1]["proxies"]["https"] == "http://127.0.0.1:7890"
    session.get("http+docker://localhost/images")
    assert "proxies" not in calls[-1][1]
