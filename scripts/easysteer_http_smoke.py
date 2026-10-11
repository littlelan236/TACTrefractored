"""Native completions HTTP contract test; no proxy or custom server."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from easysteer.vectors import to_json_payload
from vllm.steer_vectors import TactBandPayload

root = Path(__file__).resolve().parents[1]
directory = root / "logs/easysteer-http"
directory.mkdir(parents=True, exist_ok=True)
log = (directory / "server.log").open("w")
env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "VLLM_PLUGINS": "",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "TACT_EASYSTEER_DIAGNOSTICS": "1",
    "TACT_EASYSTEER_EVIDENCE_DIR": str(directory / "evidence"),
    "VLLM_CACHE_ROOT": str(root / "assets/vllm-cache/easysteer-http"),
}
env["PATH"] = str(root / ".venvs/easysteer/bin") + ":" + env.get("PATH", "")
args = [
    str(root / ".venvs/easysteer/bin/vllm"),
    "serve",
    str(root / "assets/test-models/easysteer-tiny-qwen2"),
    "--served-model-name",
    "tiny-tact",
    "--port",
    "18149",
    "--max-model-len",
    "128",
    "--max-num-seqs",
    "4",
    "--max-num-batched-tokens",
    "128",
    "--gpu-memory-utilization",
    "0.25",
    "--enable-steer-vector",
    "--steer-algorithms",
    "tact_add,tact_cap,tact_gate",
    "--steer-graph-mode",
    "in_graph",
    "--steer-vector-dtype",
    "float32",
    "--no-async-scheduling",
    "--compilation-config",
    json.dumps({"cudagraph_capture_sizes": [1, 2, 4]}),
    "--language-model-only",
]
process = subprocess.Popen(
    args, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
)
session = requests.Session()
session.trust_env = False
responses = []
try:
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("Native server exited; inspect server.log")
        try:
            if session.get("http://127.0.0.1:18149/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise TimeoutError("Native server readiness deadline")
    import numpy as np

    ot = np.eye(128, dtype=np.float32)[0]
    oa = np.eye(128, dtype=np.float32)[1]
    payload = TactBandPayload(
        {
            2: {
                "v_ot": ot,
                "v_oa": oa,
                "mu": np.full(128, 4.0, np.float32),
                "tau_ot": np.float32(0),
                "tau_oa": np.float32(0),
            }
        },
        {
            "schema_version": 1,
            "layer_count": 2,
            "hidden_size": 128,
            "sites": {"2": {"kind": "final_norm", "block_index": None}},
        },
    )
    for algorithm in [None, "tact_cap", "tact_gate"]:
        steering = (
            False
            if algorithm is None
            else {
                "vectors": [
                    {
                        "algorithm": algorithm,
                        "data": to_json_payload(payload),
                        "scale": 1 if algorithm == "tact_cap" else 0.5,
                        "normalize": False,
                        "apply": {"generation_tokens": [2]},
                    }
                ]
            }
        )
        body = {
            "model": "tiny-tact",
            "prompt": [1, 2, 1],
            "max_tokens": 4,
            "temperature": 0,
            "ignore_eos": True,
            "allowed_token_ids": [2],
            "steering": steering,
        }
        response = session.post("http://127.0.0.1:18149/v1/completions", json=body, timeout=30)
        response.raise_for_status()
        result = response.json()
        assert result["usage"]["completion_tokens"] == 4
        responses.append({"algorithm": algorithm or "baseline", "response": result})
    invalid = {
        **body,
        "steering": {
            "vectors": [
                {
                    "algorithm": "tact_cap",
                    "data": to_json_payload(payload),
                    "scale": 0.5,
                    "apply": {"generation_tokens": [2]},
                }
            ]
        },
    }
    response = session.post("http://127.0.0.1:18149/v1/completions", json=invalid, timeout=30)
    assert response.status_code == 400
    output = root / "reports/easysteer/http-smoke.json"
    output.write_text(
        json.dumps({"responses": responses, "invalid_cap_status": response.status_code}, indent=2)
        + "\n"
    )
    print("Native HTTP payload/baseline/admission smoke passed")
finally:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    log.close()
