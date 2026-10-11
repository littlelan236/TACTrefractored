"""Exercise the unified project launcher and native chat HTTP requests."""

import json
from pathlib import Path

import requests
from tact.artifacts import write_json
from tact.backends.easysteer import EasySteerProcessManager
from tact.config import resolve_config

cfg = resolve_config(
    overrides=[
        "serving.backend=easysteer",
        "steering.trigger=all_generated_think_close",
        "steering.strategy=cap",
        "steering.k=0",
        "steering.axes=artifacts/axes/registry-synthetic-27b",
        "model.path=assets/models/Qwen3.6-27B",
        "model.max_model_len=1024",
        "generation.max_tokens=32",
        "serving.max_num_batched_tokens=256",
        "serving.max_num_seqs=2",
        "serving.gpus=[6,7]",
        "serving.tensor_parallel_size=2",
        "serving.gpu_memory_utilization=0.65",
        "serving.port=18150",
        "serving.graph_requirement=required",
        "serving.easysteer_diagnostics=true",
    ]
)
root = Path(cfg["project_root"])
directory = root / "reports/easysteer/managed-chat"
session = requests.Session()
session.trust_env = False
results = []
with EasySteerProcessManager(cfg, directory) as manager:
    for method in ["none", "cap", "gated"]:
        wire = (
            False
            if method == "none"
            else json.loads(json.dumps(cfg["easysteer"]["request"]["steering"]))
        )
        if method == "gated":
            wire["vectors"][0]["algorithm"] = "tact_gate"
            wire["vectors"][0]["scale"] = 0.5
        body = {
            "model": cfg["model"]["served_name"],
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0,
            "max_tokens": 4,
            "ignore_eos": True,
            "allowed_token_ids": [cfg["easysteer"]["request"]["close_id"]],
            "steering": wire,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        }
        response = session.post("http://127.0.0.1:18150/v1/chat/completions", json=body, timeout=60)
        response.raise_for_status()
        data = response.json()
        assert data["usage"]["completion_tokens"] == 4
        results.append({"method": method, "response": data})
    manager.validate_run_evidence()
write_json(directory / "http-results.json", results)
print("Unified EasySteer managed launcher/native chat/required replay smoke passed")
