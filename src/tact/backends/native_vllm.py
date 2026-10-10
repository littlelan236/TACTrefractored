from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests
import yaml

from tact.artifacts import now, read_json, sha256, write_json
from tact.config import child_environment


def capabilities() -> dict:
    return {"backend": "native_vllm", "strategies": ["none"],
            "steering": False, "server": "vllm.entrypoints.openai.api_server"}


def model_config(cfg: dict) -> dict:
    g, s = cfg["generation"], cfg["serving"]
    templates = yaml.safe_load(Path(cfg["agent"]["template"]).read_text()).get("model", {})
    return {**templates, "model_name": "openai/" + cfg["model"]["served_name"],
            "cost_tracking": "ignore_errors", "model_kwargs": {
                "api_base": f"http://{s['host']}:{s['port']}/v1",
                "api_key": os.environ.get(s["api_key_env"], "EMPTY"),
                "timeout": cfg["timeouts"]["model_request_seconds"],
                **{k: g[k] for k in ["temperature", "seed", "top_p", "max_tokens", "presence_penalty"]},
                "extra_body": {**{k: g[k] for k in ["top_k", "min_p", "repetition_penalty"]},
                               "chat_template_kwargs": {k: g[k] for k in ["enable_thinking", "preserve_thinking"]}}
            }}


class BackendProcessManager:
    def __init__(self, cfg: dict, directory: Path):
        self.cfg, self.directory = cfg, directory
        self.process = None
        self.log = None
        self.session = requests.Session()
        self.session.trust_env = False

    def launch_spec(self):
        cfg = self.cfg
        root = Path(cfg["project_root"])
        s, m = cfg["serving"], cfg["model"]
        python = root / ".venvs/serving/bin/python"
        args = [str(python), "-m", "vllm.entrypoints.openai.api_server",
                "--model", m["path"], "--served-model-name", m["served_name"],
                "--host", s["host"], "--port", str(s["port"]),
                "--tensor-parallel-size", str(s["tensor_parallel_size"]),
                "--max-model-len", str(m["max_model_len"]), "--dtype", m["dtype"],
                "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
                "--max-num-seqs", str(s["max_num_seqs"]),
                "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
                "--seed", str(cfg["generation"]["seed"]),
                "--enable-auto-tool-choice", "--tool-call-parser", s["tool_parser"],
                "--reasoning-parser", s["reasoning_parser"],
                "--generation-config", "vllm", "--override-generation-config",
                json.dumps({k: v for k, v in cfg["generation"].items()
                            if k not in ["enable_thinking", "preserve_thinking"]})]
        args += ["--enable-prefix-caching" if s["enable_prefix_caching"] else "--no-enable-prefix-caching"]
        if s["language_model_only"]:
            args += ["--language-model-only"]
        if m["trust_remote_code"]:
            args += ["--trust-remote-code"]
        if s["graph_requirement"] == "eager":
            args += ["--enforce-eager"]
        env = child_environment(cfg)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, s["gpus"]))
        env["PATH"] = str(python.parent) + ":" + env.get("PATH", "")
        include = subprocess.check_output([str(python), "-c",
                                           "import sysconfig;print(sysconfig.get_path('include'))"], text=True).strip()
        env["CPATH"] = include
        env["C_INCLUDE_PATH"] = include
        env["VLLM_CACHE_ROOT"] = str(root / "assets/vllm-cache")
        return args, env

    def health(self) -> bool:
        s = self.cfg["serving"]
        url = f"http://{s['host']}:{s['port']}"
        headers = {"Authorization": "Bearer " + os.environ.get(s["api_key_env"], "EMPTY")}
        try:
            self.session.get(url + "/health", timeout=3, headers=headers).raise_for_status()
            models = self.session.get(url + "/v1/models", timeout=3, headers=headers)
            models.raise_for_status()
            return self.cfg["model"]["served_name"] in {m["id"] for m in models.json()["data"]}
        except (requests.RequestException, ValueError, KeyError):
            return False

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        s = self.cfg["serving"]
        if s["management"] == "external":
            if not s["external_manifest"]:
                raise ValueError("External server requires a provenance manifest")
            manifest = read_json(Path(s["external_manifest"]))
            if manifest.get("model_fingerprint") != self.cfg["model_descriptor"]["model_fingerprint"]:
                raise ValueError("External server model identity mismatch")
            if manifest.get("backend") != "native_vllm" or not self.health():
                raise ValueError("External baseline service not verified")
            write_json(self.directory / "service.json", manifest)
            return
        with socket.socket() as sock:
            if sock.connect_ex((s["host"], s["port"])) == 0:
                raise RuntimeError("Managed service port is occupied; refusing to attach or stop it")
        args, env = self.launch_spec()
        self.log = (self.directory / "vllm.log").open("w")
        self.process = subprocess.Popen(args, cwd=self.cfg["project_root"], env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(self.directory / "service.json", {
            "backend": "native_vllm", "pid": self.process.pid, "started_at": now(),
            "model_fingerprint": self.cfg["model_descriptor"]["model_fingerprint"],
            "args": args, "lock_sha256": sha256(Path(self.cfg["project_root"]) / "locks/serving.txt"),
            "gpus": s["gpus"], "capabilities": capabilities(),
        })
        deadline = time.monotonic() + s["startup_seconds"]
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"vLLM exited ({self.process.returncode}); see {self.directory}/vllm.log")
                if self.health():
                    self.verify_graph_requirement()
                    return
                time.sleep(2)
            raise TimeoutError("vLLM readiness deadline exceeded")
        except BaseException:
            self.close()
            raise

    def verify_graph_requirement(self):
        if self.cfg["serving"]["graph_requirement"] == "required":
            # Readiness is insufficient proof of graph replay. No false positive.
            raise ValueError("Baseline graph_requirement=required has no replay validator yet; use auto")

    def close(self):
        if self.process is not None:
            pid = self.process.pid
            try:
                os.killpg(pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                self.process.wait(timeout=10)
            finally:
                self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
