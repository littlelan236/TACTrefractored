"""EasySteer native HTTP serving and explicit all-occurrences selection."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from tact.artifacts import hash_json, sha256, write_json
from tact.backends.axis_bundle import validate_axis_bundle
from tact.backends.native_vllm import BackendProcessManager


def validate_easysteer(cfg):
    if cfg["serving"]["management"] != "managed":
        raise ValueError("EasySteer currently requires managed provenance")
    if cfg["steering"]["trigger"] != "all_generated_think_close":
        raise ValueError("EasySteer requires explicit all_generated_think_close semantics")
    if not cfg["serving"]["language_model_only"]:
        raise ValueError("EasySteer TACT supports text-only serving")
    desc = cfg["model_descriptor"]
    if desc["architectures"] not in (
        ["Qwen3_5ForConditionalGeneration"],
        ["Qwen3_5MoeForConditionalGeneration"],
    ):
        raise ValueError("EasySteer project backend supports audited Qwen3_5 dense/MoE")
    if cfg["steering"]["strategy"] != "none" and not cfg["steering"]["axes"]:
        raise ValueError("EasySteer steering requires a model-bound AxisBundle")
    bundle = validate_axis_bundle(cfg)
    return {
        "schema_version": 1,
        "runtime": "easysteer-vllm-0.29.0-v2",
        "numerics": "model-dtype-residual-add-fp32-hidden-delta-v1",
        "selection": "all_generated_think_close",
        "bundle_id": hash_json({"manifest": bundle}) if bundle else "none-no-axis",
        "bundle": bundle,
        "axis_manifest_sha256": sha256(Path(cfg["steering"]["axes"]) / "manifest.json")
        if bundle
        else None,
    }


def compile_request(cfg):
    root = Path(cfg["project_root"])
    python = root / ".venvs/easysteer/bin/python"
    code = """import json,sys
from transformers import AutoTokenizer
from easysteer.tact import spec_from_bundle
from easysteer.vectors import to_json_payload
c=json.loads(sys.stdin.read()); s=c['steering']
t=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
close=t.encode('</think>',add_special_tokens=False)
if len(close)!=1: raise ValueError('EasySteer requires single-token </think>')
strength=s['additive_scale'] if s['strategy']=='add' else s['lambda']
spec=spec_from_bundle(s['axes'],strategy=s['strategy'],k=s['k'],strength=strength,close_id=close[0])
if spec is False: wire=False
else:
 v=spec.vectors[0]
 wire={'vectors':[{'algorithm':v.algorithm,'data':to_json_payload(v.data),'scale':v.scale,
                  'normalize':False,'apply':v.apply.to_wire()}],'conflict':spec.conflict}
print(json.dumps({'steering':wire,'close_id':close[0]}))
"""
    from tact.config import child_environment

    result = subprocess.run(
        [str(python), "-c", code],
        input=json.dumps(cfg),
        text=True,
        capture_output=True,
        check=True,
        env=child_environment(cfg),
    )
    # vLLM may emit startup notices before the result.
    return json.loads(result.stdout.strip().splitlines()[-1])


def request_fields(cfg):
    return cfg["easysteer"]["request"]["steering"]


class EasySteerProcessManager(BackendProcessManager):
    def python_executable(self):
        return Path(self.cfg["project_root"]) / ".venvs/easysteer/bin/python"

    def launch_spec(self):
        args, env = super().launch_spec()
        root = Path(self.cfg["project_root"])
        python = root / ".venvs/easysteer/bin/python"
        args[0] = str(python)
        env["PATH"] = str(python.parent) + ":" + env.get("PATH", "")
        include = subprocess.check_output(
            [str(python), "-c", "import sysconfig;print(sysconfig.get_path('include'))"], text=True
        ).strip()
        env["CPATH"] = env["C_INCLUDE_PATH"] = include
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        env["VLLM_CACHE_ROOT"] = str(
            root
            / "assets/vllm-cache/easysteer"
            / ("diagnostics" if self.cfg["serving"]["easysteer_diagnostics"] else "production")
        )
        if validate_easysteer(self.cfg) != {
            k: v for k, v in self.cfg["easysteer"].items() if k != "request"
        }:
            raise ValueError("EasySteer AxisBundle changed after config resolution")
        args += [
            "--enable-steer-vector",
            "--steer-algorithms",
            "tact_add,tact_cap,tact_gate",
            "--steer-graph-mode",
            "in_graph",
            "--steer-vector-dtype",
            "float32",
            "--no-async-scheduling",
            "--compilation-config",
            json.dumps({"pass_config": {"enable_sp": False}}),
        ]
        if self.cfg["serving"]["graph_requirement"] == "eager":
            raise ValueError("EasySteer production backend requires in_graph compiled execution")
        runtime = root / "vendor/easysteer-runtime.json"
        actual = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    "import json,vllm;print(json.dumps({'version':vllm.__version__,'path':vllm.__file__}))",
                ],
                env=env,
                text=True,
            )
            .strip()
            .splitlines()[-1]
        )
        if actual["version"] != "0.29.0" or not runtime.is_file():
            raise ValueError("EasySteer locked source overlay is not installed")
        verify = root / "scripts/verify_easysteer_runtime.py"
        subprocess.run(
            [str(python), str(verify)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        self.evidence_directory = self.directory / "evidence" / uuid.uuid4().hex
        env["TACT_EASYSTEER_EVIDENCE_DIR"] = str(self.evidence_directory)
        if self.cfg["serving"]["easysteer_diagnostics"]:
            env["TACT_EASYSTEER_DIAGNOSTICS"] = "1"
        write_json(self.directory / "easysteer-config.json", self.cfg["easysteer"])
        return args, env

    def service_identity(self, args):
        root = Path(self.cfg["project_root"])
        return {
            "backend": "easysteer",
            "args": args,
            "easysteer": self.cfg["easysteer"],
            "lock_sha256": sha256(root / "locks/easysteer.txt"),
            "evidence_directory": str(self.evidence_directory),
            "source_manifest_sha256": sha256(root / "vendor/easysteer-runtime.json"),
        }

    def verify_graph_requirement(self):
        return None  # Actual replay is validated after real requests.

    def validate_run_evidence(self):
        records = [
            json.loads(line)
            for path in self.evidence_directory.glob("worker-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        pids = {
            record["pid"]
            for record in records
            if record["full_replays"] + record["piecewise_replays"] > 0
        }
        required = self.cfg["serving"]["graph_requirement"] == "required"
        count = self.cfg["serving"]["tensor_parallel_size"]
        if required and len(pids) != count:
            raise RuntimeError("EasySteer actual graph replay not observed on every TP worker")
        write_json(
            self.directory / "verification.json",
            {
                "steering_integration": "in_graph",
                "actual_replay_workers": sorted(pids),
                "actual_replay_attested": len(pids) == count,
                "selection": "all_generated_think_close",
                "request_hash": hash_json(self.cfg["easysteer"]["request"]),
                "note": "Per-site aggregate diagnostics; numerical and selection validation are separate",
            },
        )
