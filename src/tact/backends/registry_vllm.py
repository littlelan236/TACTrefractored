from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from tact.artifacts import hash_json, sha256, write_json
from tact.backends.native_vllm import BackendProcessManager

ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration": "TactQwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "TactQwen3_5MoeForConditionalGeneration",
}


def validate_registry(cfg):
    desc, steering, serving = cfg["model_descriptor"], cfg["steering"], cfg["serving"]
    if len(desc["architectures"]) != 1 or desc["architectures"][0] not in ARCHITECTURES:
        raise ValueError("Registry TACT supports audited Qwen3_5 dense/MoE architectures")
    if serving["enable_prefix_caching"] or not serving["language_model_only"]:
        raise ValueError("Registry TACT requires text-only serving without prefix caching")
    if serving["management"] != "managed":
        raise ValueError("Registry external services are not yet attested; use managed")
    if not serving["registry_evidence"]:
        raise ValueError("Registry validation currently requires runtime evidence")
    if steering["strategy"] != "none" and not steering["axes"]:
        raise ValueError("Registry steering requires a model-bound AxisBundle directory")
    from tact.backends.axis_bundle import validate_axis_bundle

    bundle = validate_axis_bundle(cfg)
    return {
        "schema_version": 1,
        "runtime": "vllm-0.30.0-v1",
        "numerics": "fp32-residual-v1",
        "architecture": ARCHITECTURES[desc["architectures"][0]],
        "bundle_id": hash_json({"manifest": bundle}) if bundle else "none-no-axis",
        "bundle": bundle,
        "sites": bundle["selected_sites"] if bundle else [],
        "axis_manifest_sha256": sha256(Path(steering["axes"]) / "manifest.json")
        if bundle
        else None,
    }


def request_fields(cfg):
    steering, registry = cfg["steering"], cfg["registry"]
    return {
        "tact_schema": 1,
        "tact_bundle_id": registry["bundle_id"],
        "tact_strategy": steering["strategy"],
        "tact_k": steering["k"],
        "tact_lambda": steering["lambda"],
        "tact_additive_scale": steering["additive_scale"],
        "tact_site_ids": list(range(len(registry["sites"]))),
    }


class RegistryProcessManager(BackendProcessManager):
    def launch_spec(self):
        args, env = super().launch_spec()
        cfg, registry = self.cfg, self.cfg["registry"]
        if validate_registry(cfg) != registry:
            raise ValueError("Registry artifact/config changed since resolution")
        root = Path(cfg["project_root"])
        manifest_path = self.directory / "registry-runtime.json"
        self.evidence_directory = self.directory / "evidence" / uuid.uuid4().hex
        tokenizer_probe = [
            str(root / ".venvs/serving/bin/python"),
            "-c",
            (
                "from transformers import AutoTokenizer;import sys,json;"
                "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True);"
                "print(json.dumps(t.encode('</think>',add_special_tokens=False)))"
            ),
            cfg["model"]["path"],
        ]
        close = json.loads(subprocess.check_output(tokenizer_probe, text=True))
        if len(close) != 1:
            raise ValueError("Registry TACT currently requires single-token </think>")
        manifest = {
            "schema_version": 1,
            "bundle_id": registry["bundle_id"],
            "layer_count": cfg["model_descriptor"]["layer_count"],
            "hidden_size": cfg["model_descriptor"]["hidden_size"],
            "sites": registry["sites"],
            "close_id": close[0],
            "consumer_check": cfg["serving"].get("steering_consumer_check", False),
            "consumer_check_rows": cfg["serving"].get("steering_consumer_check_rows", 32),
            "consumer_log_limit": cfg["serving"].get("steering_consumer_log_limit", 20000),
            "token_capacity": cfg["serving"]["max_num_batched_tokens"],
            "request_capacity": cfg["serving"]["max_num_seqs"],
            "axes_path": str(Path(cfg["steering"]["axes"]) / "axes.npz")
            if registry["bundle"]
            else None,
            "axes_sha256": registry["bundle"]["axes_sha256"] if registry["bundle"] else None,
            "selected_indices": registry["bundle"]["selected_indices"]
            if registry["bundle"]
            else [],
        }
        write_json(manifest_path, manifest)
        env.update(
            {
                "VLLM_PLUGINS": "tact_registry",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
                "TACT_REGISTRY_MANIFEST": str(manifest_path),
                "TACT_REGISTRY_EVIDENCE_DIR": str(self.evidence_directory),
                "VLLM_CACHE_ROOT": str(root / "assets/vllm-cache/registry" / hash_json(manifest)),
            }
        )
        args += [
            "--hf-overrides",
            json.dumps({"architectures": [registry["architecture"]]}),
            "--worker-cls",
            "tact_vllm.adapters.worker.TactWorker",
            "--no-async-scheduling",
            "--compilation-config",
            json.dumps({"pass_config": {"enable_sp": False}}),
        ]
        return args, env

    def service_identity(self, args):
        return {
            "backend": "registry_vllm",
            "args": args,
            "registry": self.cfg["registry"],
            "evidence_directory": str(self.evidence_directory),
            "lock_sha256": sha256(Path(self.cfg["project_root"]) / "locks/registry.txt"),
        }

    def verify_graph_requirement(self):
        # Replay is checked after real model requests, never inferred from readiness.
        return None

    def validate_run_evidence(self):
        files = sorted(self.evidence_directory.glob("worker-*.jsonl"))
        records = [json.loads(line) for path in files for line in path.read_text().splitlines()]
        loaded = [r for r in records if r["type"] == "loaded"]
        if len(loaded) != self.cfg["serving"]["tensor_parallel_size"]:
            raise RuntimeError("Not all TP workers attested a custom TACT backbone")
        replay = [r for r in records if r["type"] == "dispatch" and r["full_replay"]]
        replay_workers = [
            path.name
            for path in files
            if any(
                r["type"] == "dispatch" and r["full_replay"]
                for r in (json.loads(line) for line in path.read_text().splitlines())
            )
        ]
        if (
            self.cfg["serving"]["graph_requirement"] == "required"
            and len(replay_workers) != self.cfg["serving"]["tensor_parallel_size"]
        ):
            raise RuntimeError("Required FULL graph replay was not observed on every TP worker")
        consumer = [r for r in records if r["type"] == "consumer_check"]
        consumer_enabled = self.cfg["serving"].get("steering_consumer_check", False)
        if (consumer_enabled and self.cfg["steering"]["strategy"] != "none"
                and (not consumer or any(not r["all_elements_equal"] for r in consumer))):
            raise RuntimeError("Registry producer/consumer check missing or mismatched")
        write_json(
            self.directory / "verification.json",
            {
                "consumer_check_enabled": consumer_enabled,
                "consumer_records": len(consumer),
                "consumer_all_equal": bool(consumer) and all(r["all_elements_equal"] for r in consumer),
                "custom_backbone_workers": len(loaded),
                "full_replay_dispatches": len(replay),
                "full_replay_workers": replay_workers,
                "validated": True,
                "note": "Numerical mechanism validation is separate from task outcome validation",
            },
        )
