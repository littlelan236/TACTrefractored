"""Check wheel version and modified overlay files before managed serving."""

import hashlib
import importlib.metadata
import json
import sysconfig
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "vendor/easysteer-runtime.json").read_text())
if importlib.metadata.version("vllm") != manifest["vllm_version"]:
    raise RuntimeError("Wrong vLLM wheel version")
installed = Path(sysconfig.get_paths()["purelib"]) / "vllm"
if str(installed) != manifest["overlay_path"]:
    raise RuntimeError("EasySteer imported from the wrong environment")
for name, digest in manifest["source_files_sha256"].items():
    if name.startswith(("model_hooks/", "steer_vectors/")) or name in {
        "compilation/cuda_graph.py",
        "v1/worker/gpu/cudagraph_utils.py",
        "v1/worker/gpu_worker.py",
    }:
        for directory in (root / "vendor/easysteer-vllm/vllm", installed):
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Changed or missing EasySteer overlay: {path}")
print("Verified pinned EasySteer runtime and TACT extension sources")
