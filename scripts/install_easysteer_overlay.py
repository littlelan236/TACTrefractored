"""Install project-owned fork Python sources over the matching native wheel."""

import hashlib
import importlib.metadata
import json
import shutil
import sysconfig
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if importlib.metadata.version("vllm") != "0.29.0":
    raise RuntimeError("Overlay requires the pinned vLLM 0.29.0 wheel")
site = Path(sysconfig.get_paths()["purelib"]) / "vllm"
if not site.resolve().is_relative_to(root / ".venvs/easysteer"):
    raise RuntimeError("Refusing to modify a vLLM outside the EasySteer environment")
source = root / "vendor/easysteer-vllm/vllm"
shutil.copytree(
    source, site, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)
files = {
    p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source.rglob("*.py")
}
(root / "vendor/easysteer-runtime.json").write_text(
    json.dumps(
        {
            "vllm_version": "0.29.0",
            "fork_commit": "7837b07d21c0ddffd043a30001f2c5654290aec7",
            "easysteer_commit": "baecd9698f32aadef68b41536f3db477b513a0d7",
            "wheel_basis": "98dff2a81d747d1dba01a47f939f48c3526d4206",
            "source_files_sha256": files,
            "overlay_path": str(site),
        },
        indent=2,
    )
    + "\n"
)
print("Installed locked fork overlay:", site)
