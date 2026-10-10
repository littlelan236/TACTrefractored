from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tact.artifacts import hash_json, identifier, project_path, read_json, sha256
from tact.contracts import model_descriptor


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Benchmark(Strict):
    instances: str = "data/swebench/instances.json"
    splits: str = "data/swebench/splits.json"
    exclusions: str = "data/swebench/exclusions.json"
    selection: str | list[str] = "eval"


class Model(Strict):
    path: str
    served_name: str = "tact-model"
    max_model_len: int = Field(default=262144, gt=0)
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    trust_remote_code: bool = False


class Serving(Strict):
    backend: Literal["native_vllm", "registry_vllm", "easysteer"] = "native_vllm"
    management: Literal["managed", "external"] = "managed"
    host: str = "127.0.0.1"
    port: int = Field(default=18030, ge=1024, le=65535)
    gpus: list[int] = [4, 5, 6, 7]
    tensor_parallel_size: int = Field(default=4, gt=0)
    max_num_seqs: int = Field(default=8, gt=0)
    max_num_batched_tokens: int = Field(default=8192, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, lt=1)
    graph_requirement: Literal["auto", "required", "eager"] = "auto"
    tool_parser: str = "qwen3_xml"
    reasoning_parser: str = "qwen3"
    enable_prefix_caching: bool = False
    language_model_only: bool = True
    startup_seconds: int = Field(default=1200, gt=0)
    api_key_env: str = "TACT_API_KEY"
    external_manifest: str | None = None


class Generation(Strict):
    temperature: float = Field(default=0.6, ge=0)
    seed: int = 0
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=-1)
    min_p: float = Field(default=0.0, ge=0, le=1)
    presence_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0)
    max_tokens: int = Field(default=81920, gt=0)
    enable_thinking: bool = True
    preserve_thinking: bool = True


class Agent(Strict):
    step_limit: int = Field(default=250, gt=0)
    max_consecutive_format_errors: int = Field(default=10, gt=0)
    cost_limit: float = Field(default=0, ge=0)
    wall_time_limit_seconds: int = Field(default=0, ge=0)
    template: str = "configs/agent.yaml"


class Environment(Strict):
    docker_host: str = "unix:///run/user/1002/docker.sock"
    docker_executable: str = "/home/wangtianyi/.local/bin/docker"
    wheelhouse: str = "assets/wheels"
    cwd: str = "/testbed"
    user: str = "root"
    platform: str = "linux/amd64"
    locale: str = "C.UTF-8"
    constraints: list[str] = ["numpy<2"]
    repo_pins: dict[str, list[str]] = {
        "pydicom/pydicom": ["pytest<8", "numpy<1.24"], "pyvista/pyvista": ["vtk<9.3"]
    }
    network_fallback: bool = False
    variables: dict[str, str] = {}


class Timeouts(Strict):
    container_lifetime_seconds: int = Field(default=32400, gt=0)
    container_start_seconds: int = Field(default=120, gt=0)
    bootstrap_seconds: int = Field(default=600, gt=0)
    tool_command_seconds: int = Field(default=120, gt=0)
    model_request_seconds: int = Field(default=28800, gt=0)
    evaluation_seconds: int = Field(default=300, gt=0)
    pipeline_wall_seconds: int = Field(default=0, ge=0)


class Execution(Strict):
    rollout_workers: int = Field(default=8, gt=0)
    harness_workers: int = Field(default=8, gt=0)
    infrastructure_retry_limit: int = Field(default=3, ge=0)
    retry_mode: Literal["fresh_attempt"] = "fresh_attempt"


class Steering(Strict):
    strategy: Literal["none", "add", "cap", "gated"] = "none"
    axes: str | None = None
    k: float = Field(default=2, ge=0)
    lambda_: float = Field(default=0.5, alias="lambda", ge=0, le=1)
    additive_scale: float = -0.5
    sites: Literal["from_bundle"] = "from_bundle"
    trigger: Literal["first_generated_think_close"] = "first_generated_think_close"


class Research(Strict):
    annotation_chunk_steps: int = Field(default=10, gt=0)
    fit_method: Literal["tact_mean_difference", "fisher", "lda_contrast"] = "tact_mean_difference"
    label_confidence: list[Literal["high", "medium", "low"]] = ["high", "medium"]
    exclude_step_zero_from_fit: bool = True
    top_k_sites: int = Field(default=10, gt=0)
    fit_exclusions: str = "data/swebench/fit-exclusions.json"
    extraction_dtype: Literal["float32", "bfloat16"] = "float32"
    extraction_chunk_size: int = Field(default=128, gt=0)
    pca_components: int = Field(default=100, ge=2)
    ridge: float = Field(default=0.001, gt=0)


class Experiment(Strict):
    schema_version: Literal[1] = 1
    benchmark: Benchmark = Benchmark()
    model: Model
    serving: Serving = Serving()
    generation: Generation = Generation()
    agent: Agent = Agent()
    environment: Environment = Environment()
    timeouts: Timeouts = Timeouts()
    execution: Execution = Execution()
    steering: Steering = Steering()
    research: Research = Research()


def root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def select_instances(cfg: dict) -> tuple[list[dict], dict]:
    b = cfg["benchmark"]
    rows = read_json(Path(b["instances"]))
    partitions = read_json(Path(b["splits"]))["partitions"]
    excluded = set(read_json(Path(b["exclusions"]))["excluded"])
    known = {r["instance_id"]: r for r in rows}
    if len(known) != len(rows):
        raise ValueError("Duplicate benchmark instances")
    seen = set(excluded)
    if set(partitions) != {"axis-train", "axis-val", "eval"}:
        raise ValueError("Expected axis-train/axis-val/eval partitions")
    for name, ids in partitions.items():
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Overlapping/duplicate task partition: {name}")
        seen.update(ids)
    if seen != set(known):
        raise ValueError("Splits/exclusions must account for every benchmark instance")
    selection = b["selection"]
    if isinstance(selection, str):
        if selection == "all":
            ids = [iid for key in ["axis-train", "axis-val", "eval"] for iid in partitions[key]]
        elif selection in partitions:
            ids = partitions[selection]
        else:
            raise ValueError(f"Unknown task partition: {selection}")
    else:
        ids = selection
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must be non-empty and unique")
    for iid in ids:
        identifier(iid)
        if iid not in known or iid in excluded:
            raise ValueError(f"Unknown/excluded task: {iid}")
    return [known[i] for i in ids], partitions


def resolve_config(path: Path | None = None, overrides: list[str] = ()) -> dict:
    root = root_path()
    path = Path(path or root / "configs/experiment.yaml").resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = {"file": str(path), "overrides": list(overrides), "environment_overrides": []}
    for entry in overrides:
        key, separator, text = entry.partition("=")
        if not separator:
            raise ValueError("Overrides must use dotted.key=value")
        value = yaml.safe_load(text)
        parts = key.split(".")
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"Invalid override path: {key}")
        cursor[parts[-1]] = value
    cfg = Experiment.model_validate(raw).model_dump(mode="json", by_alias=True)
    if cfg["serving"]["backend"] != "native_vllm" or cfg["steering"]["strategy"] != "none":
        raise ValueError("Only native_vllm + strategy=none is implemented; steering is deferred")
    s, t = cfg["serving"], cfg["timeouts"]
    if not s["gpus"] or len(set(s["gpus"])) != len(s["gpus"]) or min(s["gpus"]) < 0:
        raise ValueError("GPU IDs must be non-empty, unique and non-negative")
    if s["tensor_parallel_size"] != len(s["gpus"]):
        raise ValueError("Baseline TP must equal the configured visible GPU count")
    if cfg["generation"]["max_tokens"] >= cfg["model"]["max_model_len"]:
        raise ValueError("max_tokens must leave room for the prompt in max_model_len")
    if t["bootstrap_seconds"] + max(t["tool_command_seconds"], t["evaluation_seconds"],
                                     t["model_request_seconds"]) > t["container_lifetime_seconds"]:
        # Requests are sequentially awaited while the task's container remains alive.
        raise ValueError("Container lifetime must exceed bootstrap plus each operation budget")
    for section, keys in {"benchmark": ["instances", "splits", "exclusions"],
                          "model": ["path"], "agent": ["template"],
                          "environment": ["wheelhouse"], "research": ["fit_exclusions"]}.items():
        for key in keys:
            cfg[section][key] = str(project_path(root, cfg[section][key]))
    cfg["model_descriptor"] = model_descriptor(Path(cfg["model"]["path"]))
    cfg["project_root"] = str(root)
    cfg["sources"] = sources
    selected, _ = select_instances(cfg)
    cfg["selected_ids"] = [row["instance_id"] for row in selected]
    cfg["input_hashes"] = {key: sha256(Path(cfg["benchmark"][key]))
                           for key in ["instances", "splits", "exclusions"]}
    cfg["template_hash"] = sha256(Path(cfg["agent"]["template"]))
    cfg["config_hash"] = hash_json(cfg)
    return cfg


def load_resolved(path: Path) -> dict:
    cfg = read_json(path)
    if cfg.get("project_root") != str(root_path()):
        raise ValueError("Resolved config belongs to another project")
    expected = cfg["config_hash"]
    payload = {k: v for k, v in cfg.items() if k != "config_hash"}
    if hash_json(payload) != expected:
        raise ValueError("Resolved config integrity check failed")
    return cfg


def child_environment(cfg: dict) -> dict[str, str]:
    prefixes = ("TACT_", "DPB_", "SWETEST_", "VLLM_", "LITELLM_", "MSWEA_", "MINI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in {
        "PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HTTP_PROXY", "HTTPS_PROXY",
        "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "CPATH", "C_INCLUDE_PATH"
    }}
    root = Path(cfg["project_root"])
    env.update({"DOCKER_HOST": cfg["environment"]["docker_host"],
                "HF_HOME": str(root / "assets/hf-cache"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True", "NO_PROXY": "127.0.0.1,localhost",
                "MSWEA_GLOBAL_CONFIG_DIR": str(root / "assets/user-config/mini-swe-agent")})
    key_env = cfg["serving"]["api_key_env"]
    if key_env in os.environ:
        env[key_env] = os.environ[key_env]
    return env
