"""Audit the requested config and its effects at actual consumer boundaries."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tact.artifacts import hash_json, write_json
from tact.backends.native_vllm import BackendProcessManager, model_config
from tact.config import (
    Experiment,
    agent_config,
    child_environment,
    load_config_file,
    load_resolved,
    prompt_templates,
    resolve_config,
    root_path,
)

PLAN_VALUES = yaml.safe_load("""
benchmark:
  instances: data/swebench/instances.json
  splits: data/swebench/splits.json
  exclusions: data/swebench/exclusions.json
  selection: eval
model:
  path: assets/models/Qwen3.6-35B-A3B
  served_name: tact-model
  max_model_len: 262144
  dtype: bfloat16
  trust_remote_code: false
serving:
  backend: native_vllm
  management: managed
  host: 127.0.0.1
  port: 8000
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  tensor_parallel_size: 8
  max_num_seqs: 8
  graph_requirement: auto
  tool_parser: qwen3_xml
  reasoning_parser: qwen3
generation:
  temperature: 0.6
  seed: 0
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  presence_penalty: 0.0
  repetition_penalty: 1.0
  max_tokens: 81920
  enable_thinking: true
  preserve_thinking: true
agent:
  step_limit: 250
  max_consecutive_format_errors: 10
  cost_limit: 0
environment:
  docker_host: unix:///run/user/1002/docker.sock
  wheelhouse: assets/wheels
  cwd: /testbed
  locale: C.UTF-8
  constraints: ["numpy<2"]
  repo_pins:
    pydicom/pydicom: ["pytest<8", "numpy<1.24"]
    pyvista/pyvista: ["vtk<9.3"]
timeouts:
  container_lifetime_seconds: 28800
  container_start_seconds: 120
  bootstrap_seconds: 600
  tool_command_seconds: 120
  model_request_seconds: 28800
  evaluation_seconds: 300
  pipeline_wall_seconds: 0
execution:
  rollout_workers: 8
  harness_workers: 8
  infrastructure_retry_limit: 3
  retry_mode: fresh_attempt
steering:
  strategy: none
  axes: null
  k: 2.0
  lambda: 0.5
  additive_scale: -0.5
  sites: from_bundle
  trigger: first_generated_think_close
research:
  annotation_chunk_steps: 10
  fit_method: tact_mean_difference
  label_confidence: [high, medium]
  exclude_step_zero_from_fit: true
  top_k_sites: 10
""")


def test_canonical_file_explicitly_matches_every_requested_value():
    raw = load_config_file(root_path() / "configs/experiment.yaml")
    cfg = resolve_config()
    for section, expected in PLAN_VALUES.items():
        for key, value in expected.items():
            assert raw[section][key] == value, (section, key)
            resolved = cfg[section][key]
            if section == "model" and key == "path" or (
                section == "benchmark" and key != "selection"
            ) or section == "environment" and key == "wheelhouse":
                resolved = str(Path(resolved).relative_to(root_path()))
            assert resolved == value, (section, key)
    assert cfg["sources"]["schema_defaults"] == []
    assert "template" not in cfg["agent"]
    assert cfg["template_hash"] == hash_json(raw["templates"])
    assert list((root_path() / "configs").glob("*.yaml")) == [
        root_path() / "configs/experiment.yaml"
    ]


def test_partial_experiment_and_cli_changes_reach_requests_and_serving(monkeypatch, tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text("generation:\n  temperature: 0.17\nserving:\n  port: 18212\n")
    cfg = resolve_config(path, ["generation.seed=123", "serving.async_scheduling=false"])
    kwargs = model_config(cfg)["model_kwargs"]
    assert kwargs["temperature"] == 0.17
    assert kwargs["seed"] == 123
    assert kwargs["api_base"] == "http://127.0.0.1:18212/v1"
    assert kwargs["timeout"] == 28800
    assert kwargs["max_tokens"] == 81920
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "/tmp/include\n")
    args, env = BackendProcessManager(cfg, tmp_path / "serving").launch_spec()
    assert args[args.index("--port") + 1] == "18212"
    assert args[args.index("--tensor-parallel-size") + 1] == "8"
    assert "--no-async-scheduling" in args
    generation = json.loads(args[args.index("--override-generation-config") + 1])
    assert generation["temperature"] == 0.17 and generation["seed"] == 123
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_templates_are_frozen_in_run_snapshot(monkeypatch, tmp_path):
    cfg = resolve_config(overrides=[
        "templates.system_template=CONFIGURED_SYSTEM",
        "templates.observation_template=CONFIGURED_OBSERVATION",
        "agent.step_limit=37",
    ])
    path = tmp_path / "resolved-config.json"
    write_json(path, cfg)
    frozen = load_resolved(path)
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=AssertionError("live template read")))
    assert agent_config(frozen)["system_template"] == "CONFIGURED_SYSTEM"
    assert agent_config(frozen)["step_limit"] == 37
    assert model_config(frozen)["observation_template"] == "CONFIGURED_OBSERVATION"


def test_old_run_templates_remain_available_after_yaml_consolidation():
    from tact.artifacts import sha256

    resource = root_path() / "resources/legacy-agent.yaml"
    cfg = {"project_root": str(root_path()),
           "agent": {"template": str(root_path() / "configs/agent.yaml")},
           "template_hash": sha256(resource)}
    assert prompt_templates(cfg) == resolve_config()["templates"]


def test_environment_does_not_override_selected_config(monkeypatch):
    monkeypatch.setenv("DPB_TEMPERATURE", "0")
    monkeypatch.setenv("DOCKER_HOST", "unix:///wrong.sock")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    cfg = resolve_config()
    assert cfg["generation"]["temperature"] == 0.6
    env = child_environment(cfg)
    assert env["DOCKER_HOST"] == "unix:///run/user/1002/docker.sock"
    assert "DPB_TEMPERATURE" not in env and "VLLM_USE_V2_MODEL_RUNNER" not in env


def test_duplicate_yaml_setting_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("generation:\n  temperature: 0.6\n  temperature: 0\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config_file(path)


def test_container_lifetime_and_request_timeout_have_independent_clocks():
    cfg = resolve_config()
    assert cfg["timeouts"]["container_lifetime_seconds"] == 28800
    assert cfg["timeouts"]["model_request_seconds"] == 28800
    with pytest.raises(ValueError, match="Container-side"):
        resolve_config(overrides=["timeouts.container_lifetime_seconds=599"])


def test_pipeline_wall_budget_is_shared_across_stages(monkeypatch, tmp_path):
    from tact import pipeline

    cfg = {"timeouts": {"pipeline_wall_seconds": 10}}
    stages = []
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: 100)
    def stage(cfg, run, name, extra=(), *, wall_deadline=None):
        stages.append((name, wall_deadline))
        return 0
    monkeypatch.setattr(pipeline, "run_stage", stage)
    assert pipeline.pipeline(cfg, tmp_path)
    assert stages == [("rollout", 110), ("evaluate", 110), ("report", 110)]


def test_expired_pipeline_budget_stops_before_starting_next_stage(monkeypatch, tmp_path):
    from tact import pipeline

    cfg = {"project_root": str(tmp_path), "timeouts": {"pipeline_wall_seconds": 10}}
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: 111)
    monkeypatch.setattr(pipeline.subprocess, "Popen", Mock(side_effect=AssertionError("started")))
    assert pipeline.run_stage(cfg, tmp_path, "report", wall_deadline=110) == 124


def test_synchronous_steering_constraint_is_checked_before_startup():
    with pytest.raises(ValueError, match="synchronous"):
        resolve_config(overrides=["serving.backend=registry_vllm", "serving.async_scheduling=true"])


def test_schema_defaults_match_explicit_operational_values():
    raw = load_config_file(root_path() / "configs/experiment.yaml")
    cfg = Experiment.model_validate(raw).model_dump(mode="json", by_alias=True)
    assert cfg == raw


def test_shared_container_spec_uses_configured_environment_and_budgets(monkeypatch):
    from tact.environments.containers import ContainerFactory
    from tact.environments.policy import EnvironmentPolicy

    cfg = resolve_config(overrides=["environment.locale=C", "timeouts.container_lifetime_seconds=29000"])
    factory = object.__new__(ContainerFactory)
    factory.cfg = cfg
    factory.policy = EnvironmentPolicy(cfg)
    factory.images = {"task": {"image_id": "sha256:fixture"}}
    monkeypatch.setattr(factory, "lock_images", lambda instances: factory.images)
    spec = factory.spec({"instance_id": "task", "repo": "pydicom/pydicom"})
    assert spec["command"] == ["sleep", "29000"]
    assert spec["cwd"] == "/testbed"
    assert spec["environment"]["LANG"] == "C"
    assert cfg["environment"]["wheelhouse"] in spec["volumes"]
    assert factory.policy.specs("pydicom/pydicom") == ["numpy<2", "pytest<8", "numpy<1.24"]
