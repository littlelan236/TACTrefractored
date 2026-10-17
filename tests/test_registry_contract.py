import pytest

from tact.backends import model_config
from tact.backends.registry_vllm import request_fields
from tact.config import resolve_config
from tact.contracts import model_descriptor


def test_registry_none_uses_same_agent_and_explicit_request_protocol():
    cfg = resolve_config(overrides=["serving.backend=registry_vllm"])
    assert cfg["registry"]["architecture"] == "TactQwen3_5MoeForConditionalGeneration"
    assert cfg["registry"]["sites"] == []
    kwargs = model_config(cfg)["model_kwargs"]
    assert kwargs["extra_body"]["vllm_xargs"] == request_fields(cfg)
    assert kwargs["extra_body"]["vllm_xargs"]["tact_strategy"] == "none"


def test_registry_cannot_steer_without_bound_axes():
    with pytest.raises(ValueError, match="AxisBundle"):
        resolve_config(overrides=["serving.backend=registry_vllm", "steering.strategy=cap"])


def test_registry_prefix_cache_selects_explicit_clean_history_policy():
    cfg = resolve_config(overrides=["serving.backend=registry_vllm",
                                   "serving.enable_prefix_caching=true"])
    assert cfg["registry"]["cache_policy"] == "clean_history_v1"
    assert cfg["registry"]["scheduler"] == "tact_vllm.adapters.scheduler.TactScheduler"
    # Cache-disabled snapshots keep the legacy registry identity.
    original = resolve_config(overrides=["serving.backend=registry_vllm"])
    assert "cache_policy" not in original["registry"]


def test_registry_rejects_external_execution():
    with pytest.raises(ValueError, match="managed"):
        resolve_config(overrides=["serving.backend=registry_vllm", "serving.management=external"])


def test_registry_cache_launch_is_isolated_from_native_and_easysteer(monkeypatch, tmp_path):
    import subprocess

    from tact.backends.native_vllm import BackendProcessManager
    from tact.backends.registry_vllm import RegistryProcessManager

    def check_output(args, **kwargs):
        return "[9]" if "AutoTokenizer" in args[2] else "/tmp/include\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    for enabled in (False, True):
        cfg = resolve_config(overrides=["serving.backend=registry_vllm",
                                       f"serving.enable_prefix_caching={str(enabled).lower()}"])
        directory = tmp_path / str(enabled)
        directory.mkdir()
        args, env = RegistryProcessManager(cfg, directory).launch_spec()
        assert ("--scheduler-cls" in args) == enabled
        assert env["VLLM_PLUGINS"] == "tact_registry"
        if enabled:
            assert args[args.index("--scheduler-cls") + 1] == cfg["registry"]["scheduler"]
            assert args[args.index("--mamba-cache-mode") + 1] == "align"
    baseline = resolve_config(overrides=["serving.enable_prefix_caching=true"])
    args, env = BackendProcessManager(baseline, tmp_path).launch_spec()
    assert "--scheduler-cls" not in args and "--mamba-cache-mode" not in args
    assert env["VLLM_PLUGINS"] == ""
    assert "registry" not in baseline


def test_nested_model_identity_paths_are_not_flattened(tmp_path):
    from tact.artifacts import write_json
    model = tmp_path / "checkpoint"
    (model / "metadata").mkdir(parents=True)
    (model / "metadata/cache.txt").write_text("cache")
    write_json(model / "config.json", {"num_hidden_layers": 2, "hidden_size": 3})
    write_json(model / "tact-identity.json", {"files": [
        {"destination": "assets/models/checkpoint/metadata/cache.txt", "bytes": 5}
    ]})
    assert model_descriptor(model)["layer_count"] == 2
