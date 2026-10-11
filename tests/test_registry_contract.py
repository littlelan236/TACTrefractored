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


def test_registry_rejects_unsupported_cache_and_external_execution():
    with pytest.raises(ValueError, match="prefix"):
        resolve_config(overrides=["serving.backend=registry_vllm", "serving.enable_prefix_caching=true"])
    with pytest.raises(ValueError, match="managed"):
        resolve_config(overrides=["serving.backend=registry_vllm", "serving.management=external"])


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
