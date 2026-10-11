"""Backend routing, explicit native selector semantics, and frozen request payloads."""
import pytest

from tact.backends import model_config
from tact.config import resolve_config

BASE = ["serving.backend=easysteer", "steering.trigger=all_generated_think_close"]


def test_baseline_is_explicit_false_in_same_runner_protocol():
    cfg = resolve_config(overrides=BASE)
    kwargs = model_config(cfg)["model_kwargs"]
    assert kwargs["extra_body"]["steering"] is False
    assert "vllm_xargs" not in kwargs["extra_body"]


def test_easysteer_requires_explicit_selection_semantics():
    with pytest.raises(ValueError, match="all_generated"):
        resolve_config(overrides=["serving.backend=easysteer"])


def test_steering_requires_model_bound_axes():
    with pytest.raises(ValueError, match="AxisBundle"):
        resolve_config(overrides=BASE + ["steering.strategy=cap"])


@pytest.mark.parametrize("strategy,algorithm,scale", [
    ("add", "tact_add", -0.5), ("cap", "tact_cap", 1), ("gated", "tact_gate", 2),
])
def test_axes_compile_to_one_native_vector_with_generated_token_selector(strategy, algorithm, scale):
    cfg = resolve_config(overrides=BASE + ["steering.strategy=" + strategy,
        "steering.lambda=2", "steering.axes=artifacts/axes/registry-synthetic-mechanism"])
    wire = model_config(cfg)["model_kwargs"]["extra_body"]["steering"]
    assert len(wire["vectors"]) == 1
    vector = wire["vectors"][0]
    assert vector["algorithm"] == algorithm and vector["scale"] == scale
    assert vector["apply"]["generation_tokens"] == [248069]
    assert vector["apply"]["generation"] is None
    assert vector["apply"]["prompt"] is None
