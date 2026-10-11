"""Consumer verification is opt-in and must not masquerade as native support."""

import pytest

from tact.config import resolve_config


def test_consumer_check_is_explicit_and_disabled_by_default():
    cfg = resolve_config()
    assert cfg["serving"]["steering_consumer_check"] is False
    assert cfg["sources"]["schema_defaults"] == []


def test_registry_consumer_capacity_is_configurable_and_bounded():
    cfg = resolve_config(overrides=[
        "serving.backend=registry_vllm", "serving.steering_consumer_check=true",
        "serving.steering_consumer_check_rows=8"])
    assert cfg["serving"]["steering_consumer_check_rows"] == 8
    with pytest.raises(ValueError):
        resolve_config(overrides=["serving.steering_consumer_check_rows=0"])
