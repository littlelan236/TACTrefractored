"""The diagnostics flag must be opt-in and travel with the resolved config."""

import pytest

from tact.config import resolve_config


def test_detailed_logging_is_disabled_in_default_unified_configuration():
    cfg = resolve_config()
    assert cfg["serving"]["easysteer_detail_log"] is False
    assert cfg["serving"]["easysteer_detail_log_limit"] == 20000
    assert cfg["sources"]["schema_defaults"] == []


def test_log_limit_must_be_positive_and_cli_can_enable_diagnostics():
    cfg = resolve_config(overrides=[
        "serving.easysteer_detail_log=true", "serving.easysteer_detail_log_limit=50"])
    assert cfg["serving"]["easysteer_detail_log"] is True
    assert cfg["serving"]["easysteer_detail_log_limit"] == 50
    with pytest.raises(ValueError):
        resolve_config(overrides=["serving.easysteer_detail_log_limit=0"])
