import pytest

from tact.artifacts import project_path, write_json
from tact.config import Experiment, load_resolved, resolve_config, select_instances
from tact.contracts import SiteRef, hf_sites, validate_axis_manifest


def test_final_norm_is_not_last_decoder_output():
    sites = hf_sites(4)
    assert sites[-1] == {"kind": "final_norm", "block_index": None}
    assert sites[-2] == {"kind": "decoder_output", "block_index": 2}
    with pytest.raises(ValueError):
        SiteRef("final_norm", 3)


def test_same_shape_different_checkpoint_is_rejected():
    descriptor = {"model_fingerprint": "new", "layer_count": 2, "hidden_size": 3,
                  "site_schema": "hf-hidden-v1", "sites": hf_sites(2)}
    bundle = {"model": {**descriptor, "model_fingerprint": "old"}, "selected_sites": [hf_sites(2)[0]]}
    with pytest.raises(ValueError, match="model_fingerprint"):
        validate_axis_manifest(bundle, descriptor)


def test_unknown_duplicate_semantic_configuration_fails():
    with pytest.raises(ValueError):
        Experiment.model_validate({"model": {"path": "x"}, "timeouts": {"timeout": 10}})


def test_symlink_asset_cannot_escape_project(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    (root / "assets").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        project_path(root, "assets")


def test_overlap_in_data_partitions_fails(tmp_path):
    write_json(tmp_path / "instances.json", [{"instance_id": "one"}, {"instance_id": "two"}])
    write_json(tmp_path / "splits.json", {"partitions": {"axis-train": ["one"], "axis-val": ["one"], "eval": ["two"]}})
    write_json(tmp_path / "exclusions.json", {"excluded": []})
    cfg = {"benchmark": {k: str(tmp_path / (k + ".json")) for k in ["instances", "splits", "exclusions"]}}
    cfg["benchmark"]["selection"] = "all"
    with pytest.raises(ValueError, match="Overlapping"):
        select_instances(cfg)


def test_real_config_override_and_unimplemented_backend():
    cfg = resolve_config(overrides=["timeouts.tool_command_seconds=17", "generation.seed=31"])
    assert cfg["timeouts"]["tool_command_seconds"] == 17
    assert cfg["timeouts"]["evaluation_seconds"] == 300
    assert cfg["generation"]["seed"] == 31
    with pytest.raises(ValueError, match="deferred"):
        resolve_config(overrides=["steering.strategy=cap"])
    with pytest.raises(ValueError, match="deferred"):
        resolve_config(overrides=["serving.backend=easysteer"])


def test_resolved_config_tampering_is_rejected(tmp_path):
    cfg = resolve_config()
    cfg["timeouts"]["tool_command_seconds"] += 1
    path = tmp_path / "resolved.json"
    write_json(path, cfg)
    with pytest.raises(ValueError, match="integrity"):
        load_resolved(path)
