import copy
import csv
import json

import numpy as np
import pytest

from tact.artifacts import read_json, sha256, write_json
from tact.contracts import hf_sites
from tact.research.axes import CLASSES, fit_all_axes
from tact.research.plots import _compatible, _load_projections, plot_bundle


@pytest.fixture
def fitted(tmp_path):
    """Two dynamic sites, three classes and separate problems; no GPU/model needed."""
    desc = {"model_fingerprint": "fixture", "layer_count": 2, "hidden_size": 4,
            "site_schema": "hf-hidden-v1", "sites": hf_sites(2)}
    latent, labels = tmp_path / "latents", tmp_path / "labels"
    latent.mkdir()
    (labels / "labels").mkdir(parents=True)
    exclusion = tmp_path / "exclusions.json"
    write_json(exclusion, {"instance_ids": []})
    cfg = {"project_root": str(tmp_path), "model_descriptor": desc, "input_hashes": {},
           "research": {"fit_exclusions": str(exclusion), "exclude_step_zero_from_fit": False,
                        "label_confidence": ["high", "medium"], "top_k_sites": 2,
                        "fit_method": "tact_mean_difference", "pca_components": 4,
                        "ridge": 0.001}}
    tasks, hashes, original = [], {}, {}
    rng = np.random.default_rng(12)
    for split in ("axis-train", "axis-val"):
        iid = f"{split}-task"
        y = np.repeat(CLASSES, 6)
        h = rng.normal(size=(len(y), 2, 4)).astype(np.float32)
        h[y == CLASSES[0], :, 0] += 3
        h[y == CLASSES[1], :, 1] += 3
        path = tmp_path / f"{iid}.traj.json"
        write_json(path, {"messages": [{"role": "assistant"} for _ in y]})
        digest = sha256(path)
        tasks.append({"instance_id": iid, "partition": split, "source_traj": path.name,
                      "source_sha256": digest})
        file = latent / f"{iid}.npz"
        np.savez_compressed(file, hidden=h, step=np.arange(len(y)))
        hashes[iid] = sha256(file)
        rows = [{"instance_id": iid, "step": i, "label": str(label), "confidence": "high",
                 "source_sha256": digest} for i, label in enumerate(y)]
        (labels / "labels" / f"{iid}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        original[split] = h
    write_json(latent / "manifest.json", {"model": desc, "run_id": "fixture", "tasks": tasks})
    write_json(latent / "verification.json", {"complete": True, "files": hashes})
    write_json(labels / "manifest.json", {"model": desc, "run_id": "fixture"})
    snapshot = copy.deepcopy(cfg)
    bundles = fit_all_axes(cfg, latent, labels, "three")
    assert cfg == snapshot
    return bundles, original


def test_three_method_pointcloud_exports_match_train_cal_scaling(fitted):
    pytest.importorskip("matplotlib", reason="Rendering tests use the analysis environment")
    bundles, original = fitted
    for bundle in bundles:
        manifest = read_json(bundle / "manifest.json")
        projected = _load_projections(bundle, manifest)
        with np.load(bundle / "axes.npz") as z:
            for split, h in original.items():
                expected = np.stack([
                    np.einsum("nld,ld->nl", h.astype(np.float64) - z["mu_cal"], z[axis])
                    / z[sigma] for axis, sigma in
                    (("v_ot", "sigma_ot"), ("v_oa_perp", "sigma_oa"))], axis=-1)
                np.testing.assert_allclose(projected[split]["projection"], expected)
    result = plot_bundle(bundles[0], bundles[1:])
    output = bundles[0] / "visualizations"
    assert len(result["methods"]) == 3
    for stem in ("tact-pointcloud", "lda_contrast-pointcloud", "fisher-pointcloud",
                 "auc-by-site", "validation-method-comparison"):
        for suffix in ("png", "pdf", "svg"):
            assert (output / f"{stem}.{suffix}").stat().st_size > 1000
    for name in ("tact", "lda_contrast", "fisher"):
        with (output / f"{name}-pointcloud.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 72  # 18 rows x two splits x two peak panels; no subsampling.
        assert {row["label"] for row in rows} == set(CLASSES)
    check = read_json(output / "verification.json")
    assert check["all_site_auc_recomputed"] and check["all_points_plotted"]
    assert (output / "index.html").exists()


def test_pointcloud_rejects_tampered_coordinates_even_with_updated_hash(fitted):
    bundles, _ = fitted
    bundle = bundles[0]
    manifest = read_json(bundle / "manifest.json")
    file = bundle / manifest["projection_files"]["axis-val"]["path"]
    with np.load(file) as z:
        data = {key: z[key] for key in z.files}
    data["projection"][:, :, 0] *= -1
    np.savez_compressed(file, **data)
    manifest["projection_files"]["axis-val"]["sha256"] = sha256(file)
    with pytest.raises(ValueError, match="AUC mismatch"):
        _load_projections(bundle, manifest)


def test_pointcloud_comparison_rejects_different_checkpoint(fitted):
    bundles, _ = fitted
    loaded = []
    for bundle in bundles[:2]:
        manifest = read_json(bundle / "manifest.json")
        loaded.append((bundle, manifest, _load_projections(bundle, manifest)))
    loaded[1][1]["model"]["model_fingerprint"] = "different-checkpoint"
    with pytest.raises(ValueError, match="differ in model"):
        _compatible(loaded)


def test_historical_steering_only_bundle_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="no saved labeled projections"):
        _load_projections(tmp_path, {})
