import numpy as np
import pytest

from tact.artifacts import read_json, sha256, write_json
from tact.contracts import hf_sites
from tact.research.axes import CLASSES
from tact.research.progressive import fit_and_plot, nested_selections


def tasks_fixture():
    return [{"instance_id": f"{split}-{i}", "partition": split,
             "counts": {label: 3 for label in CLASSES},
             "steps": list(range(1, 10)), "labels": list(np.repeat(CLASSES, 3)),
             "labels_sha256": f"label-{split}-{i}"}
            for split in ("axis-train", "axis-val") for i in range(5)]


def test_five_subsets_are_nested_reproducible_and_end_in_full_population():
    tasks = tasks_fixture()
    stages = nested_selections(tasks, 42)
    assert stages == nested_selections(list(reversed(tasks)), 42)
    for split in ("axis-train", "axis-val"):
        previous = set()
        for stage in stages:
            selected = set(stage["selected_ids"][split])
            assert previous < selected
            assert stage["counts"][split]["steps"] == len(selected) * 9
            previous = selected
        assert previous == {t["instance_id"] for t in tasks if t["partition"] == split}


def test_selection_guarantees_per_problem_contrasts_not_only_global_class_counts():
    tasks = tasks_fixture()
    for task in tasks:
        task["counts"] = {"CALIBRATED": 3}
    with pytest.raises(ValueError, match="No per-problem"):
        nested_selections(tasks, 42)


def test_five_stages_refit_tact_lda_and_plot_only_selected_rows(tmp_path):
    pytest.importorskip("matplotlib")
    tasks = tasks_fixture()
    desc = {"model_fingerprint": "synthetic", "layer_count": 1, "hidden_size": 4,
            "site_schema": "hf-hidden-v1", "sites": hf_sites(1)}
    manifest = {"model": desc, "tasks": tasks, "stages": nested_selections(tasks, 42),
                "research": {"pca_components": 3, "ridge": 0.001, "top_k_sites": 1}}
    write_json(tmp_path / "manifest.json", manifest)
    latents = tmp_path / "latents"
    latents.mkdir()
    rng = np.random.default_rng(42)
    files = {}
    for task in tasks:
        h = rng.normal(size=(9, 1, 4)).astype(np.float32)
        h[:3, :, 0] += 4
        h[3:6, :, 1] += 4
        file = latents / f"{task['instance_id']}.npz"
        np.savez_compressed(file, hidden=h, step=np.array(task["steps"]))
        files[task["instance_id"]] = sha256(file)
    write_json(latents / "verification.json", {"complete": True, "files": files, "model": desc})
    fit_and_plot(tmp_path)
    assert read_json(tmp_path / "status.json")["complete"]
    assert (tmp_path / "index.html").exists()
    axes = []
    for stage in manifest["stages"]:
        selected = {iid for ids in stage["selected_ids"].values() for iid in ids}
        for method in ("tact_mean_difference", "lda_contrast"):
            bundle = tmp_path / f"stage-{stage['stage']:02d}" / method
            saved = read_json(bundle / "manifest.json")
            assert {r["instance_id"] for r in saved["input_audit"]} == selected
            for split in ("axis-train", "axis-val"):
                with np.load(bundle / saved["projection_files"][split]["path"]) as z:
                    assert set(z["instance_id"]) == set(stage["selected_ids"][split])
                    assert len(z["label"]) == stage["counts"][split]["steps"]
            with np.load(bundle / "axes.npz") as z:
                if method == "tact_mean_difference":
                    axes.append(z["v_ot"])
    assert not np.allclose(axes[0], axes[-1])  # Actual refit, not subsampling old projections.
