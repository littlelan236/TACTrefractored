from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tact.artifacts import identifier, read_json, sha256, write_json
from tact.contracts import validate_axis_manifest

CLASSES = ("OVERTHINKING", "OVERACTING", "CALIBRATED")


def orthogonal_pair(ot, oa):
    import numpy as np
    ot, oa = np.asarray(ot, dtype=np.float64), np.asarray(oa, dtype=np.float64)
    norm = np.linalg.norm(ot)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Degenerate OT contrast")
    ot = ot / norm
    oa = oa - (oa @ ot) * ot
    norm = np.linalg.norm(oa)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Degenerate OA contrast after orthogonalization")
    return ot, oa / norm


def per_problem_difference(problems, label: str, site: int):
    import numpy as np
    contrasts = []
    for data in problems.values():
        y, h = data["labels"], data["hidden"][:, site].astype(np.float64)
        if (y == label).any() and (y == "CALIBRATED").any():
            contrasts.append(h[y == label].mean(0) - h[y == "CALIBRATED"].mean(0))
    if not contrasts:
        raise ValueError(f"No per-problem {label} vs CAL contrast")
    return np.mean(contrasts, axis=0)


def auc(positive, negative) -> float:
    import numpy as np
    from scipy.stats import rankdata
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both classes")
    n, m = len(positive), len(negative)
    ranks = rankdata(np.concatenate([positive, negative]), method="average")
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def macro_auc(problems, direction, center, label, site):
    import numpy as np
    scores = {}
    for iid, data in problems.items():
        y = data["labels"]
        pos, neg = y == label, y == "CALIBRATED"
        if pos.any() and neg.any():
            p = (data["hidden"][:, site].astype(np.float64) - center) @ direction
            scores[iid] = auc(p[pos], p[neg])
    return {"mean_auc": float(np.mean(list(scores.values()))) if scores else None,
            "per_problem": scores, "problems": len(scores)}


def fit_pair(problems, site: int, research: dict):
    import numpy as np
    blocks = {label: np.concatenate([d["hidden"][d["labels"] == label, site].astype(np.float64)
                                    for d in problems.values() if (d["labels"] == label).any()])
              for label in CLASSES}
    center = blocks["CALIBRATED"].mean(0)
    ot = per_problem_difference(problems, CLASSES[0], site)
    oa = per_problem_difference(problems, CLASSES[1], site)
    method = research["fit_method"]
    if method != "tact_mean_difference":
        from scipy.linalg import svd
        x = np.concatenate(list(blocks.values()))
        offset = x.mean(0)
        _, _, vt = svd(x - offset, full_matrices=False, check_finite=False)
        q = min(research["pca_components"], max(2, len(x) - 3), vt.shape[0])
        projection = vt[:q].T
        z = {label: (values - offset) @ projection for label, values in blocks.items()}
        means = {label: values.mean(0) for label, values in z.items()}
        sw, sb = np.zeros((q, q)), np.zeros((q, q))
        global_mean = ((x - offset) @ projection).mean(0)
        for label, values in z.items():
            residual = values - means[label]
            sw += residual.T @ residual
            delta = means[label] - global_mean
            sb += len(values) * np.outer(delta, delta)
        sw += max(research["ridge"] * np.trace(sw) / q, 1e-8) * np.eye(q)
        contrasts = [means[label] - means[CLASSES[2]] for label in CLASSES[:2]]
        if method == "fisher":
            ot, oa = [projection @ np.linalg.solve(sw, d) for d in contrasts]
        else:
            values, vectors = np.linalg.eigh(sw)
            inverse_sqrt = (vectors * (1 / np.sqrt(np.maximum(values, 1e-12)))) @ vectors.T
            values, vectors = np.linalg.eigh(inverse_sqrt @ sb @ inverse_sqrt)
            basis = inverse_sqrt @ vectors[:, np.argsort(values)[::-1][:2]]
            ot, oa = [projection @ (basis @ (basis.T @ d)) for d in contrasts]
    ot, oa = orthogonal_pair(ot, oa)
    sigmas = [float(((blocks["CALIBRATED"] - center) @ v).std()) for v in (ot, oa)]
    return ot, oa, center, sigmas


def fit_axes(cfg: dict, latent: Path, labels: Path, bundle_id: str) -> Path:
    import numpy as np
    root = Path(cfg["project_root"])
    manifest = read_json(latent / "manifest.json")
    verification = read_json(latent / "verification.json")
    label_manifest = read_json(labels / "manifest.json")
    if not verification["complete"] or manifest["model"] != cfg["model_descriptor"]:
        raise ValueError("Latent verification/model identity mismatch")
    if label_manifest["model"] != manifest["model"] or label_manifest["run_id"] != manifest["run_id"]:
        raise ValueError("Labels belong to a different checkpoint/run")
    parts = {"axis-train": {}, "axis-val": {}}
    audit = []
    excluded = set(read_json(Path(cfg["research"]["fit_exclusions"]))["instance_ids"])
    for task in manifest["tasks"]:
        iid, split = task["instance_id"], task["partition"]
        if split not in parts:
            raise ValueError("Evaluation data cannot be used for fitting or selection")
        if iid in excluded:
            continue
        source = root / task["source_traj"]
        if sha256(source) != task["source_sha256"]:
            raise ValueError(f"Trajectory changed: {iid}")
        label_path = labels / "labels" / f"{iid}.jsonl"
        rows = [json.loads(line) for line in label_path.read_text().splitlines() if line.strip()]
        seen = set()
        for row in rows:
            if row["instance_id"] != iid or row.get("source_sha256") != task["source_sha256"]:
                raise ValueError(f"Label trajectory provenance mismatch: {iid}")
            if row["step"] in seen or row["label"] not in CLASSES:
                raise ValueError(f"Invalid or duplicate labels: {iid}")
            seen.add(row["step"])
        first = int(cfg["research"]["exclude_step_zero_from_fit"])
        reader_count = sum(1 for m in read_json(source)["messages"] if m.get("role") == "assistant")
        if seen != set(range(first, reader_count)):
            raise ValueError(f"Label coverage incomplete: {iid}")
        file = latent / f"{iid}.npz"
        if sha256(file) != verification["files"][iid]:
            raise ValueError(f"Latent changed: {iid}")
        with np.load(file, allow_pickle=False) as z:
            hidden, steps = z["hidden"], z["step"].tolist()
            d = cfg["model_descriptor"]
            if hidden.shape != (len(steps), d["layer_count"], d["hidden_size"]) or not np.isfinite(hidden).all():
                raise ValueError(f"Invalid latent tensor: {iid}")
            keep = [r for r in rows if r["confidence"] in cfg["research"]["label_confidence"]]
            missing = [r["step"] for r in keep if r["step"] not in steps]
            if missing:
                raise ValueError(f"Labeled decision states unavailable: {iid} {missing}")
            positions = {step: index for index, step in enumerate(steps)}
            if keep:
                parts[split][iid] = {"hidden": hidden[[positions[r["step"]] for r in keep]].astype(np.float32),
                                    "labels": np.array([r["label"] for r in keep])}
        audit.append({"instance_id": iid, "partition": split, "kept_labels": len(keep),
                      "label_sha256": sha256(label_path), "latent_sha256": sha256(file)})
    for split, problems in parts.items():
        counts = Counter(y for data in problems.values() for y in data["labels"])
        if any(counts[label] == 0 for label in CLASSES):
            raise ValueError(f"All three classes required in {split}: {dict(counts)}")
    if set(parts["axis-train"]) & set(parts["axis-val"]):
        raise ValueError("Training/validation overlap")
    output = root / "artifacts/axes" / identifier(bundle_id)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    desc = cfg["model_descriptor"]
    arrays = {k: np.zeros((desc["layer_count"], desc["hidden_size"]), dtype=np.float32)
              for k in ["v_ot", "v_oa_perp", "mu_cal"]}
    arrays.update({k: np.zeros(desc["layer_count"], dtype=np.float32) for k in ["sigma_ot", "sigma_oa"]})
    curves = {}
    for site in range(desc["layer_count"]):
        ot, oa, center, sigmas = fit_pair(parts["axis-train"], site, cfg["research"])
        for key, value in zip(["v_ot", "v_oa_perp", "mu_cal", "sigma_ot", "sigma_oa"],
                              [ot, oa, center, *sigmas]):
            arrays[key][site] = value
        curves[str(site)] = {split: {"ot": macro_auc(problems, arrays["v_ot"][site], arrays["mu_cal"][site],
                                                   CLASSES[0], site),
                                   "oa": macro_auc(problems, arrays["v_oa_perp"][site], arrays["mu_cal"][site],
                                                   CLASSES[1], site)} for split, problems in parts.items()}
    scores = [(int(i), item["axis-val"]["ot"]["mean_auc"]) for i, item in curves.items()
              if item["axis-val"]["ot"]["mean_auc"] is not None]
    selected = [i for i, _ in sorted(scores, key=lambda item: (-item[1], item[0]))[:cfg["research"]["top_k_sites"]]]
    arrays["selected_indices"] = np.array(selected, dtype=np.int32)
    np.savez_compressed(output / "axes.npz", **arrays)
    bundle = {"schema_version": 1, "bundle_id": bundle_id, "model": desc,
              "selected_sites": [desc["sites"][i] for i in selected], "selected_indices": selected,
              "fit_method": cfg["research"]["fit_method"], "ranking": "validation per-problem OT/CAL macro AUC",
              "curves": curves, "input_audit": audit, "latent_manifest_sha256": sha256(latent / "manifest.json"),
              "label_manifest_sha256": sha256(labels / "manifest.json"),
              "input_hashes": cfg["input_hashes"], "research_config": cfg["research"],
              "axes_sha256": sha256(output / "axes.npz"), "complete": True}
    validate_axis_manifest(bundle, desc)
    write_json(output / "manifest.json", bundle)
    return output
