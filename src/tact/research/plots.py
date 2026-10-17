"""Render saved train-fitted projections; plotting never fits or selects on eval."""
from __future__ import annotations

import csv
import html
from pathlib import Path

from tact.artifacts import read_json, sha256, write_json
from tact.research.axes import CLASSES, METHODS, auc

COLORS = dict(zip(CLASSES, ("#d62728", "#2ca02c", "#7b2cbf")))
SHORT = dict(zip(CLASSES, ("OT", "OA", "CAL")))
TITLES = dict(zip(METHODS, ("TACT mean difference", "LDA contrast", "Regularized Fisher")))
SPLITS = ("axis-train", "axis-val")


def _export(fig, directory, name):
    import matplotlib.pyplot as plt

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _site_name(manifest, index):
    site = manifest["model"]["sites"][index]
    return "final norm" if site["kind"] == "final_norm" else f"decoder block {site['block_index']}"


def _peak(manifest, axis):
    candidates = [(int(i), values["axis-val"][axis]["mean_auc"])
                  for i, values in manifest["curves"].items()
                  if values["axis-val"][axis]["mean_auc"] is not None]
    if not candidates:
        raise ValueError(f"No validation AUC available for {axis}")
    return min(candidates, key=lambda item: (-item[1], item[0]))[0]


def _load_projections(bundle, manifest):
    import numpy as np

    files = manifest.get("projection_files")
    if not files or set(files) != set(SPLITS):
        raise ValueError("This bundle has no saved labeled projections. Refit with the updated "
                         "fit command and verified latents/labels; historical steering axes alone "
                         "cannot reconstruct a point cloud.")
    if sha256(bundle / "axes.npz") != manifest["axes_sha256"]:
        raise ValueError("Axis bundle changed")
    data = {}
    n_sites = manifest["model"]["layer_count"]
    if set(manifest["curves"]) != {str(i) for i in range(n_sites)}:
        raise ValueError("AUC curves do not cover the model sites")
    for split in SPLITS:
        file = bundle / files[split]["path"]
        if not file.resolve().is_relative_to(bundle.resolve()) or file.is_symlink():
            raise ValueError("Projection path leaves the physical bundle")
        if sha256(file) != files[split]["sha256"]:
            raise ValueError(f"Projection changed: {split}")
        with np.load(file, allow_pickle=False) as stored:
            rows = {key: stored[key] for key in stored.files}
        n = files[split]["rows"]
        if (rows["projection"].shape != (n, n_sites, 2)
                or not np.isfinite(rows["projection"]).all()
                or rows["site_index"].tolist() != list(range(n_sites))
                or any(rows[key].shape != (n,) for key in ("label", "instance_id", "step"))
                or not set(rows["label"].tolist()).issubset(CLASSES)):
            raise ValueError(f"Invalid projection tensor: {split}")
        keys = list(zip(rows["instance_id"].tolist(), rows["step"].tolist()))
        if len(set(keys)) != n:
            raise ValueError(f"Duplicate projected steps: {split}")
        for site in range(n_sites):
            for dim, axis in enumerate(("ot", "oa")):
                values = {}
                for iid in np.unique(rows["instance_id"]):
                    mask = rows["instance_id"] == iid
                    pos = mask & (rows["label"] == CLASSES[dim])
                    neg = mask & (rows["label"] == CLASSES[2])
                    if pos.any() and neg.any():
                        values[str(iid)] = auc(rows["projection"][pos, site, dim],
                                               rows["projection"][neg, site, dim])
                expected = manifest["curves"][str(site)][split][axis]
                if set(values) != set(expected["per_problem"]) or any(
                    not np.isclose(value, expected["per_problem"][iid], atol=1e-10, rtol=0)
                    for iid, value in values.items()
                ):
                    raise ValueError(f"Projected AUC mismatch: {split} site={site} {axis}")
                mean = float(np.mean(list(values.values()))) if values else None
                if ((mean is None) != (expected["mean_auc"] is None)
                        or (mean is not None and not np.isclose(
                            mean, expected["mean_auc"], atol=1e-10, rtol=0))):
                    raise ValueError(f"Macro AUC mismatch: {split} site={site} {axis}")
        data[split] = rows
    if set(data[SPLITS[0]]["instance_id"]) & set(data[SPLITS[1]]["instance_id"]):
        raise ValueError("Training/validation projection overlap")
    audit = {(r["partition"], r["instance_id"]): r["kept_labels"]
             for r in manifest["input_audit"] if r["kept_labels"]}
    counts = {(split, str(iid)): int((rows["instance_id"] == iid).sum())
              for split, rows in data.items() for iid in np.unique(rows["instance_id"])}
    if counts != audit:
        raise ValueError("Projection coverage differs from fitted input audit")
    return data


def _draw(ax, rows, site, title):
    xy = rows["projection"][:, site]
    for label in reversed(CLASSES):
        mask = rows["label"] == label
        ax.scatter(xy[mask, 0], xy[mask, 1], color=COLORS[label], s=13, alpha=0.45,
                   linewidths=0, label=f"{SHORT[label]} (n={mask.sum()})", rasterized=True)
    ax.axhline(0, color="#555555", linewidth=0.6)
    ax.axvline(0, color="#555555", linewidth=0.6)
    ax.set(xlabel="OT projection / train CAL standard deviation",
           ylabel="Orthogonal OA projection / train CAL standard deviation", title=title)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)


def _compatible(loaded):
    """Reject comparisons that would disguise changed checkpoint, fit data or labels."""
    import numpy as np

    first = loaded[0]
    for _, manifest, data in loaded[1:]:
        for field in ("model", "latent_manifest_sha256", "label_manifest_sha256", "input_audit"):
            if manifest.get(field) != first[1].get(field):
                raise ValueError(f"Comparison bundles differ in {field}")
        for split in SPLITS:
            for key in ("instance_id", "step", "label", "site_index"):
                if not np.array_equal(data[split][key], first[2][split][key]):
                    raise ValueError(f"Comparison projection rows differ: {split} {key}")


def plot_bundle(bundle: Path, comparison_bundles=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundles = [Path(bundle), *map(Path, comparison_bundles)]
    loaded = []
    for path in bundles:
        manifest = read_json(path / "manifest.json")
        if manifest.get("fit_method") not in METHODS or not manifest.get("curves"):
            raise ValueError("Bundle lacks fitted curves/projections; use newly fitted bundles "
                             "or the separately preserved historical visualization archive")
        loaded.append((path, manifest, _load_projections(path, manifest)))
    methods = [item[1]["fit_method"] for item in loaded]
    if len(methods) != len(set(methods)):
        raise ValueError("Comparison requires distinct fitting methods")
    _compatible(loaded)
    output = Path(bundle) / "visualizations"
    output.mkdir(exist_ok=True)
    (output / "verification.json").unlink(missing_ok=True)
    figures, curve_rows, selections = [], [], {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for method_index, (_, manifest, _) in enumerate(loaded):
        method = manifest["fit_method"]
        color = ("#b65325", "#2766aa", "#18815d")[METHODS.index(method)]
        sites = sorted(map(int, manifest["curves"]))
        for dim, axis in enumerate(("ot", "oa")):
            for split, style in (("axis-train", "--"), ("axis-val", "-")):
                values = [manifest["curves"][str(i)][split][axis]["mean_auc"] for i in sites]
                axes[dim].plot(sites, values, style, color=color,
                               marker="o" if method_index == 2 and split == "axis-val" else None,
                               markersize=3, markevery=3, label=f"{TITLES[method]} / {split}")
                for site, value in zip(sites, values):
                    curve_rows.append({"method": method, "split": split, "axis": axis,
                                       "site_index": site, "site": _site_name(manifest, site),
                                       "macro_auc": value,
                                       "n_problems": manifest["curves"][str(site)][split][axis]["problems"]})
    for ax, axis in zip(axes, ("OT", "OA")):
        ax.set(xlabel="Site index (0-based; final site = final norm)",
               ylabel="Per-problem macro AUC", ylim=(0, 1.02), title=f"{axis} / CAL")
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7)
    _export(fig, output, "auc-by-site")
    figures.append("auc-by-site")
    _csv(output / "auc-by-site.csv", curve_rows)
    for path, manifest, data in loaded:
        method = manifest["fit_method"]
        name = {"tact_mean_difference": "tact", "lda_contrast": "lda_contrast",
                "fisher": "fisher"}[method] + "-pointcloud"
        peak_sites = {axis: _peak(manifest, axis) for axis in ("ot", "oa")}
        selections[method] = peak_sites
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), constrained_layout=True)
        point_rows = []
        for column, axis in enumerate(("ot", "oa")):
            site = peak_sites[axis]
            for row_index, split in enumerate(SPLITS):
                rows = data[split]
                score = manifest["curves"][str(site)][split][axis]["mean_auc"]
                _draw(axes[row_index, column], rows, site,
                      f"{split} | {_site_name(manifest, site)} | {axis.upper()} AUC {score:.3f}")
                for index, xy in enumerate(rows["projection"][:, site]):
                    point_rows.append({"method": method, "split": split, "panel": axis,
                        "site_index": site, "site": _site_name(manifest, site),
                        "instance_id": str(rows["instance_id"][index]),
                        "step": int(rows["step"][index]), "label": str(rows["label"][index]),
                        "ot_projection_sd": float(xy[0]), "oa_projection_sd": float(xy[1])})
        fig.suptitle(f"{TITLES[method]} | fitted on axis-train only\n"
                     "Columns: validation-selected OT and OA peak sites; all fitted labeled rows")
        _export(fig, output, name)
        _csv(output / f"{name}.csv", point_rows)
        figures.append(name)
        for split, rows in data.items():
            import numpy as np
            np.savez_compressed(output / f"{name.removesuffix('-pointcloud')}-{split}-projections.npz",
                                **rows)
    if len(loaded) > 1:
        fig, axes = plt.subplots(len(loaded), 2, figsize=(13, 4.7 * len(loaded)),
                                 constrained_layout=True)
        for index, (_, manifest, data) in enumerate(loaded):
            method = manifest["fit_method"]
            for column, axis in enumerate(("ot", "oa")):
                site = selections[method][axis]
                score = manifest["curves"][str(site)]["axis-val"][axis]["mean_auc"]
                _draw(axes[index, column], data["axis-val"], site,
                      f"{TITLES[method]} | {_site_name(manifest, site)}\n"
                      f"{axis.upper()} validation AUC {score:.3f}")
        fig.suptitle("Validation projections onto train-fitted axis pairs\n"
                     "LDA/Fisher may coincide under the existing full three-class formulas")
        _export(fig, output, "validation-method-comparison")
        figures.append("validation-method-comparison")
    if len(loaded) == 1:
        import shutil
        for suffix in ("png", "pdf", "svg"):
            shutil.copyfile(output / f"auc-by-site.{suffix}", Path(bundle) / f"auc-by-site.{suffix}")
    report = ["# Drift-axis point clouds", "", ("Axes fitted on axis-train only. Peak sites "
              "selected on axis-val; these AUCs are selection metrics, not held-out eval results."),
              "", ("OT = red; OA = green; CAL = purple. Coordinates use each method's saved "
              "training CAL centroid and standard deviations. No validation refit or downsampling."),
              "", ("LDA contrast and Fisher may coincide under the existing full three-class "
              "definitions; matching plots are not independent evidence."), "",
              "| Method | OT peak site | OA peak site | Train rows | Validation rows |",
              "|---|---|---|---:|---:|"]
    for _, manifest, data in loaded:
        method = manifest["fit_method"]
        report.append(f"| {TITLES[method]} | {_site_name(manifest, selections[method]['ot'])} | "
                      f"{_site_name(manifest, selections[method]['oa'])} | "
                      f"{len(data['axis-train']['label'])} | {len(data['axis-val']['label'])} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>TACT axis point clouds</title><style>body{font:16px system-ui;',
            'max-width:1150px;margin:40px auto;padding:0 20px}img{width:100%}</style>',
            '<h1>Drift-axis point clouds</h1><p>Train-fitted axes; validation-selected sites. ',
            '<a href="REPORT.md">Report</a> · <a href="verification.json">Verification</a></p>']
    for name in figures:
        safe = html.escape(name)
        page.append(f'<h2>{safe}</h2><a href="{safe}.pdf">PDF</a> · '
                    f'<a href="{safe}.svg">SVG</a><img src="{safe}.png" alt="{safe}">')
    (output / "index.html").write_text("".join(page) + "</html>\n", encoding="utf-8")
    result = {"complete": True, "no_projection_refit": True, "all_points_plotted": True,
              "all_site_auc_recomputed": True, "site_numbering": "0-based; final site is final_norm",
              "peak_sites": selections, "bundles": [
                  {"bundle_id": m["bundle_id"], "manifest_sha256": sha256(p / "manifest.json"),
                   "projection_files": m["projection_files"]} for p, m, _ in loaded],
              "files": {p.name: sha256(p) for p in sorted(output.iterdir())
                        if p.is_file() and p.name != "verification.json"}}
    write_json(output / "verification.json", result)
    return {"complete": True, "path": str(output), "methods": methods}
