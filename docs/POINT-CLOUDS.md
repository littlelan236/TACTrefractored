# Three-method point-cloud visualization

The three methods are **TACT mean difference**, **LDA contrast**, and **regularized
Fisher**. They are the existing `research.fit_method` definitions, not three generic
dimensionality-reduction algorithms. Plotting never refits on validation labels.

## New research artifacts

After decision-point extraction and verified latent generation, fit all three
methods using one run snapshot and label set:

```bash
cd ~/tactRefractor
.venvs/runner/bin/tact fit --run-id RUN --set-id LATENTS --label-set LABELS \
  --bundle-id three-methods --all-methods
.venvs/runner/bin/tact plot --run-id RUN --bundle-id three-methods-tact \
  --compare-bundle-id three-methods-lda_contrast \
  --compare-bundle-id three-methods-fisher
```

The three generated bundles are independent, model-bound AxisBundles. The command
does not change `configs/experiment.yaml` or an existing run's resolved config.
Each bundle records its effective fitting method/research parameters. A single
`tact fit` still uses the run's configured fitting method and the exact bundle ID.
A single `tact plot` generates one method's point cloud.

Output: `artifacts/axes/three-methods-tact/visualizations/index.html`. The gallery
includes:

- `tact-pointcloud`, `lda_contrast-pointcloud`, `fisher-pointcloud`: a 2×2 figure
  per method; rows are train/validation, columns are the validation-selected OT
  and OA peak sites. Both coordinates are plotted at the same site in each panel.
- `validation-method-comparison`: the three methods side by side on validation.
- `auc-by-site`: train dashed / validation solid macro AUC curves.
- PNG/PDF/SVG figures, per-panel point CSVs, full all-site projection NPZs,
  `REPORT.md`, and hashed `verification.json`.

OT = red (`#d62728`), OA = green (`#2ca02c`), CAL = purple (`#7b2cbf`). Each
coordinate is centered on that method's saved train CAL centroid and divided by
its train CAL standard deviation. All retained high/medium-confidence fitted rows
are plotted without sampling. Fit exclusions and step-zero policy continue to
come from the run's research configuration.

Peak sites are selected using validation per-problem macro AUC, with smallest
site index breaking ties. They are selection-set metrics, not held-out eval
scores. Site indices are **0-based** and the final site is **final_norm**; older
figures used 1-based HF hidden indices excluding embedding row 0.

The LDA-contrast and Fisher definitions can span the same full three-class
contrast subspace. Identical or near-identical plots are expected for some data
and are not independent confirmation.

## Integrity checks and old bundles

Fitting now saves `axis-train-projections.npz` and `axis-val-projections.npz`
inside each bundle, with labels, instance IDs, steps and all-site coordinates.
These small projection artifacts make replotting independent of latent tensors,
model weights and GPU inference.

Plotting checks axes/projection hashes, dynamic shape/site coverage, train/val
separation, complete fitted-row coverage, and recomputes every site's per-problem
and macro AUC from the actual plotted projections. Comparisons require matching
model identity, latent/label manifest hashes, input audits and row IDs.

Old bundles containing only steering vectors cannot reconstruct a point cloud.
Refit from verified latents/labels with the updated implementation. An actionable
error is returned rather than inventing points or silently displaying an empty plot.
The current server has no retained latent dataset; no new GPU extraction was
started to restore this visualization feature.

## Preserved historical figures

The pre-refactor noloop46 exports were physically copied from the local retained
archive into:

```text
reports/pointclouds/original-noloop46/visualizations/index.html
```

This is a historical 35B dataset (33 train trajectories / 2,016 labeled steps;
15 validation trajectories / 1,081 steps), not a result for current 27B rollouts.
The archive contains the three original point clouds, method comparison, AUC
curves, CSV/NPZ coordinates, original report/summary and `RESTORATION.json` with
source file hashes. No original code/project/venv links are needed to open it.
The restored steering bundle was not modified and no historical axes were refitted.
