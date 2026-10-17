# Five nested historical TACT/LDA visualizations

Use historical **Qwen3.6-35B-A3B** labels and trajectories, preserving their original
axis-train/axis-val membership. Current model rollouts and the later dataset
repartition are not mixed into these figures.

## Population and subset policy

- Whole trajectories are the selection unit.
- Retain high/medium-confidence labels, omit step 0 and reviewed whole-trajectory
  repetition-loop fit exclusions, matching the historical rules.
- 20%, 40%, 60%, 80%, 100% of each partition's trajectory count, rounded up.
- Fixed seed 20261012; subsets are nested. A small prefix contains problems with
  OT/CAL and OA/CAL contrasts so the smallest population can fit both axes.
  These anchors use class coverage, never AUC or model outcome.
- The final stage includes **all eligible historical labels**, not all raw labels
  regardless of the exclusion policy.

At each stage, fit TACT mean difference and LDA contrast anew on **that stage's
axis-train only**, including train CAL centroid/scales and LDA PCA/ridge. Select
peak sites on **that stage's axis-val only**. Plot only the selected train/val
points for that stage. No subsampling of full-population axes is substituted for
refitting, and no validation labels are used to fit PCA, axes, centroid or scales.

Original calibration: train-only PCA up to 100 components, within-class ridge
fraction 0.001. Full three-class LDA contrast is the existing project definition.
The coordinates are OT / orthogonal OA projections in units of the selected
training CAL standard deviation. Colors: OT red, OA green, CAL purple.

## Commands

```bash
cd ~/tactRefractor
.venvs/analysis/bin/python -m tact.research.progressive prepare \
  --source ~/TACT-baseline-preserved-20261012 \
  --output-id progressive-historical-20261012
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venvs/analysis/bin/python -m tact.research.progressive extract \
  --output-id progressive-historical-20261012 --memory-gib 28 --weights-gib 20
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venvs/analysis/bin/python -m tact.research.progressive plot \
  --output-id progressive-historical-20261012
```

The `prepare` stage physically copies retained inputs into the new project,
verifies preserved file hashes, rebuilds labeled close-token decision prefixes,
and checks prompt lengths against the saved server usage. It writes the exact
five subsets before any model inference. A completed manifest is immutable.

The `extract` stage uses the independent analysis environment, FP32 forward
(TF32 disabled), FP16 storage and 256-token chunks. Only bit-identical causal
prefixes reuse cache. The first actual reused prefix is checked against an
independent chunked forward. Extraction resumes only from hash-verified files.

`--memory-gib` limits the **extraction process's PyTorch allocator per GPU**;
`--weights-gib` is its lower model-placement budget. The cap also respects the
current free memory minus a six-GiB safety reserve. Existing vLLM processes,
their settings, containers and server ports are not changed or stopped.
CUDA contexts and third-party allocations are outside the allocator's precise
accounting, so the safety reserve and live GPU monitoring remain necessary.
Sharing GPUs can slow concurrent workloads; an extraction failure is visible in
its own log and must not be repaired by stopping unrelated tasks.

## Results and audit

Output: `reports/pointclouds/progressive-historical-20261012/index.html`.

Each `stage-01` … `stage-05` contains two separately fitted bundles, exact
selection/count manifests, and a gallery under
`tact_mean_difference/visualizations/`. The gallery includes both methods'
train/val point clouds, validation comparison, AUC curves, PNG/PDF/SVG, CSV/NPZ
and projection/AUC integrity checks. The main gallery links all five stages.

`manifest.json` records the historical input hashes, model identity, exclusions,
label policy, seed, selected trajectory IDs and class counts. `latents/` records
the extracted states and hashes; `extraction-runtime.json` records GPU caps;
`status.json`, `pipeline.log` and `pipeline-exit-code.txt` report live completion
or failure. `complete` is written only after all five galleries are generated.

Stage-dependent validation peak AUC is a selection metric, not held-out eval
performance. Changes in these plots reflect both sample composition and
refitted axes; they alone do not establish steering effectiveness.
