# Vendored steering runtime sources

These are physical source copies, not Git submodules or links to another project.

- `EasySteer`: ZJU-REAL/EasySteer, upstream commit
  `baecd9698f32aadef68b41536f3db477b513a0d7`, with the local TACT client,
  payload, capability-export, and regression-test changes.
- `easysteer-vllm/vllm`: the Python runtime overlay from
  ZJU-REAL/EasySteer-vllm-v1, upstream commit
  `7837b07d21c0ddffd043a30001f2c5654290aec7`, with local TACT algorithms,
  semantic sites, graph kernels, selection, and runtime evidence changes.
- Native compiled kernels come from the pinned vLLM 0.29.0 wheel; the upstream
  wheel basis is `98dff2a81d747d1dba01a47f939f48c3526d4206`.

Both upstream Apache-2.0 licenses are retained in their source directories.
The duplicate `EasySteer/vllm-steer` tree and downloaded vectors/checkpoints are
excluded. The overlay installation script generates the machine-specific
`easysteer-runtime.json` manifest, which is also excluded.

Run `scripts/setup_easysteer.sh` from the repository root to install the wheel,
local client, and overlay, and verify source hashes. This checkout packages the
Python overlay path; it is not a complete standalone build-from-source vLLM tree.
