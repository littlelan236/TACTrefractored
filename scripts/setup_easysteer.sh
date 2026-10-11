#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
if [ ! -x .venvs/easysteer/bin/python ]; then
  "$UV" venv --python 3.12 .venvs/easysteer
fi
"$UV" pip install --python .venvs/easysteer/bin/python -r locks/easysteer.in
"$UV" pip install --python .venvs/easysteer/bin/python --no-deps -e .
"$UV" pip install --python .venvs/easysteer/bin/python --no-deps --no-build-isolation -e vendor/EasySteer
.venvs/easysteer/bin/python scripts/install_easysteer_overlay.py
.venvs/easysteer/bin/python scripts/verify_easysteer_runtime.py
.venvs/easysteer/bin/python vendor/EasySteer/tools/export_client_contracts.py
"$UV" pip freeze --python .venvs/easysteer/bin/python > locks/easysteer.txt
