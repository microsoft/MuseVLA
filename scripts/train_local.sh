#!/usr/bin/env bash
# Local single-GPU smoke test for MuseVLA training.
#
# Usage:
#   MUSE_DATA_ROOT=/path/to/Muse_Robot bash scripts/train_local.sh [extra args]
#
# Or pass the path explicitly:
#   bash scripts/train_local.sh --data_root_dir /path/to/Muse_Robot
#
# This script disables wandb network upload and runs `scripts/train.py` on a
# single GPU using the config at `musevla/configs/local_train.json`.

set -euo pipefail

# Resolve repo root from this script's location so we don't depend on CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve the dataset root. Prefer MUSE_DATA_ROOT env var; fall back to the
# user explicitly passing --data_root_dir in "$@".
EXTRA_ARGS=("$@")
if [[ -n "${MUSE_DATA_ROOT:-}" ]]; then
    EXTRA_ARGS=(--data_root_dir "${MUSE_DATA_ROOT}" "${EXTRA_ARGS[@]}")
elif ! printf '%s\n' "$@" | grep -q -- '--data_root_dir'; then
    echo "ERROR: dataset root is not set." >&2
    echo "  Set MUSE_DATA_ROOT=/path/to/Muse_Robot, or pass --data_root_dir /path/to/Muse_Robot." >&2
    exit 1
fi

# Disable wandb network upload for local runs (still writes local files).
export WANDB_MODE="${WANDB_MODE:-disabled}"
# Quiet HuggingFace tokenizer fork warnings.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Ensure the local checkout is importable as `musevla` even if the package
# isn't pip-installed in the active env (handy for editable/dev work).
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Use the active python's torchrun by default. Override by exporting PYTHON_BIN
# to point at the conda env you want, e.g.:
#   PYTHON_BIN=$(conda run -n musevla which python) bash scripts/train_local.sh
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="$(command -v torchrun)"
fi

# Single-GPU launch via torchrun so dist.* utilities work without changes.
"${TORCHRUN_BIN}" \
    --nproc_per_node=1 \
    --master_port=29500 \
    scripts/train.py \
    --config musevla/configs/local_train.json \
    "${EXTRA_ARGS[@]}"
