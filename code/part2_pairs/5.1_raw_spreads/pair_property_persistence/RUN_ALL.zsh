#!/bin/zsh
set -euo pipefail
persistence_task_dir="${0:A:h}"
cd "${persistence_task_dir}/../../.."
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec .venv/bin/python -B "${persistence_task_dir}/scripts/run.py" "$@"
