#!/bin/zsh
set -euo pipefail
fast_universe_task_dir="${0:A:h}"
cd "${fast_universe_task_dir}/../../.."
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec .venv/bin/python -B "${fast_universe_task_dir}/scripts/run.py" "$@"
