#!/bin/zsh
set -euo pipefail
task_dir="${0:A:h}"
repo_dir="${task_dir:h:h:h}"
cd "$repo_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$repo_dir/.venv/bin/python" -B "$task_dir/scripts/run.py" "$@"
