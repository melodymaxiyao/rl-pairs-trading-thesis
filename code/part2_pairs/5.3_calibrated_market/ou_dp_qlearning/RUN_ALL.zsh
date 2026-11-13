#!/bin/zsh
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/Users/maxiyao/FYP_runs/ou_dp_qlearning_v1/runtime/tmp
export XDG_CACHE_HOME=/Users/maxiyao/FYP_runs/ou_dp_qlearning_v1/runtime/cache
export MPLCONFIGDIR=/Users/maxiyao/FYP_runs/ou_dp_qlearning_v1/runtime/mpl
exec /usr/bin/nice -n 10 /Users/maxiyao/Desktop/FYP/.venv/bin/python -I -B /Users/maxiyao/FYP_runs/ou_dp_qlearning_v1/scripts/run.py "$@"
