#!/bin/zsh
set -euo pipefail
RUN_DIR=/Users/maxiyao/FYP_runs/partial_coint_kalman_v1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/runtime/tmp" XDG_CACHE_HOME="$RUN_DIR/runtime/cache" MPLCONFIGDIR="$RUN_DIR/runtime/mpl"
exec nice -n 10 /Users/maxiyao/Desktop/FYP/.venv/bin/python -I -B "$RUN_DIR/scripts/run_partial.py" "$@"
