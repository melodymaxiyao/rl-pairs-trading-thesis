#!/bin/zsh
set -eu
set -o pipefail
ROOT_DIR="${0:A:h:h:h:h}"
TASK_DIR="${0:A:h}"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$TASK_DIR/runtime/tmp" MPLCONFIGDIR="$TASK_DIR/runtime/mpl_cache" XDG_CACHE_HOME="$TASK_DIR/runtime/cache"
cd "$ROOT_DIR"
exec nice -n 10 "$ROOT_DIR/.venv/bin/python" -B "$TASK_DIR/scripts/run.py" "$@"
