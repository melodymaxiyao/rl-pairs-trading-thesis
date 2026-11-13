#!/bin/zsh
set -euo pipefail
cd /Users/maxiyao/FYP_runs/pci_signal_trading_v1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs runtime
launch_id=$(date -u +%Y%m%dT%H%M%SZ)
log_path="logs/full_${launch_id}.log"
print "Progress log: /Users/maxiyao/FYP_runs/pci_signal_trading_v1/${log_path}"
nice -n 10 /Users/maxiyao/Desktop/FYP/.venv/bin/python -u -B scripts/run_all.py 2>&1 | tee "$log_path"
