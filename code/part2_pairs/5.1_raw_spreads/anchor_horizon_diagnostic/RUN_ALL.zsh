#!/bin/zsh
set -eu
task_dir=${0:A:h}
project_dir=${task_dir:h:h:h}
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec "$project_dir/.venv/bin/python" -B "$task_dir/scripts/run.py" "$@"
