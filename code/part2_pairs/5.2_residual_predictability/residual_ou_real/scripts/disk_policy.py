"""The approved 8 GiB startup / 5 GiB checkpoint thresholds."""
from common import RUN, Pause
import shutil

def require_space(phase):
    assert phase in ('preflight', 'checkpoint'), ('Unknown disk phase', phase)
    free = shutil.disk_usage(RUN).free
    if phase == 'checkpoint':
        if free < 5 * 2**30:
            raise Pause(f'Free space below5GiB; checkpoints retained ({free / 2**30:.3f} GiB)')
    else:
        assert free >= 8 * 2**30, ('Free space below8GiB; no computation started', free / 2**30)
    return free
