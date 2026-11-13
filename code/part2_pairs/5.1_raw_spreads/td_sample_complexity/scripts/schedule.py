"""Scope gate: planning is separate from execution authorization.

No numerical simulation or fitting happens in this module. Tier 2 is an
explicitly deferred design, including when a caller requests it directly.
"""
import os

ACTIVE_TIERS = (1, 3)
DEFERRED_D_LABEL = 'not run (Tier 2 deferred)'
K_GRID = (0, 0.5, 1, 2, 4, 8)


def default_workers():
    return max(1, (os.cpu_count() or 1) // 2)


def authorize(tier, *, tier1_verified=False):
    if tier == 2:
        raise RuntimeError('Tier 2 deferred, not run: execution is forbidden')
    if tier not in ACTIVE_TIERS:
        raise ValueError('Unknown tier')
    if tier == 3 and not tier1_verified:
        raise RuntimeError('Tier 1 must be sealed and verified before Tier 3')


def world_jobs(tier, *, tier1_verified=False):
    authorize(tier, tier1_verified=tier1_verified)
    if tier == 1:
        return [dict(tier=1, k=k, D=1, world=w, case='gaussian')
                for k in K_GRID for w in range(5)]
    return [dict(tier=3, k=1, D=1, world=w, case=case)
            for case in ('protocol_ablation', 'fat_tails', 'heterogeneous')
            for w in range(5)]


def runtime_estimates(measured_serial_seconds, chosen_workers, *, cpus=None):
    """Explicit ideal-parallel extrapolation, never an observed speedup claim.

The pilot runner supplies projected serial costs based on separately measured
fit, executor, oracle and path-generation timings. Actual elapsed pilot times
must be stored with those projections. Execution order still follows the gate.
"""
    cpus = (os.cpu_count() or 1) if cpus is None else cpus
    if chosen_workers < 1 or cpus < 1:
        raise ValueError('Worker counts must be positive')
    if set(measured_serial_seconds) != set(ACTIVE_TIERS):
        raise ValueError('Estimate only the active Tier 1 and Tier 3')
    rows = []
    for tier in ACTIVE_TIERS:
        cost = measured_serial_seconds[tier]
        if cost < 0:
            raise ValueError('Negative projected runtime')
        jobs = len(world_jobs(tier, tier1_verified=(tier == 3)))
        for label, workers in [('chosen', chosen_workers), ('os.cpu_count()', cpus)]:
            effective = min(workers, jobs)
            rows.append(dict(tier=tier, worker_setting=label, workers=workers,
                             projected_serial_seconds=cost,
                             estimated_parallel_seconds=cost / effective,
                             assumption='ideal parallel scaling; not measured '
                             'at this worker count; shared-machine contention '
                             'and serial overhead can increase elapsed time'))
    return rows
