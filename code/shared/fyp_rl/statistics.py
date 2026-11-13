"""Statistical procedures: seed aggregation and moving-block bootstrap for Sharpe differences."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BootstrapConfig
from .metrics import sharpe_ratio


def seed_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def moving_block_bootstrap_sharpe_diff(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    config: BootstrapConfig | None = None,
) -> dict:
    """Pair strategy and benchmark daily returns, resample overlapping blocks of
    length `block_length` with replacement, and report the bootstrap distribution
    of Sharpe(strategy) - Sharpe(benchmark)."""
    config = config or BootstrapConfig()
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    n = len(aligned)
    if n < config.block_length:
        raise ValueError("Not enough overlapping observations for the requested block length")

    strat = aligned["strategy"].to_numpy()
    bench = aligned["benchmark"].to_numpy()
    point_estimate = sharpe_ratio(aligned["strategy"]) - sharpe_ratio(aligned["benchmark"])

    rng = np.random.default_rng(config.seed)
    n_blocks = int(np.ceil(n / config.block_length))
    max_start = n - config.block_length

    # vectorized moving-block resample: build (n_repetitions, n_blocks) start indices,
    # expand into (n_repetitions, n_blocks*block_length) index matrix, then gather once.
    starts = rng.integers(0, max_start + 1, size=(config.n_repetitions, n_blocks))
    offsets = np.arange(config.block_length)
    idx_matrix = (starts[:, :, None] + offsets[None, None, :]).reshape(config.n_repetitions, -1)[:, :n]

    strat_samples = strat[idx_matrix]
    bench_samples = bench[idx_matrix]
    strat_mean = strat_samples.mean(axis=1)
    strat_std = strat_samples.std(axis=1, ddof=0)
    bench_mean = bench_samples.mean(axis=1)
    bench_std = bench_samples.std(axis=1, ddof=0)
    ann = np.sqrt(252.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        strat_sharpe = np.where(strat_std > 0, ann * strat_mean / strat_std, 0.0)
        bench_sharpe = np.where(bench_std > 0, ann * bench_mean / bench_std, 0.0)
    diffs = strat_sharpe - bench_sharpe

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "point_estimate": float(point_estimate),
        "bootstrap_mean": float(np.mean(diffs)),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "n_repetitions": config.n_repetitions,
        "block_length": config.block_length,
        "n_observations": n,
    }
