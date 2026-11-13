"""Experiment 12 criteria. Seeds are fixed policies, not independent histories."""
from __future__ import annotations

import numpy as np
import pandas as pd

COSTS = (0.0, 0.0005, 0.001, 0.002)
PRIMARY_COST = 0.001
FOLDS = tuple(f"fold_{i:02d}" for i in range(1, 5))
TOL = 1e-10


def bootstrap_mean_seed_sharpe(strategies: pd.DataFrame, benchmark: pd.Series, config) -> dict:
    """Resample the SAME time blocks for every policy and its benchmark.

    Statistic: mean_seed(Sharpe(policy_seed) - Sharpe(benchmark)). This is
    deliberately not Sharpe(mean daily returns), nor an average of CI endpoints.
    Intervals are conditional on these trained policies; training is not refitted.
    """
    if strategies.empty or not strategies.index.equals(benchmark.index):
        raise ValueError("All policies and benchmark must have identical dates")
    if not strategies.index.is_unique or not strategies.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    values = strategies.to_numpy(dtype=float)
    bench = benchmark.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(bench).all():
        raise ValueError("Returns must be finite")
    n = len(bench)
    if n < config.block_length:
        raise ValueError("Insufficient time observations")

    def sharpes(arr, axis):
        std = arr.std(axis=axis, ddof=0)
        return np.divide(np.sqrt(252) * arr.mean(axis=axis), std,
                         out=np.zeros_like(std), where=std > 0)

    point = float(sharpes(values, 0).mean() - sharpes(bench, 0))
    rng = np.random.default_rng(config.seed)
    samples = []
    blocks = int(np.ceil(n / config.block_length))
    for offset in range(0, config.n_repetitions, 64):
        batch = min(64, config.n_repetitions - offset)
        starts = rng.integers(0, n - config.block_length + 1, size=(batch, blocks))
        indices = (starts[:, :, None] + np.arange(config.block_length)).reshape(batch, -1)[:, :n]
        policy_sharpe = np.zeros(batch)
        for seed_idx in range(values.shape[1]):
            policy_sharpe += sharpes(values[:, seed_idx][indices], 1) / values.shape[1]
        samples.extend(policy_sharpe - sharpes(bench[indices], 1))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {"mean_delta_sharpe": point, "ci_lower_95": float(lower), "ci_upper_95": float(upper),
            "n_seeds": values.shape[1], "n_observations": n,
            "block_length": config.block_length, "n_repetitions": config.n_repetitions,
            "method": "paired_time_blocks_mean_of_seed_sharpes"}


def evaluate_success_criteria(stitched_oos, fold_summary, cost_bootstrap=None):
    results = {}
    if stitched_oos.empty or fold_summary.empty:
        return results
    wf = fold_summary[fold_summary.protocol == "walk_forward"]
    for (ticker, model), sub in stitched_oos.groupby(["ticker", "model"]):
        if model in ("cash", "buy_hold"):
            continue
        model_folds = wf[(wf.ticker == ticker) & (wf.model == model)]
        bh = wf[(wf.ticker == ticker) & (wf.model == "buy_hold")]
        expected_seeds = 1 if model in ("zscore", "sma200") else 10
        per_cost = {}
        for cost in COSTS:
            fold_details = []
            for fold in FOLDS:
                m = model_folds[(model_folds.fold == fold) & (model_folds.cost == cost)]
                b = bh[(bh.fold == fold) & (bh.cost == cost)]
                complete = len(m) == 1 and len(b) == 1 and int(m.iloc[0].n_seeds) == expected_seeds
                delta = float(m.iloc[0].sharpe_mean - b.iloc[0].sharpe_mean) if complete else None
                fold_details.append({"fold": fold, "complete": complete, "mean_delta_sharpe": delta,
                                     "positive": bool(delta is not None and np.isfinite(delta) and delta > TOL)})
            st = sub[(sub.cost == cost) & (sub.n_folds_available == 4)]
            complete = (len(st) == expected_seeds and st.seed.nunique() == expected_seeds
                        and all(f["complete"] for f in fold_details))
            delta = float((st.stitched_sharpe - st.stitched_buy_hold_sharpe).mean()) if complete else None
            ci = None
            if cost_bootstrap is not None and not cost_bootstrap.empty:
                boot = cost_bootstrap[(cost_bootstrap.ticker == ticker) & (cost_bootstrap.model == model)
                                      & (cost_bootstrap.cost == cost)]
                if len(boot) == 1:
                    ci = [float(boot.iloc[0].ci_lower_95), float(boot.iloc[0].ci_upper_95)]
            positive = sum(f["positive"] for f in fold_details)
            per_cost[str(cost)] = {
                "complete": complete, "n_positive_folds": positive, "n_expected_folds": 4,
                "n_available_folds": sum(f["complete"] for f in fold_details),
                "at_least_3_of_4": bool(complete and positive >= 3), "folds": fold_details,
                "mean_stitched_delta_sharpe": delta, "block_bootstrap_ci_95": ci,
                "mean_delta_positive": bool(delta is not None and np.isfinite(delta) and delta > TOL),
            }
        primary = per_cost[str(PRIMARY_COST)]
        complete = all(item["complete"] for item in per_cost.values())
        ci = primary["block_bootstrap_ci_95"]
        criteria = {
            "criterion_1_primary_cost_3_of_4": primary["at_least_3_of_4"],
            "criterion_2_primary_mean_delta_positive": primary["mean_delta_positive"],
            "criterion_3_primary_bootstrap_ci_lower_above_zero": bool(ci is not None and ci[0] > TOL),
            "criterion_4_survives_all_nonzero_costs": all(per_cost[str(c)]["mean_delta_positive"] for c in COSTS if c > 0),
            "criterion_5_exposure_below_95pct": bool(not model_folds.empty and model_folds.exposure_mean.mean() < 0.95),
        }
        results[f"{ticker}/{model}"] = {
            "primary_cost": PRIMARY_COST, "complete": complete, "per_cost": per_cost, **criteria,
            "all_criteria_met": bool(complete and all(criteria.values())),
            "status": "complete" if complete else "incomplete_not_evaluable",
            "protocol_note": "Primary cost/seed-mean criteria clarified after initial baseline results; not a new preregistration.",
        }
    return results
