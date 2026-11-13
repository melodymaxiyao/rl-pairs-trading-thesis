"""Run the four-fold z-score-slope state-dimension comparison.

Only the two new z-slope agents are trained here.  Same-fold buy-and-hold,
bandit_3state, and tabular_q_12state artifacts are read from the already
completed walk-forward supplement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.single_stock.agents_zslope import (
    SixStateZSlopeAgent,
    TwelveStateZSlopeAgent,
    add_z_slope_feature,
)
from experiments.single_stock.run_single_stock import (
    _add_visit_diagnostics,
    _backtest,
    _config_digest,
    _is_complete,
    _metrics_payload,
    _run_fingerprint,
    _save_run,
    _single_instance_lock,
    _slice_test_with_prior_day,
    _write_json,
    resolve_gamma,
)
from experiments.single_stock.run_single_stock_walkforward import (
    _get_features as _get_walkforward_features,
    resolve_effective_split,
    validate_non_overlapping_test_periods,
    walkforward_run_dir,
)
from fyp_rl.audit_statistics import bootstrap_mean_seed_sharpe
from fyp_rl.config import BootstrapConfig, TabularQConfig
from fyp_rl.metrics import sharpe_ratio
from fyp_rl.statistics import moving_block_bootstrap_sharpe_diff


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_zslope.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "zslope"
PARENT_RESULTS_ROOT = EXPERIMENT_DIR / "results" / "walkforward"
MODELS = ("tabular_q_6state_zslope", "tabular_q_12state_zslope")
COMPARISON_MODELS = {
    "tabular_q_6state_zslope": "bandit_3state",
    "tabular_q_12state_zslope": "tabular_q_12state",
}
FOLD_NAMES = tuple(f"fold_{number:02d}" for number in range(1, 5))
EXPECTED_HORIZONS = (5, 10)
EXPECTED_TICKERS = ("SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL")
README_START = "<!-- ZSLOPE_RESULTS_START -->"
README_END = "<!-- ZSLOPE_RESULTS_END -->"
TOLERANCE = 1e-10


def load_zslope_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["folds"]) != FOLD_NAMES:
        raise ValueError(f"Exactly these ordered folds are required: {FOLD_NAMES}")
    if tuple(config["tickers"]) != EXPECTED_TICKERS:
        raise ValueError(f"Tickers must be exactly {EXPECTED_TICKERS}")
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError("z-slope horizons must be exactly [5, 10]")
    if not config["decision_interval_equals_reward_horizon"]:
        raise ValueError("decision_interval_equals_reward_horizon must be true")
    feature = config["z_slope_feature"]
    if int(feature["lookback_days"]) <= 0:
        raise ValueError("z_slope_feature.lookback_days must be positive")
    if float(feature["epsilon_deadzone"]) < 0:
        raise ValueError("z_slope_feature.epsilon_deadzone must be non-negative")
    if tuple(config["models"]) != MODELS:
        raise ValueError(f"Only the two z-slope models are allowed: {MODELS}")
    if config["diagnostics"]["head_to_head_vs_existing_model"] != COMPARISON_MODELS:
        raise ValueError("Head-to-head model mapping differs from the approved design")
    if config["models"][MODELS[0]]["gamma"] != 0.0:
        raise ValueError("tabular_q_6state_zslope gamma must be zero")
    half_life = int(config["gamma"]["tabular_q_12state_zslope_half_life_days"])
    resolved = {
        str(horizon): resolve_gamma(horizon, half_life) for horizon in EXPECTED_HORIZONS
    }
    if config["gamma"]["resolved_by_horizon"] != resolved:
        raise ValueError(f"Configured gamma values do not match the formula: {resolved}")
    if config["gamma"]["tabular_q_6state_zslope"] != 0.0:
        raise ValueError("Configured six-state gamma must be zero")
    if config["actions"] != ["TargetCash", "TargetLong"] or not config["no_shorting"]:
        raise ValueError("The experiment requires the two no-shorting target actions")
    if config["cost_grid"] != [0.0, 0.0005, 0.001, 0.002]:
        raise ValueError("The Experiment 2 cost grid must be retained")
    if config["seeds"] != list(range(10)) or int(config["episodes"]) != 100:
        raise ValueError("The Experiment 2 seeds and episode count must be retained")
    validate_non_overlapping_test_periods(config["folds"])
    return config


def resolve_model_gamma(model: str, horizon: int, config: dict[str, Any]) -> float:
    if model == "tabular_q_6state_zslope":
        return 0.0
    if model == "tabular_q_12state_zslope":
        return resolve_gamma(
            horizon, int(config["gamma"]["tabular_q_12state_zslope_half_life_days"])
        )
    raise ValueError(f"Unknown z-slope model: {model}")


def zslope_run_dir(
    root: Path,
    ticker: str,
    model: str,
    fold_name: str,
    horizon: int,
    cost: float,
    seed: int,
) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unsupported z-slope model: {model}")
    if fold_name not in FOLD_NAMES or horizon not in EXPECTED_HORIZONS:
        raise ValueError("Unsupported z-slope fold or horizon")
    return (
        root
        / ticker.lower()
        / model
        / fold_name
        / f"horizon_{horizon}"
        / f"cost_{cost:g}"
        / f"seed_{seed}"
    )


def _parent_run_dir(
    root: Path,
    ticker: str,
    model: str,
    fold_name: str,
    horizon: int,
    cost: float,
    seed: int | None,
) -> Path:
    return walkforward_run_dir(root, ticker, model, fold_name, horizon, cost, seed)


def _load_returns(run_dir: Path) -> pd.Series:
    path = run_dir / "daily_returns.csv"
    if not path.exists():
        raise RuntimeError(f"Required existing return artifact is missing: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)["daily_return"]


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise RuntimeError(f"Required existing metrics artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _features(
    ticker: str, split: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    base = _get_walkforward_features(ticker, split, config)
    return add_z_slope_feature(base, int(config["z_slope_feature"]["lookback_days"]))


def _slice_train(feat: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    train = feat.loc[split["actual_train_start"] : split["train_end"]]
    return train.dropna(subset=["z20", "z_slope"])


def _build_agent(
    model: str, config: dict[str, Any], horizon: int, episodes: int
):
    agent_config = TabularQConfig(
        gamma=resolve_model_gamma(model, horizon, config),
        episodes=episodes,
        epsilon_start=float(config["epsilon_start"]),
        epsilon_end=float(config["epsilon_end"]),
        decision_interval=horizon,
    )
    epsilon = float(config["z_slope_feature"]["epsilon_deadzone"])
    if model == "tabular_q_6state_zslope":
        return SixStateZSlopeAgent(agent_config, epsilon)
    if model == "tabular_q_12state_zslope":
        return TwelveStateZSlopeAgent(agent_config, epsilon)
    raise ValueError(f"Unknown z-slope model: {model}")


def _expected_run_count(config: dict[str, Any], smoke: bool) -> int:
    seeds = 1 if smoke else len(config["seeds"])
    return (
        len(config["tickers"])
        * len(config["folds"])
        * len(config["reward_horizons_days"])
        * len(MODELS)
        * len(config["cost_grid"])
        * seeds
    )


def _completion_records(root: Path) -> list[dict[str, Any]]:
    records = []
    pattern = "*/*/fold_*/horizon_*/cost_*/seed_*/completion.json"
    for marker_path in root.glob(pattern):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics_path = marker_path.parent / "metrics.json"
        if marker.get("status") == "complete" and metrics_path.exists():
            records.append(
                {
                    **json.loads(metrics_path.read_text(encoding="utf-8")),
                    "_run_dir": marker_path.parent,
                }
            )
    return records


def _update_status(
    root: Path, mode: str, completed: int, expected: int, current: dict[str, Any]
) -> None:
    _write_json(
        root / "status.json",
        {
            "mode": mode,
            "completed_runs": completed,
            "expected_runs": expected,
            "current": current,
            "updated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def _run_grid(config: dict[str, Any], config_path: Path, smoke: bool) -> Path:
    mode = "smoke" if smoke else "formal"
    root = RESULTS_ROOT / "smoke" if smoke else RESULTS_ROOT
    parent_root = PARENT_RESULTS_ROOT / "smoke" if smoke else PARENT_RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    digest = _config_digest(config)
    episodes = 2 if smoke else int(config["episodes"])
    seeds = [int(config["seeds"][0])] if smoke else [int(s) for s in config["seeds"]]
    expected = _expected_run_count(config, smoke)
    completed = 0
    started = time.time()

    for ticker in config["tickers"]:
        for fold_name, configured_split in config["folds"].items():
            split = resolve_effective_split(
                ticker, fold_name, configured_split, config["tsla_min_train_start"]
            )
            feat = _features(ticker, split, config)
            train_feat = _slice_train(feat, split)
            test_feat = _slice_test_with_prior_day(feat, split)
            for horizon in config["reward_horizons_days"]:
                benchmark_returns = _load_returns(
                    _parent_run_dir(
                        parent_root, ticker, "buy_hold", fold_name, horizon, 0.0, None
                    )
                )
                for model in MODELS:
                    for seed in seeds:
                        for cost in config["cost_grid"]:
                            gamma = resolve_model_gamma(model, horizon, config)
                            run_spec = {
                                "ticker": ticker,
                                "fold": fold_name,
                                "model": model,
                                "horizon": horizon,
                                "decision_interval_days": horizon,
                                "reward_horizon_days": horizon,
                                "cost": cost,
                                "seed": seed,
                                "gamma": gamma,
                                "episodes": episodes,
                                "mode": mode,
                                "z_slope_lookback_days": int(
                                    config["z_slope_feature"]["lookback_days"]
                                ),
                                "z_slope_epsilon_deadzone": float(
                                    config["z_slope_feature"]["epsilon_deadzone"]
                                ),
                            }
                            fingerprint = _run_fingerprint(digest, run_spec, episodes)
                            out_dir = zslope_run_dir(
                                root, ticker, model, fold_name, horizon, cost, seed
                            )
                            if _is_complete(out_dir, fingerprint, is_rl=True):
                                completed += 1
                                continue

                            print(
                                f"[{completed + 1}/{expected}] {ticker} {model} {fold_name} "
                                f"horizon={horizon} cost={cost:g} seed={seed}",
                                flush=True,
                            )
                            agent = _build_agent(model, config, horizon, episodes)
                            agent.train(train_feat, cost=float(cost), seed=seed)
                            result, target = _backtest(
                                test_feat, agent.rule_fn(), float(cost), horizon
                            )
                            summary = _add_visit_diagnostics(
                                agent.summary(),
                                int(config["diagnostics"]["flag_low_sample_cells_below"]),
                            )
                            metrics = _metrics_payload(result, benchmark_returns, run_spec)
                            metrics.update(
                                {
                                    "fold": fold_name,
                                    "requested_train_start": split["requested_train_start"],
                                    "actual_train_start": split["actual_train_start"],
                                    "train_start_truncated": split["train_start_truncated"],
                                }
                            )
                            run_config = {
                                "experiment_name": config["experiment_name"],
                                "parent_experiment": config["parent_experiment"],
                                "source_config": str(config_path),
                                "split": split,
                                "z_slope_feature": config["z_slope_feature"],
                                "run": run_spec,
                                "config_digest": digest,
                            }
                            _save_run(
                                out_dir,
                                run_config,
                                metrics,
                                result,
                                target,
                                fingerprint,
                                summary,
                            )
                            completed += 1
                            _update_status(root, mode, completed, expected, run_spec)

    if completed != expected:
        raise RuntimeError(f"Grid incomplete: found {completed} runs, expected {expected}")
    _write_json(
        root / f"_{mode.upper()}_COMPLETE.json",
        {
            "status": "complete",
            "config_digest": digest,
            "completed_runs": completed,
            "expected_runs": expected,
            "episodes": episodes,
            "elapsed_seconds_this_invocation": time.time() - started,
            "completed_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )
    return root


def validate_smoke(config: dict[str, Any], root: Path) -> None:
    records = _completion_records(root)
    expected = _expected_run_count(config, smoke=True)
    if len(records) != expected:
        raise RuntimeError(f"Smoke grid has {len(records)} complete runs; expected {expected}")
    active_cost_runs = 0
    for record in records:
        numeric = [v for v in record.values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Smoke run contains non-finite metrics: {record['_run_dir']}")
        daily = pd.read_csv(Path(record["_run_dir"]) / "daily_returns.csv")
        if not np.isfinite(daily.select_dtypes(include=[np.number]).to_numpy()).all():
            raise RuntimeError(f"Smoke run contains non-finite daily output: {record['_run_dir']}")
        cost = float(record["cost"])
        total_paid = float(record["total_cost_paid"])
        daily_paid = float(daily["cost_paid"].sum())
        if not np.isclose(total_paid, daily_paid, rtol=1e-10, atol=1e-12):
            raise RuntimeError(f"Cost total does not match daily debits: {record['_run_dir']}")
        if cost == 0.0 and abs(total_paid) > TOLERANCE:
            raise RuntimeError(f"Zero-cost run paid a transaction cost: {record['_run_dir']}")
        if cost > 0.0 and float(record["turnover"]) > 0.0:
            if total_paid <= 0.0:
                raise RuntimeError(f"Nonzero-cost turnover was not charged: {record['_run_dir']}")
            active_cost_runs += 1
    if active_cost_runs == 0:
        raise RuntimeError("Smoke grid produced no active path on which to verify costs")

    _write_json(
        root / "_SMOKE_VALIDATED.json",
        {
            "status": "validated",
            "config_digest": _config_digest(config),
            "completed_runs": len(records),
            "checks": [
                "complete_7x2x2x4x4_grid",
                "finite_metrics_and_daily_outputs",
                "daily_cost_debits_match_reported_total",
                "zero_cost_is_zero",
                "nonzero_cost_charged_when_turnover_occurs",
            ],
            "validated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def require_valid_smoke(config: dict[str, Any]) -> None:
    marker_path = RESULTS_ROOT / "smoke" / "_SMOKE_VALIDATED.json"
    if not marker_path.exists():
        raise RuntimeError("Run the z-slope --smoke successfully before --formal")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "validated" or marker.get("config_digest") != _config_digest(config):
        raise RuntimeError("The z-slope smoke marker is stale or does not match the config")


def _new_returns(
    ticker: str, model: str, fold: str, horizon: int, cost: float, seed: int
) -> pd.Series:
    return _load_returns(zslope_run_dir(RESULTS_ROOT, ticker, model, fold, horizon, cost, seed))


def _parent_returns(
    ticker: str, model: str, fold: str, horizon: int, cost: float, seed: int | None
) -> pd.Series:
    return _load_returns(
        _parent_run_dir(PARENT_RESULTS_ROOT, ticker, model, fold, horizon, cost, seed)
    )


def _stitched(parts: list[pd.Series], label: str) -> pd.Series:
    result = pd.concat(parts)
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError(f"Stitched dates overlap or are unordered for {label}")
    return result


def _new_stitched(
    ticker: str,
    model: str,
    horizon: int,
    cost: float,
    seed: int,
    folds: list[str],
) -> pd.Series:
    return _stitched(
        [_new_returns(ticker, model, fold, horizon, cost, seed) for fold in folds],
        f"{ticker}/{model}/horizon_{horizon}",
    )


def _parent_stitched(
    ticker: str,
    model: str,
    horizon: int,
    cost: float,
    seed: int | None,
    folds: list[str],
) -> pd.Series:
    return _stitched(
        [_parent_returns(ticker, model, fold, horizon, cost, seed) for fold in folds],
        f"parent/{ticker}/{model}/horizon_{horizon}",
    )


def _cumulative_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def _fold_tables(
    metrics: pd.DataFrame, config: dict[str, Any], summary_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = float(config["primary_cost"])
    selected = metrics[metrics["cost"] == cost]
    fold_rows = []
    head_rows = []
    for ticker in config["tickers"]:
        for model in MODELS:
            existing_model = COMPARISON_MODELS[model]
            for horizon in config["reward_horizons_days"]:
                for fold in config["folds"]:
                    new = selected[
                        (selected["ticker"] == ticker)
                        & (selected["model"] == model)
                        & (selected["horizon"] == horizon)
                        & (selected["fold"] == fold)
                    ]
                    if len(new) != len(config["seeds"]):
                        raise RuntimeError(f"Incomplete new fold metrics for {ticker}/{model}/{fold}")
                    parent_metrics = [
                        _load_metrics(
                            _parent_run_dir(
                                PARENT_RESULTS_ROOT,
                                ticker,
                                existing_model,
                                fold,
                                horizon,
                                cost,
                                int(seed),
                            )
                        )
                        for seed in config["seeds"]
                    ]
                    buy_hold = _load_metrics(
                        _parent_run_dir(
                            PARENT_RESULTS_ROOT, ticker, "buy_hold", fold, horizon, cost, None
                        )
                    )
                    new_sharpe = float(new["sharpe"].mean())
                    new_cumulative = float(new["cumulative_return"].mean())
                    existing_sharpe = float(np.mean([m["sharpe"] for m in parent_metrics]))
                    existing_cumulative = float(
                        np.mean([m["cumulative_return"] for m in parent_metrics])
                    )
                    fold_rows.append(
                        {
                            "ticker": ticker,
                            "model": model,
                            "horizon": horizon,
                            "fold": fold,
                            "cost": cost,
                            "mean_model_sharpe": new_sharpe,
                            "buy_hold_sharpe": float(buy_hold["sharpe"]),
                            "mean_delta_sharpe_vs_buy_hold": new_sharpe
                            - float(buy_hold["sharpe"]),
                            "mean_cumulative_return": new_cumulative,
                            "buy_hold_cumulative_return": float(buy_hold["cumulative_return"]),
                            "mean_cumulative_return_diff_vs_buy_hold": new_cumulative
                            - float(buy_hold["cumulative_return"]),
                            "n_seeds": len(new),
                        }
                    )
                    head_rows.append(
                        {
                            "ticker": ticker,
                            "new_model": model,
                            "existing_model": existing_model,
                            "horizon": horizon,
                            "fold": fold,
                            "cost": cost,
                            "new_mean_sharpe": new_sharpe,
                            "existing_mean_sharpe": existing_sharpe,
                            "direct_delta_sharpe": new_sharpe - existing_sharpe,
                            "new_mean_cumulative_return": new_cumulative,
                            "existing_mean_cumulative_return": existing_cumulative,
                            "direct_cumulative_return_difference": new_cumulative
                            - existing_cumulative,
                            "zslope_better_sharpe": new_sharpe - existing_sharpe > TOLERANCE,
                            "n_paired_seeds": len(config["seeds"]),
                        }
                    )
    folds = pd.DataFrame(fold_rows)
    heads = pd.DataFrame(head_rows)
    folds.to_csv(summary_dir / "fold_metrics.csv", index=False)
    heads.to_csv(summary_dir / "head_to_head_fold.csv", index=False)
    return folds, heads


def _bootstrap_vs_buy_hold(
    ticker: str,
    model: str,
    horizon: int,
    folds: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    cost = float(config["primary_cost"])
    benchmark = _parent_stitched(ticker, "buy_hold", horizon, cost, None, folds)
    policies = pd.DataFrame(
        {
            str(seed): _new_stitched(
                ticker, model, horizon, cost, int(seed), folds
            )
            for seed in config["seeds"]
        }
    )
    result = bootstrap_mean_seed_sharpe(
        policies, benchmark, BootstrapConfig(**config["bootstrap"])
    )
    cumulative = [_cumulative_return(policies[column]) for column in policies]
    return {
        "ticker": ticker,
        "model": model,
        "horizon": horizon,
        "cost": cost,
        "folds": ",".join(folds),
        "n_folds": len(folds),
        "mean_delta_sharpe": float(result["mean_delta_sharpe"]),
        "ci_lower_95": float(result["ci_lower_95"]),
        "ci_upper_95": float(result["ci_upper_95"]),
        "significantly_beats_buy_hold": float(result["ci_lower_95"]) > TOLERANCE,
        "mean_cumulative_return": float(np.mean(cumulative)),
        "buy_hold_cumulative_return": _cumulative_return(benchmark),
        "mean_cumulative_return_diff_vs_buy_hold": float(np.mean(cumulative))
        - _cumulative_return(benchmark),
        "n_seeds": len(config["seeds"]),
        "n_observations": len(benchmark),
        "method": str(result["method"]),
    }


def _significance_tables(
    config: dict[str, Any], summary_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    stitched_rows = []
    all_folds = list(config["folds"])
    for ticker in config["tickers"]:
        for model in MODELS:
            for horizon in config["reward_horizons_days"]:
                for fold in all_folds:
                    fold_rows.append(
                        _bootstrap_vs_buy_hold(ticker, model, horizon, [fold], config)
                    )
                stitched_rows.append(
                    _bootstrap_vs_buy_hold(ticker, model, horizon, all_folds, config)
                )
    folds = pd.DataFrame(fold_rows)
    stitched = pd.DataFrame(stitched_rows)
    folds.to_csv(summary_dir / "fold_significance.csv", index=False)
    stitched.to_csv(summary_dir / "stitched_significance.csv", index=False)
    return folds, stitched


def _fold_direction_consistency(
    fold_significance: pd.DataFrame,
    head_fold: pd.DataFrame,
    summary_dir: Path,
) -> pd.DataFrame:
    rows = []
    for (ticker, model, horizon), group in fold_significance.groupby(
        ["ticker", "model", "horizon"]
    ):
        head = head_fold[
            (head_fold["ticker"] == ticker)
            & (head_fold["new_model"] == model)
            & (head_fold["horizon"] == horizon)
        ]
        positive_bh = int((group["mean_delta_sharpe"] > TOLERANCE).sum())
        positive_head = int((head["direct_delta_sharpe"] > TOLERANCE).sum())
        rows.append(
            {
                "ticker": ticker,
                "model": model,
                "existing_model": COMPARISON_MODELS[model],
                "horizon": horizon,
                "positive_folds_vs_buy_hold": positive_bh,
                "positive_folds_vs_existing_model": positive_head,
                "all_folds_same_direction_vs_buy_hold": positive_bh in (0, 4),
                "all_folds_same_direction_vs_existing_model": positive_head in (0, 4),
                "at_least_3_of_4_better_than_existing": positive_head >= 3,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "fold_direction_consistency.csv", index=False)
    return frame


def _head_to_head_stitched(
    config: dict[str, Any], consistency: pd.DataFrame, summary_dir: Path
) -> pd.DataFrame:
    cost = float(config["primary_cost"])
    folds = list(config["folds"])
    bootstrap = BootstrapConfig(**config["bootstrap"])
    seed_rows = []
    rows = []
    for ticker in config["tickers"]:
        for model in MODELS:
            existing_model = COMPARISON_MODELS[model]
            for horizon in config["reward_horizons_days"]:
                sharpe_diffs = []
                cumulative_diffs = []
                for seed in config["seeds"]:
                    new = _new_stitched(ticker, model, horizon, cost, int(seed), folds)
                    existing = _parent_stitched(
                        ticker, existing_model, horizon, cost, int(seed), folds
                    )
                    paired = moving_block_bootstrap_sharpe_diff(new, existing, bootstrap)
                    sharpe_diff = sharpe_ratio(new) - sharpe_ratio(existing)
                    cumulative_diff = _cumulative_return(new) - _cumulative_return(existing)
                    sharpe_diffs.append(sharpe_diff)
                    cumulative_diffs.append(cumulative_diff)
                    seed_rows.append(
                        {
                            "ticker": ticker,
                            "new_model": model,
                            "existing_model": existing_model,
                            "horizon": horizon,
                            "cost": cost,
                            "seed": seed,
                            "direct_delta_sharpe": sharpe_diff,
                            "direct_cumulative_return_difference": cumulative_diff,
                            "ci_lower_95": paired["ci_lower_95"],
                            "ci_upper_95": paired["ci_upper_95"],
                            "significantly_better_for_this_seed": paired["ci_lower_95"]
                            > TOLERANCE,
                            "method": "moving_block_bootstrap_sharpe_diff",
                        }
                    )
                direction = consistency[
                    (consistency["ticker"] == ticker)
                    & (consistency["model"] == model)
                    & (consistency["horizon"] == horizon)
                ].iloc[0]
                mean_delta = float(np.mean(sharpe_diffs))
                consistent_improvement = bool(
                    mean_delta > TOLERANCE
                    and direction["at_least_3_of_4_better_than_existing"]
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "new_model": model,
                        "existing_model": existing_model,
                        "horizon": horizon,
                        "cost": cost,
                        "mean_paired_stitched_delta_sharpe": mean_delta,
                        "mean_paired_stitched_cumulative_return_difference": float(
                            np.mean(cumulative_diffs)
                        ),
                        "positive_seed_pairs": int(
                            np.sum(np.asarray(sharpe_diffs) > TOLERANCE)
                        ),
                        "positive_folds": int(
                            direction["positive_folds_vs_existing_model"]
                        ),
                        "conclusion": "改善" if consistent_improvement else "没有一致改善",
                        "conclusion_rule": "mean stitched delta Sharpe > 0 and >=3/4 positive folds",
                    }
                )
    pd.DataFrame(seed_rows).to_csv(
        summary_dir / "head_to_head_seed_bootstrap.csv", index=False
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "head_to_head_stitched.csv", index=False)
    return frame


def _leave_one_fold_out(config: dict[str, Any], summary_dir: Path) -> pd.DataFrame:
    cost = float(config["primary_cost"])
    all_folds = list(config["folds"])
    rows = []
    for ticker in config["tickers"]:
        for model in MODELS:
            for horizon in config["reward_horizons_days"]:
                full = _bootstrap_vs_buy_hold(ticker, model, horizon, all_folds, config)
                for excluded in all_folds:
                    included = [fold for fold in all_folds if fold != excluded]
                    benchmark = _parent_stitched(
                        ticker, "buy_hold", horizon, cost, None, included
                    )
                    deltas = [
                        sharpe_ratio(
                            _new_stitched(
                                ticker, model, horizon, cost, int(seed), included
                            )
                        )
                        - sharpe_ratio(benchmark)
                        for seed in config["seeds"]
                    ]
                    delta = float(np.mean(deltas))
                    rows.append(
                        {
                            "ticker": ticker,
                            "model": model,
                            "horizon": horizon,
                            "cost": cost,
                            "excluded_fold": excluded,
                            "included_folds": ",".join(included),
                            "full_four_fold_mean_delta_sharpe": full[
                                "mean_delta_sharpe"
                            ],
                            "leave_one_fold_out_mean_delta_sharpe": delta,
                            "direction_flip": (full["mean_delta_sharpe"] > TOLERANCE)
                            != (delta > TOLERANCE),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "leave_one_fold_out.csv", index=False)
    return frame


def _visit_summary(
    records: list[dict[str, Any]], config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    rows = []
    for record in records:
        visits = pd.read_csv(Path(record["_run_dir"]) / "visit_counts.csv")
        for visit in visits.to_dict("records"):
            rows.append(
                {
                    "ticker": record["ticker"],
                    "model": record["model"],
                    "fold": record["fold"],
                    "horizon": record["horizon"],
                    "cost": record["cost"],
                    "seed": record["seed"],
                    **visit,
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(summary_dir / "visit_counts_long.csv", index=False)
    distribution = (
        detail.groupby(["model", "fold", "horizon"])["visit_count"]
        .agg(n_cells="size", minimum="min", mean="mean", median="median", maximum="max")
        .reset_index()
    )
    threshold = int(config["diagnostics"]["flag_low_sample_cells_below"])
    low = (
        detail.assign(low=detail["visit_count"] < threshold)
        .groupby(["model", "fold", "horizon"])["low"]
        .sum()
        .reset_index(name="cells_below_threshold")
    )
    distribution = distribution.merge(low, on=["model", "fold", "horizon"])
    distribution.to_csv(summary_dir / "visit_count_distribution.csv", index=False)
    return distribution


def _format_generated_results(
    config: dict[str, Any],
    stitched: pd.DataFrame,
    head: pd.DataFrame,
    consistency: pd.DataFrame,
    visits: pd.DataFrame,
) -> str:
    lines = [
        README_START,
        "### 四折拼接显著性结果",
        "",
        "以下均为主成本 0.001；ΔSharpe 相对同成本 buy-and-hold，置信区间按 "
        "Experiment 2 的 paired-time-block、seed-mean 口径计算。",
        "",
        "| ticker | model | h | stitched ΔSharpe | 95% CI | 累计收益 | 显著 |",
        "|---|---|---:|---:|---|---:|:---:|",
    ]
    for row in stitched.sort_values(["model", "horizon", "ticker"]).itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.model} | {row.horizon} | {row.mean_delta_sharpe:.6f} | "
            f"[{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}] | "
            f"{row.mean_cumulative_return:.6f} | "
            f"{'是' if row.significantly_beats_buy_hold else '否'} |"
        )

    lines.extend(
        [
            "",
            "### 与 Experiment 2 的直接头对头",
            "",
            "结论规则预先固定为：四折拼接的配对 seed 平均 ΔSharpe > 0，且至少 "
            "3/4 个单折方向为正，记为“改善”；否则记为“没有一致改善”。",
            "",
            "| ticker | 新模型 | 对照模型 | h | 直接 ΔSharpe | 累计收益差 | 正向 folds | 结论 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in head.sort_values(["new_model", "horizon", "ticker"]).itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.new_model} | {row.existing_model} | {row.horizon} | "
            f"{row.mean_paired_stitched_delta_sharpe:.6f} | "
            f"{row.mean_paired_stitched_cumulative_return_difference:.6f} | "
            f"{row.positive_folds}/4 | {row.conclusion} |"
        )

    inconsistent = int(
        (~consistency["all_folds_same_direction_vs_existing_model"]).sum()
    )
    total_comparisons = len(consistency)
    low_cells = int(visits["cells_below_threshold"].sum())
    lookback = int(config["z_slope_feature"]["lookback_days"])
    epsilon = float(config["z_slope_feature"]["epsilon_deadzone"])
    lines.extend(
        [
            "",
            "### k / epsilon 的初步脆弱性判断",
            "",
            f"本轮只估计了 k={lookback}、epsilon={epsilon:g} 一个参数点，因此不能从这些结果 "
            "识别参数敏感性，也不能把该选择称为已稳健。"
            f"观察性警报：{inconsistent}/{total_comparisons} 个头对头组合在四折间方向不完全一致；"
            f"训练访问计数低于 30 的 state-action cells 共 {low_cells} 个。"
            "若方向不一致或低访问格较多，应把结论标为脆弱并在后续单独扫描 k/epsilon。",
            "",
            "逐折指标、方向一致性、leave-one-fold-out、逐 seed 直接 bootstrap 和完整访问计数分别见 "
            "`results/zslope/summary/fold_metrics.csv`、"
            "`fold_direction_consistency.csv`、`leave_one_fold_out.csv`、"
            "`head_to_head_seed_bootstrap.csv` 与 `visit_counts_long.csv`。",
            README_END,
        ]
    )
    return "\n".join(lines)


def _update_readme(generated: str) -> None:
    path = EXPERIMENT_DIR / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise RuntimeError("README z-slope generated-result markers are missing")
    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    path.write_text(text[:start] + generated + text[end:], encoding="utf-8")


def finalize_formal(config: dict[str, Any]) -> None:
    completion_path = RESULTS_ROOT / "_FORMAL_COMPLETE.json"
    if not completion_path.exists():
        raise RuntimeError("The z-slope formal completion marker is absent")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("config_digest") != _config_digest(config):
        raise RuntimeError("The formal completion marker does not match the config")
    records = _completion_records(RESULTS_ROOT)
    expected = _expected_run_count(config, smoke=False)
    if len(records) != expected:
        raise RuntimeError(f"Formal result set has {len(records)} runs; expected {expected}")

    summary_dir = RESULTS_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "_run_dir"} for record in records]
    )
    metrics["seed"] = metrics["seed"].astype(str)
    metrics.sort_values(
        ["ticker", "model", "fold", "horizon", "cost", "seed"]
    ).to_csv(summary_dir / "metrics_long.csv", index=False)
    _, head_fold = _fold_tables(metrics, config, summary_dir)
    fold_significance, stitched = _significance_tables(config, summary_dir)
    consistency = _fold_direction_consistency(fold_significance, head_fold, summary_dir)
    head = _head_to_head_stitched(config, consistency, summary_dir)
    _leave_one_fold_out(config, summary_dir)
    visits = _visit_summary(records, config, summary_dir)
    _update_readme(_format_generated_results(config, stitched, head, consistency, visits))
    _write_json(
        summary_dir / "_FINALIZED.json",
        {
            "status": "complete",
            "config_digest": _config_digest(config),
            "n_runs": len(records),
            "n_stitched_significance_rows": len(stitched),
            "n_head_to_head_rows": len(head),
            "finalized_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    config = load_zslope_config(args.config)

    if args.smoke:
        smoke_root = RESULTS_ROOT / "smoke"
        with _single_instance_lock(smoke_root, "zslope_smoke"):
            root = _run_grid(config, args.config, smoke=True)
            validate_smoke(config, root)
        print(f"z-slope smoke validation passed: {root}")
        return
    if args.finalize_only:
        with _single_instance_lock(RESULTS_ROOT, "zslope_formal"):
            finalize_formal(config)
        print(f"z-slope summaries rebuilt: {RESULTS_ROOT / 'summary'}")
        return

    require_valid_smoke(config)
    with _single_instance_lock(RESULTS_ROOT, "zslope_formal"):
        _run_grid(config, args.config, smoke=False)
        finalize_formal(config)
    print(f"z-slope run and summaries complete: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
