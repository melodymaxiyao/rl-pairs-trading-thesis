"""Run the four-fold walk-forward supplement to the single-stock experiment."""
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

from experiments.single_stock.run_single_stock import (
    ALL_MODELS,
    BASELINE_MODELS,
    RL_MODELS,
    _add_visit_diagnostics,
    _backtest,
    _baseline_target,
    _build_agent,
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
from fyp_rl.audit_statistics import bootstrap_mean_seed_sharpe
from fyp_rl.config import BootstrapConfig
from fyp_rl.data import load_ohlcv
from fyp_rl.execution import simulate_positions
from fyp_rl.features import build_feature_frame
from fyp_rl.metrics import sharpe_ratio
from fyp_rl.statistics import moving_block_bootstrap_sharpe_diff


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_walkforward.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "walkforward"
FOLD_NAMES = tuple(f"fold_{number:02d}" for number in range(1, 5))
EXPECTED_HORIZONS = (5, 10)
README_START = "<!-- WALKFORWARD_RESULTS_START -->"
README_END = "<!-- WALKFORWARD_RESULTS_END -->"
TOLERANCE = 1e-10


def load_walkforward_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["folds"]) != FOLD_NAMES:
        raise ValueError(f"Exactly these ordered folds are required: {FOLD_NAMES}")
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError("Walk-forward horizons must be exactly [5, 10]; 30 days is excluded")
    if not config["decision_interval_equals_reward_horizon"]:
        raise ValueError("decision_interval_equals_reward_horizon must be true")
    resolved = {
        str(horizon): resolve_gamma(horizon, int(config["gamma"]["tabular_q_half_life_days"]))
        for horizon in EXPECTED_HORIZONS
    }
    if config["gamma"]["resolved_by_horizon"] != resolved:
        raise ValueError(f"Configured gamma values do not match resolve_gamma: {resolved}")
    if config["models"]["rule_baselines"] != list(BASELINE_MODELS):
        raise ValueError("Configured baselines differ from the parent experiment")
    if config["actions"] != ["TargetCash", "TargetLong"] or not config["no_shorting"]:
        raise ValueError("The supplement requires the parent experiment's no-shorting actions")
    validate_non_overlapping_test_periods(config["folds"])
    return config


def validate_non_overlapping_test_periods(folds: dict[str, dict[str, str]]) -> None:
    intervals = sorted(
        (pd.Timestamp(split["test_start"]), pd.Timestamp(split["test_end"]), fold_name)
        for fold_name, split in folds.items()
    )
    for (_, previous_end, previous_name), (current_start, _, current_name) in zip(
        intervals, intervals[1:]
    ):
        if current_start <= previous_end:
            raise ValueError(f"Test periods overlap: {previous_name} and {current_name}")


def resolve_effective_split(
    ticker: str, fold_name: str, configured_split: dict[str, str], tsla_min_train_start: str
) -> dict[str, Any]:
    requested_start = pd.Timestamp(configured_split["train_start"])
    minimum_start = pd.Timestamp(tsla_min_train_start) if ticker == "TSLA" else requested_start
    actual_start = max(requested_start, minimum_start)
    return {
        "name": fold_name,
        "requested_train_start": configured_split["train_start"],
        "actual_train_start": actual_start.strftime("%Y-%m-%d"),
        "train_start": actual_start.strftime("%Y-%m-%d"),
        "train_end": configured_split["train_end"],
        "test_start": configured_split["test_start"],
        "test_end": configured_split["test_end"],
        "train_start_truncated": actual_start > requested_start,
        "train_start_constraint": tsla_min_train_start if ticker == "TSLA" else None,
    }


def _get_features(ticker: str, split: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    raw_start = (
        pd.Timestamp(split["actual_train_start"])
        - pd.Timedelta(days=int(config["feature_warmup_calendar_days"]))
    ).strftime("%Y-%m-%d")
    return build_feature_frame(load_ohlcv(ticker, raw_start, split["test_end"]))


def _slice_train(feat: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    train = feat.loc[split["actual_train_start"] : split["train_end"]]
    return train.dropna(subset=["z20", "trend", "sma200"])


def walkforward_run_dir(
    root: Path,
    ticker: str,
    model: str,
    fold_name: str,
    horizon: int,
    cost: float,
    seed: int | None,
    allowed_horizons: tuple[int, ...] = EXPECTED_HORIZONS,
) -> Path:
    if horizon not in allowed_horizons:
        raise ValueError(f"Unsupported walk-forward horizon: {horizon}")
    seed_name = "seed_NA" if seed is None else f"seed_{seed}"
    return (
        root
        / ticker.lower()
        / model
        / fold_name
        / f"horizon_{horizon}"
        / f"cost_{cost:g}"
        / seed_name
    )


def assert_no_30day_outputs(root: Path) -> None:
    offenders = sorted(path for path in root.rglob("horizon_30") if path.is_dir())
    if offenders:
        raise RuntimeError(f"30-day output is forbidden in this supplement: {offenders[0]}")


def _expected_run_count(config: dict[str, Any], smoke: bool) -> int:
    seeds = 1 if smoke else len(config["seeds"])
    per_fold_horizon_cost = len(BASELINE_MODELS) + len(RL_MODELS) * seeds
    return (
        len(config["tickers"])
        * len(config["folds"])
        * len(config["reward_horizons_days"])
        * len(config["cost_grid"])
        * per_fold_horizon_cost
    )


def _completion_records(root: Path) -> list[dict[str, Any]]:
    records = []
    pattern = "*/*/fold_*/horizon_*/cost_*/seed_*/completion.json"
    for marker_path in root.glob(pattern):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete":
            continue
        metrics_path = marker_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
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


def build_run_config(
    config: dict[str, Any], config_path: Path, split: dict[str, Any], run_spec: dict[str, Any]
) -> dict[str, Any]:
    """Build persisted metadata, including any ticker-specific train truncation."""
    return {
        "experiment_name": config["experiment_name"],
        "parent_experiment": config["parent_experiment"],
        "source_config": str(config_path),
        "split": split,
        "run": run_spec,
        "config_digest": _config_digest(config),
    }


def _run_grid(config: dict[str, Any], config_path: Path, smoke: bool) -> Path:
    mode = "smoke" if smoke else "formal"
    root = RESULTS_ROOT / "smoke" if smoke else RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    assert_no_30day_outputs(root)
    digest = _config_digest(config)
    episodes = 2 if smoke else int(config["episodes"])
    seeds = [int(config["seeds"][0])] if smoke else [int(seed) for seed in config["seeds"]]
    expected = _expected_run_count(config, smoke)
    completed = 0
    started = time.time()

    for ticker in config["tickers"]:
        for fold_name, configured_split in config["folds"].items():
            split = resolve_effective_split(
                ticker, fold_name, configured_split, config["tsla_min_train_start"]
            )
            feat = _get_features(ticker, split, config)
            train_feat = _slice_train(feat, split)
            test_feat = _slice_test_with_prior_day(feat, split)
            for horizon in config["reward_horizons_days"]:
                benchmark_target = _baseline_target("buy_hold", test_feat, horizon)
                benchmark_returns = simulate_positions(
                    test_feat[["Open", "Close"]],
                    benchmark_target,
                    cost=0.0,
                    initial_position=0.0,
                ).daily_returns
                for model in ALL_MODELS:
                    model_seeds: list[int | None] = [None] if model in BASELINE_MODELS else seeds
                    for seed in model_seeds:
                        for cost in config["cost_grid"]:
                            gamma = (
                                0.0
                                if model in BASELINE_MODELS or model == "bandit_3state"
                                else resolve_gamma(
                                    horizon,
                                    int(config["gamma"]["tabular_q_half_life_days"]),
                                )
                            )
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
                                "episodes": episodes if model in RL_MODELS else None,
                                "mode": mode,
                            }
                            fingerprint = _run_fingerprint(digest, run_spec, episodes)
                            out_dir = walkforward_run_dir(
                                root, ticker, model, fold_name, horizon, cost, seed
                            )
                            if _is_complete(out_dir, fingerprint, model in RL_MODELS):
                                completed += 1
                                continue

                            print(
                                f"[{completed + 1}/{expected}] {ticker} {model} {fold_name} "
                                f"horizon={horizon} cost={cost:g} seed={seed}",
                                flush=True,
                            )
                            if model in BASELINE_MODELS:
                                target = _baseline_target(model, test_feat, horizon)
                                result = simulate_positions(
                                    test_feat[["Open", "Close"]],
                                    target,
                                    cost=cost,
                                    initial_position=0.0,
                                )
                                q_summary = None
                            else:
                                agent = _build_agent(model, config, horizon, episodes)
                                agent.train(train_feat, cost=cost, seed=int(seed))
                                result, target = _backtest(
                                    test_feat, agent.rule_fn(), cost, horizon
                                )
                                q_summary = _add_visit_diagnostics(
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
                            run_config = build_run_config(config, config_path, split, run_spec)
                            _save_run(
                                out_dir,
                                run_config,
                                metrics,
                                result,
                                target,
                                fingerprint,
                                q_summary,
                            )
                            completed += 1
                            _update_status(root, mode, completed, expected, run_spec)

    assert_no_30day_outputs(root)
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
    assert_no_30day_outputs(root)
    records = _completion_records(root)
    expected = _expected_run_count(config, smoke=True)
    if len(records) != expected:
        raise RuntimeError(f"Smoke grid has {len(records)} complete runs; expected {expected}")
    for record in records:
        numeric = [value for value in record.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Smoke run contains non-finite metrics: {record['_run_dir']}")

    for ticker in config["tickers"]:
        for fold_name in config["folds"]:
            for horizon in config["reward_horizons_days"]:
                controls = sorted(
                    (
                        float(record["cost"]),
                        float(record["final_equity"]),
                        float(record["total_cost_paid"]),
                    )
                    for record in records
                    if record["ticker"] == ticker
                    and record["fold"] == fold_name
                    and int(record["horizon"]) == horizon
                    and record["model"] == "buy_hold"
                )
                if len(controls) != len(config["cost_grid"]):
                    raise RuntimeError(
                        f"Missing cost controls for {ticker}/{fold_name}/horizon_{horizon}"
                    )
                if len({equity for _, equity, _ in controls}) != len(controls):
                    raise RuntimeError(
                        f"Costs did not change buy-hold output for {ticker}/{fold_name}/horizon_{horizon}"
                    )
                if controls[0][2] != 0.0 or any(paid <= 0.0 for _, _, paid in controls[1:]):
                    raise RuntimeError(
                        f"Transaction-cost control failed for {ticker}/{fold_name}/horizon_{horizon}"
                    )

    _write_json(
        root / "_SMOKE_VALIDATED.json",
        {
            "status": "validated",
            "config_digest": _config_digest(config),
            "completed_runs": len(records),
            "checks": [
                "complete_grid",
                "finite_metrics",
                "cost_sensitive_buy_hold_control",
                "no_30day_outputs",
            ],
            "validated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def require_valid_smoke(config: dict[str, Any]) -> None:
    marker_path = RESULTS_ROOT / "smoke" / "_SMOKE_VALIDATED.json"
    if not marker_path.exists():
        raise RuntimeError("Run the walk-forward --smoke successfully before --formal")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "validated" or marker.get("config_digest") != _config_digest(config):
        raise RuntimeError("Walk-forward smoke marker is stale or does not match the config")


def _load_returns(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "daily_returns.csv", index_col=0, parse_dates=True)
    return frame["daily_return"]


def _seed_values(model: str, config: dict[str, Any]) -> list[int | None]:
    return [None] if model in BASELINE_MODELS else [int(seed) for seed in config["seeds"]]


def _stitched_returns(
    ticker: str,
    model: str,
    horizon: int,
    cost: float,
    seed: int | None,
    included_folds: list[str],
) -> pd.Series:
    parts = [
        _load_returns(
            walkforward_run_dir(
                RESULTS_ROOT, ticker, model, fold_name, horizon, cost, seed
            )
        )
        for fold_name in included_folds
    ]
    stitched = pd.concat(parts)
    if not stitched.index.is_unique or not stitched.index.is_monotonic_increasing:
        raise ValueError(f"Stitched dates overlap or are unordered for {ticker}/{model}/horizon_{horizon}")
    return stitched


def _fold_delta_summary(
    metrics: pd.DataFrame, config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    primary_cost = float(config["primary_cost"])
    selected = metrics[metrics["cost"] == primary_cost]
    rows = []
    for ticker in config["tickers"]:
        for model in ALL_MODELS:
            if model == "buy_hold":
                continue
            for horizon in config["reward_horizons_days"]:
                for fold_name in config["folds"]:
                    model_rows = selected[
                        (selected["ticker"] == ticker)
                        & (selected["model"] == model)
                        & (selected["horizon"] == horizon)
                        & (selected["fold"] == fold_name)
                    ]
                    benchmark_rows = selected[
                        (selected["ticker"] == ticker)
                        & (selected["model"] == "buy_hold")
                        & (selected["horizon"] == horizon)
                        & (selected["fold"] == fold_name)
                    ]
                    if model_rows.empty or len(benchmark_rows) != 1:
                        raise RuntimeError(f"Incomplete fold summary for {ticker}/{model}/{fold_name}")
                    rows.append(
                        {
                            "ticker": ticker,
                            "model": model,
                            "horizon": horizon,
                            "fold": fold_name,
                            "cost": primary_cost,
                            "mean_model_sharpe": float(model_rows["sharpe"].mean()),
                            "buy_hold_sharpe": float(benchmark_rows.iloc[0]["sharpe"]),
                            "mean_delta_sharpe": float(
                                model_rows["sharpe"].mean() - benchmark_rows.iloc[0]["sharpe"]
                            ),
                            "n_seeds": len(model_rows),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(summary_dir / "fold_delta_sharpe.csv", index=False)
    return result


def _stitched_significance(
    config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    primary_cost = float(config["primary_cost"])
    bootstrap_config = BootstrapConfig(**config["bootstrap"])
    folds = list(config["folds"])
    rows = []
    for ticker in config["tickers"]:
        for model in ALL_MODELS:
            if model == "buy_hold":
                continue
            for horizon in config["reward_horizons_days"]:
                benchmark = _stitched_returns(
                    ticker, "buy_hold", horizon, primary_cost, None, folds
                )
                seeds = _seed_values(model, config)
                policies = pd.DataFrame(
                    {
                        str(seed): _stitched_returns(
                            ticker, model, horizon, primary_cost, seed, folds
                        )
                        for seed in seeds
                    }
                )
                if len(seeds) == 1:
                    result = moving_block_bootstrap_sharpe_diff(
                        policies.iloc[:, 0], benchmark, bootstrap_config
                    )
                    mean_delta = float(result["point_estimate"])
                    lower = float(result["ci_lower_95"])
                    upper = float(result["ci_upper_95"])
                    method = "moving_block_bootstrap_sharpe_diff"
                else:
                    result = bootstrap_mean_seed_sharpe(policies, benchmark, bootstrap_config)
                    mean_delta = float(result["mean_delta_sharpe"])
                    lower = float(result["ci_lower_95"])
                    upper = float(result["ci_upper_95"])
                    method = str(result["method"])
                rows.append(
                    {
                        "ticker": ticker,
                        "model": model,
                        "horizon": horizon,
                        "cost": primary_cost,
                        "mean_delta_sharpe": mean_delta,
                        "ci_lower_95": lower,
                        "ci_upper_95": upper,
                        "significantly_beats_buy_hold": lower > TOLERANCE,
                        "n_seeds": len(seeds),
                        "n_folds": len(folds),
                        "n_observations": len(benchmark),
                        "block_length": bootstrap_config.block_length,
                        "n_repetitions": bootstrap_config.n_repetitions,
                        "method": method,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "stitched_significance.csv", index=False)
    return frame


def _leave_one_fold_out(
    config: dict[str, Any], stitched: pd.DataFrame, summary_dir: Path
) -> pd.DataFrame:
    primary_cost = float(config["primary_cost"])
    all_folds = list(config["folds"])
    rows = []
    for full_row in stitched.itertuples(index=False):
        seeds = _seed_values(full_row.model, config)
        for excluded_fold in all_folds:
            included = [fold for fold in all_folds if fold != excluded_fold]
            benchmark = _stitched_returns(
                full_row.ticker, "buy_hold", full_row.horizon, primary_cost, None, included
            )
            seed_deltas = []
            for seed in seeds:
                policy = _stitched_returns(
                    full_row.ticker,
                    full_row.model,
                    full_row.horizon,
                    primary_cost,
                    seed,
                    included,
                )
                seed_deltas.append(sharpe_ratio(policy) - sharpe_ratio(benchmark))
            leave_out_delta = float(np.mean(seed_deltas))
            full_delta = float(full_row.mean_delta_sharpe)
            rows.append(
                {
                    "ticker": full_row.ticker,
                    "model": full_row.model,
                    "horizon": full_row.horizon,
                    "cost": primary_cost,
                    "excluded_fold": excluded_fold,
                    "included_folds": ",".join(included),
                    "full_four_fold_mean_delta_sharpe": full_delta,
                    "leave_one_fold_out_mean_delta_sharpe": leave_out_delta,
                    "change_from_full": leave_out_delta - full_delta,
                    "direction_flip": (
                        (full_delta > TOLERANCE and leave_out_delta <= TOLERANCE)
                        or (full_delta <= TOLERANCE and leave_out_delta > TOLERANCE)
                    ),
                    "n_seeds": len(seeds),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "leave_one_fold_out.csv", index=False)
    return frame


def _main_direction_crosscheck(
    stitched: pd.DataFrame, config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    main_path = EXPERIMENT_DIR / "results" / "summary" / "metrics_long.csv"
    if not main_path.exists():
        raise RuntimeError("Main experiment metrics are required for the direction cross-check")
    main = pd.read_csv(main_path)
    primary_cost = float(config["primary_cost"])
    main = main[(main["cost"] == primary_cost) & (main["horizon"].isin(EXPECTED_HORIZONS))]
    rows = []
    for current in stitched.itertuples(index=False):
        model_rows = main[
            (main["ticker"] == current.ticker)
            & (main["model"] == current.model)
            & (main["horizon"] == current.horizon)
        ]
        benchmark = main[
            (main["ticker"] == current.ticker)
            & (main["model"] == "buy_hold")
            & (main["horizon"] == current.horizon)
        ]
        if model_rows.empty or benchmark.empty:
            raise RuntimeError(f"Missing main result for {current.ticker}/{current.model}/{current.horizon}")
        main_delta = float(model_rows["sharpe"].mean() - benchmark["sharpe"].mean())
        walkforward_delta = float(current.mean_delta_sharpe)
        rows.append(
            {
                "ticker": current.ticker,
                "model": current.model,
                "horizon": current.horizon,
                "main_fixed_window_mean_delta_sharpe": main_delta,
                "walkforward_mean_delta_sharpe": walkforward_delta,
                "same_direction": (main_delta > TOLERANCE) == (walkforward_delta > TOLERANCE),
                "walkforward_significant": bool(current.significantly_beats_buy_hold),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "main_direction_crosscheck.csv", index=False)
    return frame


def _format_generated_results(
    significance: pd.DataFrame, leave_out: pd.DataFrame, crosscheck: pd.DataFrame
) -> str:
    total = len(significance)
    significant = significance[significance["significantly_beats_buy_hold"]]
    count_table = (
        significance.groupby(["model", "horizon"])["significantly_beats_buy_hold"]
        .agg([("significant", "sum"), ("total", "size")])
        .reset_index()
    )
    lines = [
        README_START,
        "### Walk-forward results",
        "",
        f"At the primary 10 bps cost, {len(significant)}/{total} "
        "(ticker, model, horizon) combinations significantly beat same-cost buy-and-hold.",
        "",
        "| model | horizon | significant | total |",
        "|---|---:|---:|---:|",
    ]
    for row in count_table.itertuples(index=False):
        lines.append(f"| {row.model} | {row.horizon} | {int(row.significant)} | {row.total} |")
    lines.extend(["", "Full confidence intervals are in `results/walkforward/summary/stitched_significance.csv`."])

    dependencies = leave_out[leave_out["direction_flip"]]
    lines.extend(["", "#### Leave-one-fold-out diagnosis", ""])
    if dependencies.empty:
        lines.append("No combination changed the sign of mean ΔSharpe after removing one fold.")
    else:
        lines.extend(
            [
                "The following cases changed the sign of mean ΔSharpe when one fold was removed:",
                "",
                "| ticker | model | horizon | excluded fold | full ΔSharpe | leave-one-out ΔSharpe |",
                "|---|---|---:|---|---:|---:|",
            ]
        )
        for row in dependencies.sort_values(["ticker", "model", "horizon", "excluded_fold"]).itertuples():
            lines.append(
                f"| {row.ticker} | {row.model} | {row.horizon} | {row.excluded_fold} | "
                f"{row.full_four_fold_mean_delta_sharpe:.6f} | "
                f"{row.leave_one_fold_out_mean_delta_sharpe:.6f} |"
            )
    lines.extend(
        [
            "",
            "All four exclusion variants are in `results/walkforward/summary/leave_one_fold_out.csv`.",
            "",
            "#### Main-experiment direction cross-check",
            "",
            f"The fixed-window and walk-forward mean ΔSharpe directions agree for "
            f"{int(crosscheck['same_direction'].sum())}/{len(crosscheck)} overlapping combinations. "
            "The parent experiment reported 0/126 significant wins; the walk-forward significance count above "
            "uses the same primary-cost, stitched-OOS seed-mean interpretation where applicable.",
            "",
            "Combination-level comparisons are in `results/walkforward/summary/main_direction_crosscheck.csv`.",
            README_END,
        ]
    )
    return "\n".join(lines)


def _update_readme(
    significance: pd.DataFrame, leave_out: pd.DataFrame, crosscheck: pd.DataFrame
) -> None:
    readme_path = EXPERIMENT_DIR / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    generated = _format_generated_results(significance, leave_out, crosscheck)
    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    readme_path.write_text(text[:start] + generated + text[end:], encoding="utf-8")


def finalize_formal(config: dict[str, Any]) -> None:
    assert_no_30day_outputs(RESULTS_ROOT)
    completion_path = RESULTS_ROOT / "_FORMAL_COMPLETE.json"
    if not completion_path.exists():
        raise RuntimeError("Walk-forward formal completion marker is absent")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("config_digest") != _config_digest(config):
        raise RuntimeError("Walk-forward completion marker does not match the config")
    records = _completion_records(RESULTS_ROOT)
    expected = _expected_run_count(config, smoke=False)
    if len(records) != expected:
        raise RuntimeError(f"Walk-forward set has {len(records)} runs; expected {expected}")

    summary_dir = RESULTS_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "_run_dir"} for record in records]
    )
    metrics["seed"] = metrics["seed"].astype(str)
    metrics.sort_values(
        ["ticker", "model", "fold", "horizon", "cost", "seed"]
    ).to_csv(summary_dir / "metrics_long.csv", index=False)
    _fold_delta_summary(metrics, config, summary_dir)
    significance = _stitched_significance(config, summary_dir)
    leave_out = _leave_one_fold_out(config, significance, summary_dir)
    crosscheck = _main_direction_crosscheck(significance, config, summary_dir)
    _update_readme(significance, leave_out, crosscheck)
    _write_json(
        summary_dir / "_FINALIZED.json",
        {
            "status": "complete",
            "config_digest": _config_digest(config),
            "n_runs": len(records),
            "n_comparisons": len(significance),
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
    config = load_walkforward_config(args.config)

    if args.smoke:
        smoke_root = RESULTS_ROOT / "smoke"
        with _single_instance_lock(smoke_root, "walkforward_smoke"):
            root = _run_grid(config, args.config, smoke=True)
            validate_smoke(config, root)
        print(f"Walk-forward smoke validation passed: {root}")
        return
    if args.finalize_only:
        with _single_instance_lock(RESULTS_ROOT, "walkforward_formal"):
            finalize_formal(config)
        print(f"Walk-forward summaries rebuilt: {RESULTS_ROOT / 'summary'}")
        return

    require_valid_smoke(config)
    with _single_instance_lock(RESULTS_ROOT, "walkforward_formal"):
        _run_grid(config, args.config, smoke=False)
        finalize_formal(config)
    print(f"Walk-forward run and summaries complete: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()