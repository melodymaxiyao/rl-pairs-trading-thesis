"""Run the short, recent-window x short-horizon supplement to the single-stock experiment.

Tests whether a training window that is short and immediately adjacent to the test
period (4 years, 2018-2021) recovers a signal that the parent experiment's long,
early-starting window (14 years, from 2008) could not, when paired with a much
higher decision frequency (horizon down to 1 trading day). Only the 3-state bandit
is trained here (a 4-year window cannot support the 6-state or 12-state tables at
any of the tested horizons at an acceptable visit-count floor); the parent's four
rule baselines are re-run as the same-window comparison group. Does not modify the
parent experiment's or the walk-forward supplement's code, configs, or results.
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

from experiments.single_stock.run_single_stock import (
    BASELINE_MODELS,
    _add_visit_diagnostics,
    _backtest,
    _baseline_target,
    _build_agent,
    _completion_records,
    _config_digest,
    _get_features,
    _is_complete,
    _metrics_payload,
    _run_dir,
    _run_fingerprint,
    _save_run,
    _single_instance_lock,
    _slice_test_with_prior_day,
    _slice_train,
    _write_json,
)
from fyp_rl.audit_statistics import bootstrap_mean_seed_sharpe
from fyp_rl.baselines import buy_and_hold_baseline
from fyp_rl.config import BootstrapConfig
from fyp_rl.execution import simulate_positions


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_short_recent_window.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "short_recent_window"
RL_MODELS = ("bandit_3state",)
ALL_MODELS = BASELINE_MODELS + RL_MODELS
EXPECTED_HORIZONS = (1, 5, 10)
EXPECTED_SPLIT = {
    "train_start": "2018-01-01",
    "train_end": "2021-12-31",
    "test_start": "2022-01-01",
    "test_end": "2025-12-31",
}
README_START = "<!-- SHORT_RECENT_WINDOW_RESULTS_START -->"
README_END = "<!-- SHORT_RECENT_WINDOW_RESULTS_END -->"
TOLERANCE = 1e-10


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError(f"reward_horizons_days must be {list(EXPECTED_HORIZONS)}")
    if not config["decision_interval_equals_reward_horizon"]:
        raise ValueError("decision_interval_equals_reward_horizon must be true")
    if config["split"] != EXPECTED_SPLIT:
        raise ValueError(f"split must be exactly {EXPECTED_SPLIT}")
    if set(config["tickers"]) != {"SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL"}:
        raise ValueError("tickers must be exactly the parent experiment's seven tickers")
    if config["models"]["rule_baselines"] != list(BASELINE_MODELS):
        raise ValueError("Configured rule baselines do not match runner model names")
    if list(config["models"].keys()) != ["rule_baselines", "bandit_3state"]:
        raise ValueError("This supplement trains only bandit_3state; no other learned model is permitted")
    if config["actions"] != ["TargetCash", "TargetLong"] or not config["no_shorting"]:
        raise ValueError("This experiment requires the two no-shorting target actions")
    if config["z_window_days"] != 20 or config["sma_trend_window_days"] != 200:
        raise ValueError("Shared build_feature_frame currently requires z_window=20 and SMA window=200")
    expected_visits = config["diagnostics"]["expected_avg_visits_per_cell"]
    if set(expected_visits) != {"1", "5", "10"}:
        raise ValueError("expected_avg_visits_per_cell must have entries for horizons 1, 5, and 10")
    return config


def _expected_run_count(config: dict[str, Any], smoke: bool) -> int:
    n_ticker_horizons = len(config["tickers"]) * len(config["reward_horizons_days"])
    n_seeds = 1 if smoke else len(config["seeds"])
    return n_ticker_horizons * len(config["cost_grid"]) * (
        len(BASELINE_MODELS) + len(RL_MODELS) * n_seeds
    )


def _seed_values(model: str, config: dict[str, Any]) -> list[int | None]:
    return [None] if model in BASELINE_MODELS else [int(seed) for seed in config["seeds"]]


def _run_grid(config: dict[str, Any], config_path: Path, smoke: bool) -> Path:
    mode = "smoke" if smoke else "formal"
    root = RESULTS_ROOT / "smoke" if smoke else RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    config_digest = _config_digest(config)
    episodes = 2 if smoke else int(config["episodes"])
    seeds = [int(config["seeds"][0])] if smoke else [int(seed) for seed in config["seeds"]]
    expected = _expected_run_count(config, smoke)
    completed = 0
    started = time.time()
    split = config["split"]

    for ticker in config["tickers"]:
        feat = _get_features(ticker, split, config)
        train_feat = _slice_train(feat, split)
        test_feat = _slice_test_with_prior_day(feat, split)
        for horizon in config["reward_horizons_days"]:
            benchmark_target = buy_and_hold_baseline(test_feat, interval=horizon)
            benchmark = simulate_positions(
                test_feat[["Open", "Close"]], benchmark_target, cost=0.0, initial_position=0.0
            ).daily_returns
            for model in ALL_MODELS:
                model_seeds: list[int | None] = [None] if model in BASELINE_MODELS else seeds
                for seed in model_seeds:
                    for cost in config["cost_grid"]:
                        gamma = 0.0
                        run_spec = {
                            "ticker": ticker,
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
                        fingerprint = _run_fingerprint(config_digest, run_spec, episodes)
                        out_dir = _run_dir(root, ticker, model, horizon, cost, seed)
                        if _is_complete(out_dir, fingerprint, model in RL_MODELS):
                            completed += 1
                            continue

                        print(
                            f"[{completed + 1}/{expected}] {ticker} {model} "
                            f"horizon={horizon} cost={cost:g} seed={seed}",
                            flush=True,
                        )
                        if model in BASELINE_MODELS:
                            target = _baseline_target(model, test_feat, horizon)
                            result = simulate_positions(
                                test_feat[["Open", "Close"]], target, cost, initial_position=0.0
                            )
                            q_summary = None
                        else:
                            agent = _build_agent(model, config, horizon, episodes)
                            agent.train(train_feat, cost=cost, seed=int(seed))
                            result, target = _backtest(test_feat, agent.rule_fn(), cost, horizon)
                            threshold = int(config["diagnostics"]["flag_low_sample_cells_below"])
                            q_summary = _add_visit_diagnostics(agent.summary(), threshold)

                        metrics = _metrics_payload(result, benchmark, run_spec)
                        run_config = {
                            "experiment_name": config["experiment_name"],
                            "parent_experiment": config["parent_experiment"],
                            "source_config": str(config_path),
                            "split": split,
                            "run": run_spec,
                            "config_digest": config_digest,
                        }
                        _save_run(out_dir, run_config, metrics, result, target, fingerprint, q_summary)
                        completed += 1
                        _write_json(
                            root / "status.json",
                            {
                                "mode": mode,
                                "completed_runs": completed,
                                "expected_runs": expected,
                                "current": run_spec,
                                "updated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                            },
                        )

    if completed != expected:
        raise RuntimeError(f"Grid incomplete: found {completed} completed runs, expected {expected}")
    _write_json(
        root / f"_{mode.upper()}_COMPLETE.json",
        {
            "status": "complete",
            "config_digest": config_digest,
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
    for record in records:
        if not all(np.isfinite(value) for value in record.values() if isinstance(value, (int, float))):
            raise RuntimeError(f"Smoke run contains a non-finite metric: {record['_run_dir']}")

    for ticker in config["tickers"]:
        for horizon in config["reward_horizons_days"]:
            controls = sorted(
                (
                    float(record["cost"]),
                    float(record["final_equity"]),
                    float(record["total_cost_paid"]),
                )
                for record in records
                if record["ticker"] == ticker
                and record["horizon"] == horizon
                and record["model"] == "buy_hold"
            )
            if len(controls) != len(config["cost_grid"]):
                raise RuntimeError(f"Missing smoke cost controls for {ticker}, horizon={horizon}")
            equities = [item[1] for item in controls]
            if len(set(equities)) != len(equities):
                raise RuntimeError(f"Costs did not produce distinct buy-hold outputs for {ticker}, horizon={horizon}")
            if controls[0][2] != 0.0 or any(item[2] <= 0.0 for item in controls[1:]):
                raise RuntimeError(f"Transaction-cost control failed for {ticker}, horizon={horizon}")

    _write_json(
        root / "_SMOKE_VALIDATED.json",
        {
            "status": "validated",
            "config_digest": _config_digest(config),
            "completed_runs": len(records),
            "checks": ["complete_grid", "finite_metrics", "cost_sensitive_buy_hold_control"],
            "validated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def require_valid_smoke(config: dict[str, Any]) -> None:
    marker_path = RESULTS_ROOT / "smoke" / "_SMOKE_VALIDATED.json"
    if not marker_path.exists():
        raise RuntimeError("Run --smoke successfully before --formal")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "validated" or marker.get("config_digest") != _config_digest(config):
        raise RuntimeError("Smoke marker is stale or does not match the current config")


def _load_daily_returns(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "daily_returns.csv", index_col=0, parse_dates=True)
    return frame["daily_return"]


def _visit_summaries(
    records: list[dict[str, Any]], summary_dir: Path, config: dict[str, Any]
) -> pd.DataFrame:
    low_sample_threshold = int(config["diagnostics"]["flag_low_sample_cells_below"])
    expected_visits = config["diagnostics"]["expected_avg_visits_per_cell"]
    rows = []
    for record in records:
        if record["model"] not in RL_MODELS:
            continue
        visits = pd.read_csv(Path(record["_run_dir"]) / "visit_counts.csv")
        for row in visits.to_dict("records"):
            rows.append(
                {
                    "ticker": record["ticker"],
                    "state_space": record["model"],
                    "horizon": record["horizon"],
                    "cost": record["cost"],
                    "seed": record["seed"],
                    **row,
                }
            )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(summary_dir / "visit_counts_long.csv", index=False)
    grouped = detailed.groupby(["state_space", "horizon"])["visit_count"]
    distribution = grouped.agg(n_cells="size", minimum="min", mean="mean", median="median", maximum="max").reset_index()
    distribution["p25"] = grouped.quantile(0.25).to_numpy()
    distribution["p75"] = grouped.quantile(0.75).to_numpy()
    low_counts = detailed.assign(low=detailed["visit_count"] < low_sample_threshold).groupby(
        ["state_space", "horizon"]
    )["low"].sum()
    distribution["cells_below_threshold"] = low_counts.to_numpy()
    distribution["low_sample_threshold"] = low_sample_threshold
    episodes = int(config["episodes"])
    distribution["expected_single_pass_visits"] = distribution["horizon"].astype(str).map(expected_visits).astype(float)
    distribution["expected_cumulative_visits"] = distribution["expected_single_pass_visits"] * episodes
    distribution["mean_vs_expected_cumulative_ratio"] = distribution["mean"] / distribution["expected_cumulative_visits"]
    distribution.to_csv(summary_dir / "visit_count_distribution.csv", index=False)
    return distribution


def _significance_table(
    records: list[dict[str, Any]], config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    """Bootstrap ΔSharpe vs buy-and-hold, per ticker x horizon, at the primary cost.

    Aggregates over the 10 trained seeds with the same seed-block bootstrap used by
    the walk-forward supplement, so the statistic is directly comparable across both
    supplements: mean_seed(Sharpe(policy_seed) - Sharpe(buy_hold)).
    """
    primary_cost = float(config["primary_cost"])
    bootstrap_config = BootstrapConfig(**config["bootstrap"])
    selected = [record for record in records if float(record["cost"]) == primary_cost]
    rows = []
    for ticker in config["tickers"]:
        for horizon in config["reward_horizons_days"]:
            benchmark_record = next(
                record
                for record in selected
                if record["ticker"] == ticker
                and record["horizon"] == horizon
                and record["model"] == "buy_hold"
            )
            benchmark = _load_daily_returns(Path(benchmark_record["_run_dir"]))
            policy_records = [
                record
                for record in selected
                if record["ticker"] == ticker
                and record["horizon"] == horizon
                and record["model"] == "bandit_3state"
            ]
            policies = pd.DataFrame(
                {
                    str(record["seed"]): _load_daily_returns(Path(record["_run_dir"]))
                    for record in policy_records
                }
            )
            result = bootstrap_mean_seed_sharpe(policies, benchmark, bootstrap_config)
            rows.append(
                {
                    "ticker": ticker,
                    "model": "bandit_3state",
                    "horizon": horizon,
                    "cost": primary_cost,
                    "mean_delta_sharpe": float(result["mean_delta_sharpe"]),
                    "ci_lower_95": float(result["ci_lower_95"]),
                    "ci_upper_95": float(result["ci_upper_95"]),
                    "significantly_beats_buy_hold": float(result["ci_lower_95"]) > TOLERANCE,
                    "n_seeds": len(policy_records),
                    "n_observations": len(benchmark),
                    "method": str(result["method"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "significance_by_horizon.csv", index=False)
    return frame


def _crosswindow_comparison(config: dict[str, Any], summary_dir: Path) -> pd.DataFrame:
    """Side-by-side bandit_3state vs buy_hold at horizon=5: this 4-year window vs the
    parent experiment's 14-year window, using the identical bootstrap statistic for both.
    """
    shared_horizon = int(config["crosswindow_comparison"]["shared_horizon_days"])
    primary_cost = float(config["primary_cost"])
    bootstrap_config = BootstrapConfig(**config["bootstrap"])
    parent_root = EXPERIMENT_DIR / "results"
    rows = []
    for ticker in config["tickers"]:
        short_policy_dirs = {
            seed: _run_dir(RESULTS_ROOT, ticker, "bandit_3state", shared_horizon, primary_cost, seed)
            for seed in config["seeds"]
        }
        short_benchmark_dir = _run_dir(RESULTS_ROOT, ticker, "buy_hold", shared_horizon, primary_cost, None)
        short_policies = pd.DataFrame(
            {str(seed): _load_daily_returns(path) for seed, path in short_policy_dirs.items()}
        )
        short_benchmark = _load_daily_returns(short_benchmark_dir)
        short_result = bootstrap_mean_seed_sharpe(short_policies, short_benchmark, bootstrap_config)

        long_policy_dirs = {
            seed: _run_dir(parent_root, ticker, "bandit_3state", shared_horizon, primary_cost, seed)
            for seed in config["seeds"]
        }
        long_benchmark_dir = _run_dir(parent_root, ticker, "buy_hold", shared_horizon, primary_cost, None)
        if not long_benchmark_dir.exists():
            raise RuntimeError(
                f"Parent experiment results for {ticker} horizon={shared_horizon} are required "
                "for the cross-window comparison; run the parent experiment first."
            )
        long_policies = pd.DataFrame(
            {str(seed): _load_daily_returns(path) for seed, path in long_policy_dirs.items()}
        )
        long_benchmark = _load_daily_returns(long_benchmark_dir)
        long_result = bootstrap_mean_seed_sharpe(long_policies, long_benchmark, bootstrap_config)

        short_delta = float(short_result["mean_delta_sharpe"])
        long_delta = float(long_result["mean_delta_sharpe"])
        rows.append(
            {
                "ticker": ticker,
                "horizon": shared_horizon,
                "cost": primary_cost,
                "short_window_years": 4,
                "short_mean_delta_sharpe": short_delta,
                "short_ci_lower_95": float(short_result["ci_lower_95"]),
                "short_ci_upper_95": float(short_result["ci_upper_95"]),
                "short_significant": float(short_result["ci_lower_95"]) > TOLERANCE,
                "long_window_years": 14,
                "long_mean_delta_sharpe": long_delta,
                "long_ci_lower_95": float(long_result["ci_lower_95"]),
                "long_ci_upper_95": float(long_result["ci_upper_95"]),
                "long_significant": float(long_result["ci_lower_95"]) > TOLERANCE,
                "same_direction": (short_delta > TOLERANCE) == (long_delta > TOLERANCE),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "crosswindow_comparison_h5.csv", index=False)
    return frame


def _format_visit_table(distribution: pd.DataFrame) -> str:
    lines = [
        "| horizon | cells | min | p25 | median | mean | p75 | max | cells < threshold | "
        "expected single-pass mean | expected cumulative mean (x episodes) | mean / expected cumulative |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.sort_values("horizon").itertuples():
        lines.append(
            f"| {row.horizon} | {row.n_cells} | {row.minimum:.0f} | "
            f"{row.p25:.1f} | {row.median:.1f} | {row.mean:.1f} | {row.p75:.1f} | "
            f"{row.maximum:.0f} | {row.cells_below_threshold} | {row.expected_single_pass_visits:.0f} | "
            f"{row.expected_cumulative_visits:.0f} | {row.mean_vs_expected_cumulative_ratio:.2f} |"
        )
    return "\n".join(lines)


def _format_significance_table(significance: pd.DataFrame) -> str:
    lines = [
        "| ticker | horizon | mean ΔSharpe | 95% CI | significant |",
        "|---|---:|---:|---|:---:|",
    ]
    for row in significance.sort_values(["horizon", "ticker"]).itertuples():
        lines.append(
            f"| {row.ticker} | {row.horizon} | {row.mean_delta_sharpe:.6f} | "
            f"[{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}] | "
            f"{'yes' if row.significantly_beats_buy_hold else 'no'} |"
        )
    return "\n".join(lines)


def _format_crosswindow_table(comparison: pd.DataFrame) -> str:
    lines = [
        "| ticker | 4-yr ΔSharpe | 4-yr 95% CI | 4-yr sig. | 14-yr ΔSharpe | 14-yr 95% CI | 14-yr sig. | same direction |",
        "|---|---:|---|:---:|---:|---|:---:|:---:|",
    ]
    for row in comparison.sort_values("ticker").itertuples():
        lines.append(
            f"| {row.ticker} | {row.short_mean_delta_sharpe:.6f} | "
            f"[{row.short_ci_lower_95:.6f}, {row.short_ci_upper_95:.6f}] | "
            f"{'yes' if row.short_significant else 'no'} | "
            f"{row.long_mean_delta_sharpe:.6f} | "
            f"[{row.long_ci_lower_95:.6f}, {row.long_ci_upper_95:.6f}] | "
            f"{'yes' if row.long_significant else 'no'} | "
            f"{'yes' if row.same_direction else 'no'} |"
        )
    return "\n".join(lines)


def _format_generated_results(
    significance: pd.DataFrame, comparison: pd.DataFrame, distribution: pd.DataFrame, config: dict[str, Any]
) -> str:
    total = len(significance)
    significant = significance[significance["significantly_beats_buy_hold"]]
    h1 = significance[significance["horizon"] == 1]
    h1_significant = h1[h1["significantly_beats_buy_hold"]]
    direction_changed = comparison[~comparison["same_direction"]]

    lines = [
        README_START,
        "### Results",
        "",
        "#### Significance by horizon (21 combinations: 7 tickers x horizons 1/5/10)",
        "",
        f"At the primary 10 bps cost, {len(significant)}/{total} (ticker, horizon) combinations "
        "significantly beat same-cost buy-and-hold with `bandit_3state`.",
        "",
        _format_significance_table(significance),
        "",
        "Full intervals are in `results/short_recent_window/summary/significance_by_horizon.csv`.",
        "",
        "#### horizon = 1 in context",
        "",
        f"At horizon = 1 day, {len(h1_significant)}/{len(h1)} tickers significantly beat buy-and-hold. "
        "This is read against the prior evidence noted in the config "
        "(`prior_evidence_note`): 第8轮 already scanned horizon in {3,5,10,15,20,30} days under a long "
        "training window and found no significant S0-S2 signal at horizon <= 20 days, and horizon = 1 "
        "here is shorter than anything tested there. The question this sub-experiment answers is not "
        "whether horizon = 1 itself carries a signal (that recurrence is expected), but whether shortening "
        "and freshening the training window from 14 years to 4 years overturns that known no-signal "
        "conclusion. It is answered by the row above and by the cross-window table below, not treated as "
        "an independent new finding.",
        "",
        "#### Cross-window comparison at horizon = 5 (this 4-year window vs the parent 14-year window)",
        "",
        _format_crosswindow_table(comparison),
        "",
    ]
    if direction_changed.empty:
        lines.append(
            "Shortening the training window from 14 years to 4 years did not flip the sign of mean "
            "ΔSharpe for any ticker at horizon = 5."
        )
    else:
        flipped = ", ".join(sorted(direction_changed["ticker"]))
        lines.append(
            f"Shortening the training window from 14 years to 4 years flipped the sign of mean ΔSharpe "
            f"at horizon = 5 for: {flipped}. See `results/short_recent_window/summary/crosswindow_comparison_h5.csv` "
            "for the full detail; a sign flip alone is not evidence of a tradeable edge unless the shorter "
            "window's own CI excludes zero, which the significance table above already reports."
        )
    anomalies = distribution[
        (distribution["mean_vs_expected_cumulative_ratio"] < 0.5)
        | (distribution["mean_vs_expected_cumulative_ratio"] > 2.0)
    ]
    if anomalies.empty:
        ratio_note = (
            "Realized mean visit counts fall within 0.5x-2x of the expected cumulative figure "
            "at every horizon, so the pre-registered feasibility floor holds as designed."
        )
    else:
        anomaly_horizons = ", ".join(str(int(h)) for h in sorted(anomalies["horizon"]))
        ratio_note = (
            f"Horizon(s) {anomaly_horizons} fall outside 0.5x-2x of the expected cumulative figure; "
            "see `results/short_recent_window/summary/visit_count_distribution.csv` for the per-horizon "
            "numbers before treating any horizon-level result as reliable."
        )
    lines.extend(
        [
            "",
            "Full detail is in `results/short_recent_window/summary/crosswindow_comparison_h5.csv`.",
            "",
            "#### Visit-count diagnostics",
            "",
            "`diagnostics.expected_avg_visits_per_cell` in the config (168/34/17 for horizons 1/5/10) is a "
            "single-pass feasibility floor: the number of decision points in one training pass through the "
            "4-year window, divided across the 6 (state, action) cells. It was computed to rule out infeasible "
            "designs (2-year training, 6-state tables) before training; it does not by itself predict the final "
            "visit count. The agent accumulates visit counts across all 100 training episodes without resetting "
            "between episodes, so the realized mean is expected to be roughly 100x the single-pass floor — a "
            "wide margin of safety, not a discrepancy. The table below reports both the single-pass floor and "
            "that floor scaled by 100 episodes for direct comparison.",
            "",
            _format_visit_table(distribution),
            "",
            ratio_note,
            "",
            "Full cell-level counts are in `results/short_recent_window/summary/visit_counts_long.csv`.",
            README_END,
        ]
    )
    return "\n".join(lines)


def _update_readme(significance: pd.DataFrame, comparison: pd.DataFrame, distribution: pd.DataFrame, config: dict[str, Any]) -> None:
    readme_path = EXPERIMENT_DIR / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    generated = _format_generated_results(significance, comparison, distribution, config)
    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    readme_path.write_text(text[:start] + generated + text[end:], encoding="utf-8")


def finalize_formal(config: dict[str, Any]) -> None:
    complete_path = RESULTS_ROOT / "_FORMAL_COMPLETE.json"
    if not complete_path.exists():
        raise RuntimeError("Formal completion marker is absent; refusing to report partial results")
    marker = json.loads(complete_path.read_text(encoding="utf-8"))
    if marker.get("config_digest") != _config_digest(config):
        raise RuntimeError("Formal completion marker does not match the current config")
    records = _completion_records(RESULTS_ROOT)
    expected = _expected_run_count(config, smoke=False)
    if len(records) != expected:
        raise RuntimeError(f"Formal result set has {len(records)} runs; expected {expected}")

    summary_dir = RESULTS_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = [{key: value for key, value in record.items() if key != "_run_dir"} for record in records]
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame["seed"] = metrics_frame["seed"].astype(str)
    metrics_frame.sort_values(
        ["ticker", "model", "horizon", "cost", "seed"]
    ).to_csv(summary_dir / "metrics_long.csv", index=False)

    distribution = _visit_summaries(records, summary_dir, config)
    significance = _significance_table(records, config, summary_dir)
    comparison = _crosswindow_comparison(config, summary_dir)
    _update_readme(significance, comparison, distribution, config)
    _write_json(
        summary_dir / "_FINALIZED.json",
        {
            "status": "complete",
            "config_digest": _config_digest(config),
            "n_runs": len(records),
            "n_significance_rows": len(significance),
            "n_crosswindow_rows": len(comparison),
            "finalized_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the complete grid with two RL episodes and one seed")
    mode.add_argument("--formal", action="store_true", help="Run or resume the complete 100-episode grid")
    mode.add_argument("--finalize-only", action="store_true", help="Rebuild summaries from a complete formal grid")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        smoke_root = RESULTS_ROOT / "smoke"
        with _single_instance_lock(smoke_root, "short_recent_window_smoke"):
            root = _run_grid(config, args.config, smoke=True)
            validate_smoke(config, root)
        print(f"Smoke validation passed: {root}")
        return
    if args.finalize_only:
        with _single_instance_lock(RESULTS_ROOT, "short_recent_window_formal"):
            finalize_formal(config)
        print(f"Formal summaries rebuilt: {RESULTS_ROOT / 'summary'}")
        return

    require_valid_smoke(config)
    with _single_instance_lock(RESULTS_ROOT, "short_recent_window_formal"):
        _run_grid(config, args.config, smoke=False)
        finalize_formal(config)
    print(f"Formal run and summaries complete: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
