"""Run the three-fold short-recent-window single-stock v2 experiment.

Each fold uses four recent training years followed by one out-of-sample year.
The full train/test block advances by two years between folds. Only the parent's
3-state bandit and four rule baselines are included at horizons 1, 5, and 10.
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
    _config_digest,
    _get_features,
    _is_complete,
    _metrics_payload,
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


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_short_recent_window_v2.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "short_recent_window_v2"
RL_MODELS = ("bandit_3state",)
ALL_MODELS = BASELINE_MODELS + RL_MODELS
FOLD_NAMES = ("fold_01", "fold_02", "fold_03")
EXPECTED_HORIZONS = (1, 5, 10)
EXPECTED_FOLDS = {
    "fold_01": {
        "train_start": "2016-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
    },
    "fold_02": {
        "train_start": "2018-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    "fold_03": {
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
}
TOLERANCE = 1e-10


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["folds"] != EXPECTED_FOLDS:
        raise ValueError(f"folds must be exactly {EXPECTED_FOLDS}")
    if tuple(config["folds"]) != FOLD_NAMES:
        raise ValueError(f"fold order must be {FOLD_NAMES}")
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError(f"reward_horizons_days must be {list(EXPECTED_HORIZONS)}")
    if not config["decision_interval_equals_reward_horizon"]:
        raise ValueError("decision_interval_equals_reward_horizon must be true")
    if set(config["tickers"]) != {"SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL"}:
        raise ValueError("tickers must be exactly the parent experiment's seven tickers")
    if config["models"]["rule_baselines"] != list(BASELINE_MODELS):
        raise ValueError("Configured rule baselines do not match runner model names")
    if list(config["models"]) != ["rule_baselines", "bandit_3state"]:
        raise ValueError("This supplement trains only bandit_3state")
    if config["actions"] != ["TargetCash", "TargetLong"] or not config["no_shorting"]:
        raise ValueError("This experiment requires the two no-shorting target actions")
    if config["z_window_days"] != 20 or config["sma_trend_window_days"] != 200:
        raise ValueError("Shared features require z_window=20 and SMA window=200")
    validate_fold_chronology(config["folds"])
    return config


def validate_fold_chronology(folds: dict[str, dict[str, str]]) -> None:
    tests = []
    for fold_name, split in folds.items():
        train_start = pd.Timestamp(split["train_start"])
        train_end = pd.Timestamp(split["train_end"])
        test_start = pd.Timestamp(split["test_start"])
        test_end = pd.Timestamp(split["test_end"])
        if train_end >= test_start:
            raise ValueError(f"Training reaches the test period in {fold_name}")
        if train_start != test_start - pd.DateOffset(years=4):
            raise ValueError(f"{fold_name} does not use the preceding four calendar years")
        if train_end != test_start - pd.Timedelta(days=1):
            raise ValueError(f"{fold_name} training is not adjacent to testing")
        if test_end != test_start + pd.DateOffset(years=1) - pd.Timedelta(days=1):
            raise ValueError(f"{fold_name} test period is not one calendar year")
        tests.append((test_start, test_end, fold_name))
    for (_, previous_end, previous_name), (current_start, _, current_name) in zip(tests, tests[1:]):
        if current_start <= previous_end:
            raise ValueError(f"Test periods overlap: {previous_name} and {current_name}")


def v2_run_dir(
    root: Path,
    ticker: str,
    model: str,
    fold_name: str,
    horizon: int,
    cost: float,
    seed: int | None,
) -> Path:
    if fold_name not in FOLD_NAMES or horizon not in EXPECTED_HORIZONS:
        raise ValueError("Unsupported fold or horizon")
    seed_name = "seed_NA" if seed is None else f"seed_{seed}"
    return (
        root / ticker.lower() / model / fold_name / f"horizon_{horizon}"
        / f"cost_{cost:g}" / seed_name
    )


def _expected_run_count(config: dict[str, Any], smoke: bool) -> int:
    seeds = 1 if smoke else len(config["seeds"])
    return (
        len(config["tickers"])
        * len(config["folds"])
        * len(config["reward_horizons_days"])
        * len(config["cost_grid"])
        * (len(BASELINE_MODELS) + len(RL_MODELS) * seeds)
    )


def _completion_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for marker_path in root.glob("*/*/fold_*/horizon_*/cost_*/seed_*/completion.json"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        metrics_path = marker_path.parent / "metrics.json"
        if marker.get("status") == "complete" and metrics_path.exists():
            records.append({
                **json.loads(metrics_path.read_text(encoding="utf-8")),
                "_run_dir": marker_path.parent,
            })
    return records


def _run_grid(
    config: dict[str, Any], config_path: Path, smoke: bool,
    shard_index: int = 0, shard_count: int = 1,
) -> Path:
    mode = "smoke" if smoke else "formal"
    root = RESULTS_ROOT / "smoke" if smoke else RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if smoke and shard_count != 1:
        raise ValueError("Smoke validation must run without sharding")
    tickers = [
        ticker for index, ticker in enumerate(config["tickers"])
        if index % shard_count == shard_index
    ]
    if not tickers:
        raise ValueError("This shard has no assigned tickers")
    digest = _config_digest(config)
    episodes = 2 if smoke else int(config["episodes"])
    seeds = [int(config["seeds"][0])] if smoke else [int(seed) for seed in config["seeds"]]
    expected = _expected_run_count(config, smoke) * len(tickers) // len(config["tickers"])
    completed = 0
    started = time.time()

    for ticker in tickers:
        for fold_name, split in config["folds"].items():
            feat = _get_features(ticker, split, config)
            train_feat = _slice_train(feat, split)
            test_feat = _slice_test_with_prior_day(feat, split)
            for horizon in config["reward_horizons_days"]:
                benchmark_target = buy_and_hold_baseline(test_feat, interval=horizon)
                benchmark = simulate_positions(
                    test_feat[["Open", "Close"]], benchmark_target, cost=0.0,
                    initial_position=0.0,
                ).daily_returns
                for model in ALL_MODELS:
                    model_seeds: list[int | None] = [None] if model in BASELINE_MODELS else seeds
                    for seed in model_seeds:
                        for cost in config["cost_grid"]:
                            run_spec = {
                                "ticker": ticker,
                                "fold": fold_name,
                                "model": model,
                                "horizon": horizon,
                                "decision_interval_days": horizon,
                                "reward_horizon_days": horizon,
                                "cost": cost,
                                "seed": seed,
                                "gamma": 0.0,
                                "episodes": episodes if model in RL_MODELS else None,
                                "mode": mode,
                            }
                            fingerprint = _run_fingerprint(digest, run_spec, episodes)
                            out_dir = v2_run_dir(
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
                                    test_feat[["Open", "Close"]], target, cost,
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
                            metrics = _metrics_payload(result, benchmark, run_spec)
                            metrics["fold"] = fold_name
                            run_config = {
                                "experiment_name": config["experiment_name"],
                                "parent_experiment": config["parent_experiment"],
                                "source_config": str(config_path),
                                "split": split,
                                "run": run_spec,
                                "config_digest": digest,
                            }
                            _save_run(
                                out_dir, run_config, metrics, result, target,
                                fingerprint, q_summary,
                            )
                            completed += 1
                            status_name = (
                                "status.json" if shard_count == 1
                                else f"status_shard_{shard_index:02d}.json"
                            )
                            _write_json(root / status_name, {
                                "mode": mode,
                                "shard_index": shard_index,
                                "shard_count": shard_count,
                                "assigned_tickers": tickers,
                                "completed_runs": completed,
                                "expected_runs": expected,
                                "current": run_spec,
                                "updated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                            })

    if completed != expected:
        raise RuntimeError(f"Grid incomplete: found {completed} runs, expected {expected}")
    marker_name = (
        f"_{mode.upper()}_COMPLETE.json" if shard_count == 1
        else f"_{mode.upper()}_SHARD_{shard_index:02d}_COMPLETE.json"
    )
    _write_json(root / marker_name, {
        "status": "complete",
        "config_digest": digest,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_tickers": tickers,
        "completed_runs": completed,
        "expected_runs": expected,
        "episodes": episodes,
        "elapsed_seconds_this_invocation": time.time() - started,
        "completed_at_utc": pd.Timestamp.now("UTC").isoformat(),
    })
    return root


def validate_smoke(config: dict[str, Any], root: Path) -> None:
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
                    (float(row["cost"]), float(row["final_equity"]), float(row["total_cost_paid"]))
                    for row in records
                    if row["ticker"] == ticker and row["fold"] == fold_name
                    and row["horizon"] == horizon and row["model"] == "buy_hold"
                )
                if len(controls) != len(config["cost_grid"]):
                    raise RuntimeError(f"Missing cost controls for {ticker}/{fold_name}/{horizon}")
                if len({equity for _, equity, _ in controls}) != len(controls):
                    raise RuntimeError(f"Costs did not change buy-hold for {ticker}/{fold_name}/{horizon}")
                if controls[0][2] != 0.0 or any(paid <= 0.0 for _, _, paid in controls[1:]):
                    raise RuntimeError(f"Cost control failed for {ticker}/{fold_name}/{horizon}")
    _write_json(root / "_SMOKE_VALIDATED.json", {
        "status": "validated",
        "config_digest": _config_digest(config),
        "completed_runs": len(records),
        "checks": ["complete_grid", "finite_metrics", "cost_sensitive_buy_hold_control"],
        "validated_at_utc": pd.Timestamp.now("UTC").isoformat(),
    })


def require_valid_smoke(config: dict[str, Any]) -> None:
    marker_path = RESULTS_ROOT / "smoke" / "_SMOKE_VALIDATED.json"
    if not marker_path.exists():
        raise RuntimeError("Run v2 --smoke successfully before --formal")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "validated" or marker.get("config_digest") != _config_digest(config):
        raise RuntimeError("V2 smoke marker is stale or does not match the config")


def _load_returns(path: Path) -> pd.Series:
    return pd.read_csv(path / "daily_returns.csv", index_col=0, parse_dates=True)["daily_return"]


def _returns(
    ticker: str, model: str, fold: str, horizon: int, cost: float, seed: int | None
) -> pd.Series:
    return _load_returns(v2_run_dir(RESULTS_ROOT, ticker, model, fold, horizon, cost, seed))


def _bootstrap_row(
    ticker: str, horizon: int, cost: float, folds: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    def oos_returns(model: str, fold: str, seed: int | None) -> pd.Series:
        returns = _returns(ticker, model, fold, horizon, cost, seed)
        return returns.loc[pd.Timestamp(config["folds"][fold]["test_start"]) :]

    benchmark = pd.concat([
        oos_returns("buy_hold", fold, None) for fold in folds
    ])
    policies = pd.DataFrame({
        str(seed): pd.concat([
            oos_returns("bandit_3state", fold, int(seed)) for fold in folds
        ])
        for seed in config["seeds"]
    })
    if not benchmark.index.is_unique or not benchmark.index.is_monotonic_increasing:
        raise ValueError(f"Fold returns overlap or are unordered for {ticker}/horizon_{horizon}")
    result = bootstrap_mean_seed_sharpe(
        policies, benchmark, BootstrapConfig(**config["bootstrap"])
    )
    return {
        "ticker": ticker,
        "model": "bandit_3state",
        "horizon": horizon,
        "cost": cost,
        "folds": ",".join(folds),
        "n_folds": len(folds),
        "mean_delta_sharpe": float(result["mean_delta_sharpe"]),
        "ci_lower_95": float(result["ci_lower_95"]),
        "ci_upper_95": float(result["ci_upper_95"]),
        "significantly_beats_buy_hold": float(result["ci_lower_95"]) > TOLERANCE,
        "n_seeds": len(config["seeds"]),
        "n_observations": len(benchmark),
        "method": str(result["method"]),
    }


def _significance_tables(config: dict[str, Any], summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = float(config["primary_cost"])
    fold_rows = []
    stitched_rows = []
    for ticker in config["tickers"]:
        for horizon in config["reward_horizons_days"]:
            for fold in config["folds"]:
                fold_rows.append(_bootstrap_row(ticker, horizon, cost, [fold], config))
            stitched_rows.append(
                _bootstrap_row(ticker, horizon, cost, list(config["folds"]), config)
            )
    fold_frame = pd.DataFrame(fold_rows)
    stitched_frame = pd.DataFrame(stitched_rows)
    fold_frame.to_csv(summary_dir / "fold_significance.csv", index=False)
    stitched_frame.to_csv(summary_dir / "stitched_significance.csv", index=False)
    return fold_frame, stitched_frame


def _fold_consistency(fold_significance: pd.DataFrame, summary_dir: Path) -> pd.DataFrame:
    rows = []
    for (ticker, horizon), group in fold_significance.groupby(["ticker", "horizon"]):
        deltas = group.sort_values("folds")["mean_delta_sharpe"]
        positive = int((deltas > TOLERANCE).sum())
        rows.append({
            "ticker": ticker,
            "horizon": horizon,
            "positive_folds": positive,
            "negative_or_zero_folds": len(deltas) - positive,
            "all_same_direction": positive in (0, len(deltas)),
            "all_positive": positive == len(deltas),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "fold_direction_consistency.csv", index=False)
    return frame


def _visit_summary(records: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    rows = []
    for record in records:
        if record["model"] != "bandit_3state":
            continue
        visits = pd.read_csv(Path(record["_run_dir"]) / "visit_counts.csv")
        for visit in visits.to_dict("records"):
            rows.append({
                "ticker": record["ticker"], "fold": record["fold"],
                "horizon": record["horizon"], "cost": record["cost"],
                "seed": record["seed"], **visit,
            })
    detailed = pd.DataFrame(rows)
    detailed.to_csv(summary_dir / "visit_counts_long.csv", index=False)
    distribution = (
        detailed.groupby(["fold", "horizon"])["visit_count"]
        .agg(n_cells="size", minimum="min", mean="mean", median="median", maximum="max")
        .reset_index()
    )
    distribution.to_csv(summary_dir / "visit_count_distribution.csv", index=False)
    return distribution


def _write_report(
    config: dict[str, Any], stitched: pd.DataFrame, consistency: pd.DataFrame,
    summary_dir: Path,
) -> None:
    lines = [
        "# Short recent window v2 results",
        "",
        "Three fixed four-year training windows are followed by separate one-year tests.",
        "The stitched result combines only the non-overlapping 2020, 2022, and 2024 OOS years.",
        "",
        "| Ticker | Horizon | Stitched mean delta Sharpe | 95% CI | Significant | Same fold direction |",
        "|---|---:|---:|---|:---:|:---:|",
    ]
    consistency_map = {
        (row.ticker, row.horizon): row.all_same_direction
        for row in consistency.itertuples(index=False)
    }
    for row in stitched.sort_values(["horizon", "ticker"]).itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.horizon} | {row.mean_delta_sharpe:.6f} | "
            f"[{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}] | "
            f"{'yes' if row.significantly_beats_buy_hold else 'no'} | "
            f"{'yes' if consistency_map[(row.ticker, row.horizon)] else 'no'} |"
        )
    lines += [
        "",
        "Fold-level intervals are in `fold_significance.csv`; stitched intervals are in "
        "`stitched_significance.csv`. These intervals are conditional on the trained policies.",
    ]
    (summary_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_formal(config: dict[str, Any]) -> None:
    complete_path = RESULTS_ROOT / "_FORMAL_COMPLETE.json"
    records = _completion_records(RESULTS_ROOT)
    expected = _expected_run_count(config, smoke=False)
    if len(records) != expected:
        raise RuntimeError(f"Formal result set has {len(records)} runs; expected {expected}")
    if complete_path.exists():
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if marker.get("config_digest") != _config_digest(config):
            raise RuntimeError("Formal completion marker does not match the v2 config")
    else:
        _write_json(complete_path, {
            "status": "complete",
            "config_digest": _config_digest(config),
            "completed_runs": len(records),
            "expected_runs": expected,
            "episodes": int(config["episodes"]),
            "assembled_from_shards": True,
            "completed_at_utc": pd.Timestamp.now("UTC").isoformat(),
        })
    summary_dir = RESULTS_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([
        {key: value for key, value in record.items() if key != "_run_dir"}
        for record in records
    ])
    metrics["seed"] = metrics["seed"].astype(str)
    metrics.sort_values(
        ["ticker", "model", "fold", "horizon", "cost", "seed"]
    ).to_csv(summary_dir / "metrics_long.csv", index=False)
    _visit_summary(records, summary_dir)
    fold_significance, stitched = _significance_tables(config, summary_dir)
    consistency = _fold_consistency(fold_significance, summary_dir)
    _write_report(config, stitched, consistency, summary_dir)
    _write_json(summary_dir / "_FINALIZED.json", {
        "status": "complete",
        "config_digest": _config_digest(config),
        "n_runs": len(records),
        "n_fold_significance_rows": len(fold_significance),
        "n_stitched_significance_rows": len(stitched),
        "finalized_at_utc": pd.Timestamp.now("UTC").isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.smoke:
        smoke_root = RESULTS_ROOT / "smoke"
        with _single_instance_lock(smoke_root, "short_recent_window_v2_smoke"):
            root = _run_grid(config, args.config, smoke=True)
            validate_smoke(config, root)
        print(f"V2 smoke validation passed: {root}")
        return
    if args.finalize_only:
        with _single_instance_lock(RESULTS_ROOT, "short_recent_window_v2_formal"):
            finalize_formal(config)
        print(f"V2 formal summaries rebuilt: {RESULTS_ROOT / 'summary'}")
        return
    require_valid_smoke(config)
    lock_name = (
        "short_recent_window_v2_formal" if args.shard_count == 1
        else f"short_recent_window_v2_formal_shard_{args.shard_index:02d}"
    )
    with _single_instance_lock(RESULTS_ROOT, lock_name):
        _run_grid(
            config, args.config, smoke=False,
            shard_index=args.shard_index, shard_count=args.shard_count,
        )
        if args.shard_count == 1:
            finalize_formal(config)
    if args.shard_count == 1:
        print(f"V2 formal run and summaries complete: {RESULTS_ROOT}")
    else:
        print(
            f"V2 formal shard {args.shard_index}/{args.shard_count} complete; "
            "run --finalize-only after every shard finishes."
        )


if __name__ == "__main__":
    main()