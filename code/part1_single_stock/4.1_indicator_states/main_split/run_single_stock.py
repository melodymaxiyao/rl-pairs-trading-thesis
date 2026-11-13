"""Run the single-stock state-complexity x decision-horizon experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.single_stock.agents_6state import SixStateTabularQAgent
from fyp_rl.agents.tabular_q import TabularQAgent
from fyp_rl.baselines import (
    buy_and_hold_baseline,
    cash_baseline,
    sma200_rule_baseline,
    zscore_rule_baseline,
)
from fyp_rl.config import BootstrapConfig, TabularQConfig
from fyp_rl.data import load_ohlcv
from fyp_rl.execution import simulate_positions, single_step_execution
from fyp_rl.features import bandit_state, build_feature_frame, tabular_state
from fyp_rl.metrics import full_metrics
from fyp_rl.statistics import moving_block_bootstrap_sharpe_diff


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results"
BASELINE_MODELS = ("cash", "buy_hold", "zscore_band", "sma200_trend")
RL_MODELS = ("bandit_3state", "tabular_q_6state", "tabular_q_12state")
ALL_MODELS = BASELINE_MODELS + RL_MODELS
RESULTS_START = "<!-- GENERATED_RESULTS_START -->"
RESULTS_END = "<!-- GENERATED_RESULTS_END -->"


def resolve_gamma(decision_interval_days: int, half_life_days: int = 60) -> float:
    """Resolve the semi-MDP discount, rounded to the reported four decimals."""
    if decision_interval_days <= 0 or half_life_days <= 0:
        raise ValueError("decision_interval_days and half_life_days must be positive")
    return round(0.5 ** (decision_interval_days / half_life_days), 4)


def twelve_state_from_bucket(z_bucket_value: int, trend: int, position: int) -> int:
    """Expose the established 12-state encoding in discrete-component form."""
    if z_bucket_value not in (0, 1, 2):
        raise ValueError("z_bucket_value must be 0, 1, or 2")
    if trend not in (0, 1) or position not in (0, 1):
        raise ValueError("trend and position must be 0 or 1")
    return z_bucket_value * 4 + trend * 2 + position


def decision_state(feat: pd.DataFrame, loc: int, model: str, position: int) -> int:
    """Build a state only from the feature row available at decision close."""
    row = feat.iloc[loc]
    if model == "bandit_3state":
        return bandit_state(float(row["z20"]))
    if model == "tabular_q_6state":
        from experiments.single_stock.agents_6state import six_state

        return six_state(float(row["z20"]), position)
    if model == "tabular_q_12state":
        return tabular_state(float(row["z20"]), float(row["trend"]), position)
    raise ValueError(f"Unknown state model: {model}")


class ThreeStateBanditAgent:
    """Gamma-zero signal diagnostic with the experiment-wide execution reward."""

    def __init__(self, horizon: int, episodes: int, epsilon_start: float, epsilon_end: float):
        self.horizon = horizon
        self.episodes = episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.q = np.zeros((3, 2), dtype=np.float64)
        self.counts = np.zeros((3, 2), dtype=np.int64)

    def train(self, feat_train: pd.DataFrame, cost: float, seed: int) -> None:
        rng = random.Random(seed)
        decision_locs = list(range(0, len(feat_train) - self.horizon, self.horizon))
        valid_locs = [loc for loc in decision_locs if not pd.isna(feat_train["z20"].iloc[loc])]
        if not valid_locs:
            raise ValueError("No valid decision points for three-state bandit training")

        for episode in range(self.episodes):
            fraction = episode / max(1, self.episodes - 1)
            epsilon = self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** fraction
            position = 0
            for loc in valid_locs:
                state = decision_state(feat_train, loc, "bandit_3state", position)
                if rng.random() < epsilon:
                    action = rng.randint(0, 1)
                else:
                    action = int(np.argmax(self.q[state]))
                reward, _, _ = single_step_execution(
                    feat_train,
                    loc,
                    loc + self.horizon,
                    float(position),
                    float(action),
                    cost,
                )
                self.counts[state, action] += 1
                alpha = 1.0 / self.counts[state, action]
                self.q[state, action] += alpha * (reward - self.q[state, action])
                position = action

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def rule_fn(self):
        def rule(row: pd.Series, previous_position: float) -> float:
            if pd.isna(row["z20"]):
                return previous_position
            return float(self.greedy_action(bandit_state(float(row["z20"]))))

        return rule

    def summary(self) -> dict[str, Any]:
        labels = {0: "oversold", 1: "neutral", 2: "overbought"}
        return {
            "q_table": self.q.tolist(),
            "visit_counts": self.counts.tolist(),
            "state_labels": labels,
            "greedy_policy": {state: self.greedy_action(state) for state in range(3)},
        }


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(_json_text(value))
            temporary.flush()
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def _single_instance_lock(root: Path, mode: str):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{mode}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another {mode} run is already active. Do not start duplicate runners."
            ) from exc
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected_horizons = [5, 10, 30]
    if config["reward_horizons_days"] != expected_horizons:
        raise ValueError(f"reward_horizons_days must be {expected_horizons}")
    if not config["decision_interval_equals_reward_horizon"]:
        raise ValueError("decision_interval_equals_reward_horizon must be true")
    half_life = int(config["gamma"]["tabular_q_half_life_days"])
    resolved = {str(horizon): resolve_gamma(horizon, half_life) for horizon in expected_horizons}
    if config["gamma"]["resolved_by_horizon"] != resolved:
        raise ValueError(f"Configured gamma values do not match formula: {resolved}")
    if config["z_window_days"] != 20 or config["sma_trend_window_days"] != 200:
        raise ValueError("Shared build_feature_frame currently requires z_window=20 and SMA window=200")
    if config["models"]["rule_baselines"] != list(BASELINE_MODELS):
        raise ValueError("Configured rule baselines do not match runner model names")
    if config["actions"] != ["TargetCash", "TargetLong"] or not config["no_shorting"]:
        raise ValueError("This experiment requires the two no-shorting target actions")
    return config


def _get_features(ticker: str, split: dict[str, str], config: dict[str, Any]) -> pd.DataFrame:
    warmup_start = (
        pd.Timestamp(split["train_start"])
        - pd.Timedelta(days=int(config["feature_warmup_calendar_days"]))
    ).strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(ticker, warmup_start, split["test_end"])
    return build_feature_frame(ohlcv)


def _slice_train(feat: pd.DataFrame, split: dict[str, str]) -> pd.DataFrame:
    train = feat.loc[split["train_start"] : split["train_end"]]
    return train.dropna(subset=["z20", "trend", "sma200"])


def _slice_test_with_prior_day(feat: pd.DataFrame, split: dict[str, str]) -> pd.DataFrame:
    test_start = pd.Timestamp(split["test_start"])
    prior_days = feat.index[feat.index < test_start]
    if len(prior_days) == 0:
        raise ValueError("No pre-test trading day is available for the first decision")
    test = feat.loc[prior_days[-1] : split["test_end"]]
    if len(test) < 2:
        raise ValueError("Test slice contains no executable test day")
    return test


def _backtest(test_feat: pd.DataFrame, rule_fn, cost: float, horizon: int):
    from fyp_rl.baselines import build_decision_series

    target = build_decision_series(test_feat, rule_fn, interval=horizon, initial_target=0.0)
    result = simulate_positions(test_feat[["Open", "Close"]], target, cost, initial_position=0.0)
    return result, target


def _baseline_target(model: str, test_feat: pd.DataFrame, horizon: int) -> pd.Series:
    functions = {
        "cash": cash_baseline,
        "buy_hold": buy_and_hold_baseline,
        "zscore_band": zscore_rule_baseline,
        "sma200_trend": sma200_rule_baseline,
    }
    return functions[model](test_feat, interval=horizon)


def _run_dir(root: Path, ticker: str, model: str, horizon: int, cost: float, seed: int | None) -> Path:
    seed_name = "seed_NA" if seed is None else f"seed_{seed}"
    return root / ticker.lower() / model / f"horizon_{horizon}" / f"cost_{cost:g}" / seed_name


def _run_fingerprint(config_digest: str, run_spec: dict[str, Any], episodes: int) -> str:
    payload = {"config_digest": config_digest, "run_spec": run_spec, "episodes": episodes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _is_complete(out_dir: Path, fingerprint: str, is_rl: bool) -> bool:
    required = ["config.json", "metrics.json", "daily_returns.csv", "equity_curve.csv", "actions.csv", "trades.csv"]
    if is_rl:
        required.extend(["q_table.json", "visit_counts.csv"])
    marker_path = out_dir / "completion.json"
    if not marker_path.exists() or any(not (out_dir / name).exists() for name in required):
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("status") == "complete" and marker.get("fingerprint") == fingerprint


def _add_visit_diagnostics(summary: dict[str, Any], threshold: int) -> dict[str, Any]:
    counts = np.asarray(summary["visit_counts"], dtype=np.int64)
    labels = summary.get("state_labels", {})
    low_cells = []
    for state in range(counts.shape[0]):
        for action in range(counts.shape[1]):
            count = int(counts[state, action])
            if count < threshold:
                low_cells.append(
                    {
                        "state": state,
                        "state_label": labels.get(state, labels.get(str(state), str(state))),
                        "action": action,
                        "visit_count": count,
                    }
                )
    return {
        **summary,
        "low_sample_threshold": threshold,
        "low_sample_cells": low_cells,
        "has_low_sample_cells": bool(low_cells),
    }


def _save_run(
    out_dir: Path,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    result,
    target: pd.Series,
    fingerprint: str,
    q_summary: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", run_config)
    _write_json(out_dir / "metrics.json", metrics)
    result.daily.to_csv(out_dir / "daily_returns.csv")
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    target.to_csv(out_dir / "actions.csv", header=["target_position"])
    result.trades.to_csv(out_dir / "trades.csv", index=False)
    if q_summary is not None:
        _write_json(out_dir / "q_table.json", q_summary)
        counts = np.asarray(q_summary["visit_counts"], dtype=np.int64)
        labels = q_summary["state_labels"]
        visits = [
            {
                "state": state,
                "state_label": labels.get(state, labels.get(str(state), str(state))),
                "action": action,
                "action_label": "TargetCash" if action == 0 else "TargetLong",
                "visit_count": int(counts[state, action]),
                "low_sample": int(counts[state, action]) < int(q_summary["low_sample_threshold"]),
            }
            for state in range(counts.shape[0])
            for action in range(counts.shape[1])
        ]
        pd.DataFrame(visits).to_csv(out_dir / "visit_counts.csv", index=False)
    _write_json(
        out_dir / "completion.json",
        {"status": "complete", "fingerprint": fingerprint, "run_spec": run_config["run"]},
    )


def _metrics_payload(result, benchmark_returns: pd.Series, run_spec: dict[str, Any]) -> dict[str, Any]:
    metrics = full_metrics(result.equity_curve, result.daily_returns, benchmark_returns)
    metrics.update(
        {
            "ticker": run_spec["ticker"],
            "model": run_spec["model"],
            "horizon": run_spec["horizon"],
            "cost": run_spec["cost"],
            "seed": run_spec["seed"] if run_spec["seed"] is not None else "NA",
            "gamma": run_spec["gamma"],
            "turnover": result.turnover,
            "total_cost_paid": result.total_cost_paid,
            "num_entries": result.num_entries,
            "num_exits": result.num_exits,
            "exposure": result.exposure,
            "avg_holding_period": result.avg_holding_period,
            "final_equity": float(result.equity_curve.iloc[-1]),
        }
    )
    if not all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))):
        raise ValueError(f"Non-finite metric produced for {run_spec}")
    if not np.isfinite(result.daily_returns.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite daily return produced for {run_spec}")
    return metrics


def _build_agent(model: str, config: dict[str, Any], horizon: int, episodes: int):
    gamma = resolve_gamma(horizon, int(config["gamma"]["tabular_q_half_life_days"]))
    if model == "bandit_3state":
        return ThreeStateBanditAgent(
            horizon,
            episodes,
            float(config["epsilon_start"]),
            float(config["epsilon_end"]),
        )
    tabular_config = TabularQConfig(
        gamma=gamma,
        episodes=episodes,
        epsilon_start=float(config["epsilon_start"]),
        epsilon_end=float(config["epsilon_end"]),
        decision_interval=horizon,
    )
    if model == "tabular_q_6state":
        return SixStateTabularQAgent(tabular_config)
    if model == "tabular_q_12state":
        return TabularQAgent(tabular_config)
    raise ValueError(f"Unknown RL model: {model}")


def _expected_run_count(config: dict[str, Any], smoke: bool) -> int:
    n_ticker_horizons = len(config["tickers"]) * len(config["reward_horizons_days"])
    n_seeds = 1 if smoke else len(config["seeds"])
    return n_ticker_horizons * len(config["cost_grid"]) * (
        len(BASELINE_MODELS) + len(RL_MODELS) * n_seeds
    )


def _update_status(root: Path, mode: str, completed: int, expected: int, current: dict[str, Any]) -> None:
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
    root.mkdir(parents=True, exist_ok=True)
    config_digest = _config_digest(config)
    episodes = 2 if smoke else int(config["episodes"])
    seeds = [int(config["seeds"][0])] if smoke else [int(seed) for seed in config["seeds"]]
    expected = _expected_run_count(config, smoke)
    completed = 0
    started = time.time()

    for ticker, split in config["tickers"].items():
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
                        gamma = 0.0 if model in BASELINE_MODELS or model == "bandit_3state" else resolve_gamma(
                            horizon, int(config["gamma"]["tabular_q_half_life_days"])
                        )
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
                            "source_config": str(config_path),
                            "split": split,
                            "run": run_spec,
                            "config_digest": config_digest,
                            "platform": platform.platform(),
                            "python_version": sys.version,
                        }
                        _save_run(out_dir, run_config, metrics, result, target, fingerprint, q_summary)
                        completed += 1
                        _update_status(root, mode, completed, expected, run_spec)

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


def _load_daily_returns(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "daily_returns.csv", index_col=0, parse_dates=True)
    return frame["daily_return"]


def _completion_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for marker_path in root.glob("*/*/horizon_*/cost_*/seed_*/completion.json"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete":
            continue
        metrics = json.loads((marker_path.parent / "metrics.json").read_text(encoding="utf-8"))
        records.append({**metrics, "_run_dir": marker_path.parent})
    return records


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


def _visit_summaries(
    records: list[dict[str, Any]], summary_dir: Path, low_sample_threshold: int
) -> pd.DataFrame:
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
    distribution.to_csv(summary_dir / "visit_count_distribution.csv", index=False)
    return distribution


def _bootstrap_summaries(
    records: list[dict[str, Any]], config: dict[str, Any], summary_dir: Path
) -> pd.DataFrame:
    primary_cost = float(config["primary_cost"])
    selected = [record for record in records if float(record["cost"]) == primary_cost]
    benchmark_dirs = {
        (record["ticker"], int(record["horizon"])): Path(record["_run_dir"])
        for record in selected
        if record["model"] == "buy_hold"
    }
    bootstrap_config = BootstrapConfig(**config["bootstrap"])
    rows = []
    for record in selected:
        if record["model"] == "buy_hold":
            continue
        key = (record["ticker"], int(record["horizon"]))
        result = moving_block_bootstrap_sharpe_diff(
            _load_daily_returns(Path(record["_run_dir"])),
            _load_daily_returns(benchmark_dirs[key]),
            bootstrap_config,
        )
        rows.append(
            {
                "ticker": record["ticker"],
                "state_space": record["model"],
                "horizon": record["horizon"],
                "cost": record["cost"],
                "seed": record["seed"],
                **result,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "bootstrap_intervals.csv", index=False)
    return frame


def _format_visit_table(distribution: pd.DataFrame) -> str:
    lines = [
        "| state space | horizon | cells | min | p25 | median | mean | p75 | max | cells < threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.sort_values(["state_space", "horizon"]).itertuples():
        lines.append(
            f"| {row.state_space} | {row.horizon} | {row.n_cells} | {row.minimum:.0f} | "
            f"{row.p25:.1f} | {row.median:.1f} | {row.mean:.1f} | {row.p75:.1f} | "
            f"{row.maximum:.0f} | {row.cells_below_threshold} |"
        )
    return "\n".join(lines)


def _experiment12_crosscheck(records: list[dict[str, Any]], config: dict[str, Any]) -> str:
    legacy_path = (
        REPO_ROOT
        / "experiments"
        / "12_single_stock_qlearning_comparison"
        / "results"
        / "corrected_train_window_v1"
        / "summary"
        / "metrics_long.csv"
    )
    if not legacy_path.exists():
        return "Experiment 12 corrected summary was not available; no numerical cross-check was generated."
    legacy = pd.read_csv(legacy_path)
    legacy = legacy[(legacy["model"] == "tabular_q") & (legacy["cost"] == float(config["primary_cost"]))]
    current = pd.DataFrame(
        [
            {key: value for key, value in record.items() if key != "_run_dir"}
            for record in records
            if record["model"] == "tabular_q_12state"
            and int(record["horizon"]) == 5
            and float(record["cost"]) == float(config["primary_cost"])
        ]
    )
    overlap = sorted(set(current["ticker"]) & set(legacy["ticker"])) if not current.empty else []
    if not overlap:
        return "No overlapping ticker was found for the Experiment 12 numerical cross-check."
    lines = [
        "The values below are a diagnostic cross-check, not a matched causal comparison: the train/test protocols differ.",
        "",
        "| ticker | this experiment mean Sharpe (gamma=0.9439) | Experiment 12 mean Sharpe (gamma=0.90) | difference |",
        "|---|---:|---:|---:|",
    ]
    for ticker in overlap:
        current_mean = float(current.loc[current["ticker"] == ticker, "sharpe"].mean())
        legacy_mean = float(legacy.loc[legacy["ticker"] == ticker, "sharpe"].mean())
        lines.append(f"| {ticker} | {current_mean:.6f} | {legacy_mean:.6f} | {current_mean - legacy_mean:.6f} |")
    return "\n".join(lines)


def _update_readme_results(distribution: pd.DataFrame, crosscheck: str) -> None:
    readme_path = EXPERIMENT_DIR / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    generated = (
        f"{RESULTS_START}\n"
        "## Formal-run results\n\n"
        "Generated only after the complete formal grid passed completeness checks.\n\n"
        "### State-action visit-count distribution\n\n"
        f"{_format_visit_table(distribution)}\n\n"
        "Full cell-level counts are in `results/summary/visit_counts_long.csv`.\n\n"
        "### Experiment 12 cross-check\n\n"
        f"{crosscheck}\n"
        f"{RESULTS_END}"
    )
    start = text.index(RESULTS_START)
    end = text.index(RESULTS_END) + len(RESULTS_END)
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
    distribution = _visit_summaries(
        records,
        summary_dir,
        int(config["diagnostics"]["flag_low_sample_cells_below"]),
    )
    _bootstrap_summaries(records, config, summary_dir)
    _update_readme_results(distribution, _experiment12_crosscheck(records, config))
    _write_json(
        summary_dir / "_FINALIZED.json",
        {
            "status": "complete",
            "config_digest": _config_digest(config),
            "n_runs": len(records),
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
        with _single_instance_lock(smoke_root, "smoke"):
            root = _run_grid(config, args.config, smoke=True)
            validate_smoke(config, root)
        print(f"Smoke validation passed: {root}")
        return
    if args.finalize_only:
        with _single_instance_lock(RESULTS_ROOT, "formal"):
            finalize_formal(config)
        print(f"Formal summaries rebuilt: {RESULTS_ROOT / 'summary'}")
        return

    require_valid_smoke(config)
    with _single_instance_lock(RESULTS_ROOT, "formal"):
        _run_grid(config, args.config, smoke=False)
        finalize_formal(config)
    print(f"Formal run and summaries complete: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()