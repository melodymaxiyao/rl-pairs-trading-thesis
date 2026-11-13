"""Stage-1 signal diagnostic for causal confirmed turning points.

No policy is trained here.  The diagnostic compares future gross log returns
following confirmed troughs and confirmed peaks across the four non-overlapping
Experiment-2 OOS folds, bootstraps their mean spread with paired time blocks,
and applies the already-audited Holm step-down implementation to the complete
42-comparison family in the same run.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.single_stock.features_confirmed_turns import (
    CONFIRMED_PEAK,
    CONFIRMED_TROUGH,
    NO_CONFIRMED_TURN,
    TURN_STATE_LABELS,
    build_confirmed_turn_features,
)
from experiments.single_stock.holm_correct_zslope import (
    holm_step_down,
    two_sided_bootstrap_p_value,
)
from experiments.single_stock.run_single_stock import (
    _config_digest,
    _single_instance_lock,
    _write_json,
)
from experiments.single_stock.run_single_stock_walkforward import (
    resolve_effective_split,
    validate_non_overlapping_test_periods,
)
from fyp_rl.config import BootstrapConfig
from fyp_rl.data import load_ohlcv
from fyp_rl.execution import single_step_execution
from fyp_rl.features import build_feature_frame


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_confirmed_turns.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "confirmed_turns" / "signal_diagnostic"
README_START = "<!-- CONFIRMED_TURNS_SIGNAL_RESULTS_START -->"
README_END = "<!-- CONFIRMED_TURNS_SIGNAL_RESULTS_END -->"
EXPECTED_FOLDS = tuple(f"fold_{number:02d}" for number in range(1, 5))
EXPECTED_TICKERS = ("SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL")
EXPECTED_HORIZONS = (5, 10)
EXPECTED_K_MULTIPLIERS = (1.0, 1.5, 2.0)
EXPECTED_FAMILY_SIZE = 42


def load_confirmed_turns_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    feature = config["confirmed_turn_feature"]
    diagnostic = config["signal_diagnostic"]
    if feature["smoothing_method"] != "causal_ema":
        raise ValueError("Only the preregistered causal_ema smoother is supported")
    if int(feature["smooth_window"]) <= 1:
        raise ValueError("smooth_window must be greater than 1")
    if int(feature["vol_lookback_days"]) <= 1:
        raise ValueError("vol_lookback_days must be greater than 1")
    if int(feature["confirmation_lag_days"]) < 1:
        raise ValueError("confirmation_lag_days must be at least 1")
    if tuple(float(k) for k in feature["k_multipliers"]) != EXPECTED_K_MULTIPLIERS:
        raise ValueError(f"k_multipliers must be exactly {EXPECTED_K_MULTIPLIERS}")
    if tuple(config["folds"]) != EXPECTED_FOLDS:
        raise ValueError(f"folds must be exactly {EXPECTED_FOLDS}")
    if tuple(config["tickers"]) != EXPECTED_TICKERS:
        raise ValueError(f"tickers must be exactly {EXPECTED_TICKERS}")
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError(f"horizons must be exactly {EXPECTED_HORIZONS}")
    if int(diagnostic["minimum_events_per_turn_state"]) < 1:
        raise ValueError("minimum_events_per_turn_state must be positive")
    if not 0.0 < float(diagnostic["holm_alpha"]) < 1.0:
        raise ValueError("holm_alpha must be in (0, 1)")
    if int(config["bootstrap"]["n_repetitions"]) != 5000:
        raise ValueError("The diagnostic requires the preregistered 5,000 repetitions")
    if config["cost_grid"] != [0.0, 0.0005, 0.001, 0.002]:
        raise ValueError("Experiment-2 cost grid must remain recorded in config")
    if config["seeds"] != list(range(10)) or int(config["episodes"]) != 100:
        raise ValueError("Experiment-2 seeds and episodes must remain recorded in config")
    validate_non_overlapping_test_periods(config["folds"])
    family_size = (
        len(config["tickers"])
        * len(config["reward_horizons_days"])
        * len(feature["k_multipliers"])
    )
    if family_size != EXPECTED_FAMILY_SIZE:
        raise ValueError(f"Signal diagnostic family must contain {EXPECTED_FAMILY_SIZE} tests")
    return config


def moving_block_bootstrap_turn_spread(
    future_returns: pd.Series,
    turn_states: pd.Series,
    config: BootstrapConfig,
    minimum_events_per_state: int,
) -> dict[str, Any]:
    """Bootstrap mean(trough return)-mean(peak return) with shared time blocks."""
    aligned = pd.concat(
        [future_returns.rename("future_return"), turn_states.rename("turn_state")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty or not aligned.index.is_unique or not aligned.index.is_monotonic_increasing:
        raise ValueError("Diagnostic observations must have unique increasing dates")
    values = aligned["future_return"].to_numpy(dtype=np.float64)
    states = aligned["turn_state"].to_numpy(dtype=np.int8)
    if not np.isfinite(values).all():
        raise ValueError("Future returns must be finite")
    peak_count = int(np.sum(states == CONFIRMED_PEAK))
    trough_count = int(np.sum(states == CONFIRMED_TROUGH))
    if peak_count < minimum_events_per_state or trough_count < minimum_events_per_state:
        return {
            "point_estimate": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "raw_p_value": 1.0,
            "n_observations": len(aligned),
            "n_peak_events": peak_count,
            "n_trough_events": trough_count,
            "n_valid_bootstrap_draws": 0,
            "n_invalid_bootstrap_draws": config.n_repetitions,
            "sufficient_events": False,
            "bootstrap_samples": np.asarray([], dtype=np.float64),
        }
    if len(aligned) < config.block_length:
        raise ValueError("Insufficient observations for configured block length")

    peak_mean = float(values[states == CONFIRMED_PEAK].mean())
    trough_mean = float(values[states == CONFIRMED_TROUGH].mean())
    point = trough_mean - peak_mean
    rng = np.random.default_rng(config.seed)
    blocks = int(np.ceil(len(aligned) / config.block_length))
    samples: list[float] = []
    invalid = 0
    for offset in range(0, config.n_repetitions, 64):
        batch = min(64, config.n_repetitions - offset)
        starts = rng.integers(
            0,
            len(aligned) - config.block_length + 1,
            size=(batch, blocks),
        )
        indices = (
            starts[:, :, None] + np.arange(config.block_length)
        ).reshape(batch, -1)[:, : len(aligned)]
        sampled_values = values[indices]
        sampled_states = states[indices]
        peak_mask = sampled_states == CONFIRMED_PEAK
        trough_mask = sampled_states == CONFIRMED_TROUGH
        peak_n = peak_mask.sum(axis=1)
        trough_n = trough_mask.sum(axis=1)
        valid = (peak_n > 0) & (trough_n > 0)
        invalid += int((~valid).sum())
        if valid.any():
            sampled_peak_means = np.divide(
                (sampled_values * peak_mask).sum(axis=1)[valid], peak_n[valid]
            )
            sampled_trough_means = np.divide(
                (sampled_values * trough_mask).sum(axis=1)[valid], trough_n[valid]
            )
            samples.extend((sampled_trough_means - sampled_peak_means).tolist())
    distribution = np.asarray(samples, dtype=np.float64)
    if len(distribution) == 0:
        raise RuntimeError("No valid bootstrap draw contained both turn states")
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    return {
        "point_estimate": point,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "raw_p_value": two_sided_bootstrap_p_value(distribution),
        "n_observations": len(aligned),
        "n_peak_events": peak_count,
        "n_trough_events": trough_count,
        "n_valid_bootstrap_draws": len(distribution),
        "n_invalid_bootstrap_draws": invalid,
        "sufficient_events": True,
        "bootstrap_samples": distribution,
    }


def _build_features(
    ticker: str,
    split: dict[str, Any],
    config: dict[str, Any],
    k_multiplier: float,
) -> pd.DataFrame:
    requested_raw_start = pd.Timestamp(split["actual_train_start"]) - pd.Timedelta(
        days=int(config["feature_warmup_calendar_days"])
    )
    # A newly listed ticker can have its first cached trading day immediately
    # after a weekend/holiday raw_start.  The shared loader intentionally
    # demands literal calendar-date coverage and would try the network in that
    # case.  Clamp only to an actually present local first row; never fabricate
    # history and never cross the effective training start.
    cache_path = REPO_ROOT / "data" / "raw" / f"{ticker.lower()}_daily_adjusted_ohlcv.csv"
    raw_start = requested_raw_start
    if cache_path.exists():
        first_cached = pd.Timestamp(
            pd.read_csv(cache_path, usecols=[0], nrows=1).iloc[0, 0]
        )
        if requested_raw_start < first_cached < pd.Timestamp(split["actual_train_start"]):
            raw_start = first_cached
    ohlcv = load_ohlcv(
        ticker,
        raw_start.strftime("%Y-%m-%d"),
        split["test_end"],
    )
    base = build_feature_frame(ohlcv)
    feature_config = config["confirmed_turn_feature"]
    turns = build_confirmed_turn_features(
        base["z20"],
        smooth_window=int(feature_config["smooth_window"]),
        vol_lookback_days=int(feature_config["vol_lookback_days"]),
        k_multiplier=k_multiplier,
        confirmation_lag_days=int(feature_config["confirmation_lag_days"]),
        local_vol_ddof=int(feature_config["local_vol_ddof"]),
    )
    return pd.concat([base, turns], axis=1)


def _fold_observations(
    feat: pd.DataFrame,
    split: dict[str, Any],
    ticker: str,
    fold: str,
    horizon: int,
    k_multiplier: float,
) -> pd.DataFrame:
    test_start = pd.Timestamp(split["test_start"])
    test_end = pd.Timestamp(split["test_end"])
    rows = []
    for loc, decision_date in enumerate(feat.index):
        if decision_date < test_start or decision_date > test_end:
            continue
        end_loc = loc + horizon
        if end_loc >= len(feat) or feat.index[end_loc] > test_end:
            continue
        if pd.isna(feat["z20"].iloc[loc]):
            continue
        reward, _, _ = single_step_execution(
            feat,
            loc,
            end_loc,
            0.0,
            1.0,
            0.0,
        )
        state = int(feat["confirmed_turn_state"].iloc[loc])
        source_date = feat["confirmed_turn_source_date"].iloc[loc]
        rows.append(
            {
                "date": decision_date,
                "ticker": ticker,
                "fold": fold,
                "horizon": horizon,
                "k_multiplier": k_multiplier,
                "confirmed_turn_state": state,
                "confirmed_turn_label": TURN_STATE_LABELS[state],
                "confirmed_turn_source_date": source_date,
                "future_log_return": float(reward),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No diagnostic observations for {ticker}/{fold}/h{horizon}")
    return result


def _fold_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["ticker", "horizon", "k_multiplier", "fold"]
    for keys, group in observations.groupby(group_columns, sort=False):
        ticker, horizon, k_multiplier, fold = keys
        state_groups = {
            state: group.loc[
                group["confirmed_turn_state"] == state, "future_log_return"
            ]
            for state in TURN_STATE_LABELS
        }
        peak = state_groups[CONFIRMED_PEAK]
        trough = state_groups[CONFIRMED_TROUGH]
        rows.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "k_multiplier": k_multiplier,
                "fold": fold,
                "n_observations": len(group),
                "n_no_confirmed_turn": len(state_groups[NO_CONFIRMED_TURN]),
                "n_confirmed_peak": len(peak),
                "n_confirmed_trough": len(trough),
                "mean_return_no_confirmed_turn": float(
                    state_groups[NO_CONFIRMED_TURN].mean()
                ),
                "mean_return_confirmed_peak": float(peak.mean()) if len(peak) else np.nan,
                "mean_return_confirmed_trough": float(trough.mean())
                if len(trough)
                else np.nan,
                "trough_minus_peak_mean_return": float(trough.mean() - peak.mean())
                if len(peak) and len(trough)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _stitched_tests(
    observations: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    bootstrap = BootstrapConfig(**config["bootstrap"])
    minimum = int(config["signal_diagnostic"]["minimum_events_per_turn_state"])
    rows = []
    distributions: dict[str, np.ndarray] = {}
    for ticker in config["tickers"]:
        for horizon in config["reward_horizons_days"]:
            for k_multiplier in config["confirmed_turn_feature"]["k_multipliers"]:
                selected = observations[
                    (observations["ticker"] == ticker)
                    & (observations["horizon"] == horizon)
                    & (observations["k_multiplier"] == k_multiplier)
                ].sort_values("date")
                indexed = selected.set_index("date")
                result = moving_block_bootstrap_turn_spread(
                    indexed["future_log_return"],
                    indexed["confirmed_turn_state"],
                    bootstrap,
                    minimum,
                )
                key = f"{ticker}__h{horizon}__k{k_multiplier:g}"
                distributions[key] = result.pop("bootstrap_samples")
                rows.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "k_multiplier": float(k_multiplier),
                        "mean_return_confirmed_peak": float(
                            indexed.loc[
                                indexed["confirmed_turn_state"] == CONFIRMED_PEAK,
                                "future_log_return",
                            ].mean()
                        ),
                        "mean_return_confirmed_trough": float(
                            indexed.loc[
                                indexed["confirmed_turn_state"] == CONFIRMED_TROUGH,
                                "future_log_return",
                            ].mean()
                        ),
                        "trough_minus_peak_mean_return": result["point_estimate"],
                        "ci_lower_95": result["ci_lower_95"],
                        "ci_upper_95": result["ci_upper_95"],
                        "raw_p_value": result["raw_p_value"],
                        "n_folds": len(config["folds"]),
                        **{
                            name: value
                            for name, value in result.items()
                            if name
                            not in {
                                "point_estimate",
                                "ci_lower_95",
                                "ci_upper_95",
                                "raw_p_value",
                            }
                        },
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_FAMILY_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_FAMILY_SIZE} stitched tests, found {len(frame)}")
    return frame, distributions


def _apply_holm_and_gate(
    stitched: pd.DataFrame,
    folds: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    alpha = float(config["signal_diagnostic"]["holm_alpha"])
    correction = holm_step_down(stitched["raw_p_value"], alpha=alpha)
    holm = stitched.iloc[correction["original_index"].to_numpy()].reset_index(drop=True)
    holm = pd.concat(
        [
            holm,
            correction[
                ["holm_rank", "holm_adjusted_threshold", "passes_holm_correction"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    fold_direction = (
        folds.assign(
            positive=folds["trough_minus_peak_mean_return"].fillna(-np.inf) > 0.0
        )
        .groupby(["ticker", "horizon", "k_multiplier"], as_index=False)["positive"]
        .sum()
        .rename(columns={"positive": "positive_folds"})
    )
    holm = holm.merge(
        fold_direction, on=["ticker", "horizon", "k_multiplier"], how="left"
    )
    gate_config = config["signal_diagnostic"]["stage2_gate"]
    minimum_positive_folds = int(gate_config["minimum_positive_folds"])
    holm["passes_direction_and_holm"] = (
        holm["passes_holm_correction"]
        & (holm["trough_minus_peak_mean_return"] > 0.0)
        & (holm["positive_folds"] >= minimum_positive_folds)
    )
    sensitivity = (
        holm.groupby(["ticker", "horizon"], as_index=False)
        .agg(
            k_values_tested=("k_multiplier", "size"),
            k_values_positive=("trough_minus_peak_mean_return", lambda x: int((x > 0).sum())),
            k_values_passing_holm=("passes_holm_correction", "sum"),
            k_values_passing_direction_and_holm=("passes_direction_and_holm", "sum"),
        )
    )
    minimum_k = int(
        gate_config["minimum_k_multipliers_passing_per_ticker_horizon"]
    )
    sensitivity["robust_enough_for_stage2"] = (
        sensitivity["k_values_passing_direction_and_holm"] >= minimum_k
    )
    passing_groups = sensitivity[sensitivity["robust_enough_for_stage2"]]
    gate_passed = not passing_groups.empty
    gate = {
        "status": "passed" if gate_passed else "failed",
        "stage2_allowed": gate_passed,
        "holm_family_size": len(holm),
        "holm_alpha": alpha,
        "n_holm_significant_comparisons": int(holm["passes_holm_correction"].sum()),
        "n_direction_and_holm_comparisons": int(holm["passes_direction_and_holm"].sum()),
        "n_robust_ticker_horizon_groups": len(passing_groups),
        "minimum_positive_folds": minimum_positive_folds,
        "minimum_k_multipliers_passing": minimum_k,
        "passing_ticker_horizon_groups": passing_groups.to_dict("records"),
        "decision": (
            "Signal diagnostic passed the preregistered gate; wait for explicit user approval before implementing stage 2."
            if gate_passed
            else "Signal diagnostic found no threshold-robust Holm-significant relationship; do not implement or run stage 2."
        ),
        "config_digest": _config_digest(config),
    }
    return holm, sensitivity, gate


def _format_report(
    config: dict[str, Any],
    holm: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gate: dict[str, Any],
) -> str:
    feature = config["confirmed_turn_feature"]
    lines = [
        "# Confirmed-turn signal diagnostic",
        "",
        "本阶段只检验 confirmed-turn 标签与未来收益的统计关系，没有训练或回测 Q-learning 模型。",
        f"特征采用 causal EMA(span={feature['smooth_window']})、"
        f"z20 rolling-vol lookback={feature['vol_lookback_days']}、"
        f"确认滞后 d={feature['confirmation_lag_days']}，并扫描 "
        f"k={feature['k_multipliers']}。",
        "检验统计量为 confirmed trough 后平均未来 log return 减 confirmed peak 后平均未来 log return；"
        "正值符合预期。全部 42 项检验在同一次运行内执行 Holm-Bonferroni 校正。",
        "",
        "## 完整 Holm 排序",
        "",
        "| rank | ticker | h | k | trough-peak mean | 95% CI | raw p | Holm门槛 | +folds | 通过Holm |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for row in holm.itertuples(index=False):
        ci = (
            f"[{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}]"
            if np.isfinite(row.ci_lower_95) and np.isfinite(row.ci_upper_95)
            else "insufficient events"
        )
        point = (
            f"{row.trough_minus_peak_mean_return:.6f}"
            if np.isfinite(row.trough_minus_peak_mean_return)
            else "NA"
        )
        lines.append(
            f"| {row.holm_rank} | {row.ticker} | {row.horizon} | {row.k_multiplier:g} | "
            f"{point} | {ci} | {row.raw_p_value:.6f} | "
            f"{row.holm_adjusted_threshold:.8f} | {int(row.positive_folds)}/4 | "
            f"{'是' if row.passes_holm_correction else '否'} |"
        )
    lines.extend(
        [
            "",
            "## k_multiplier 敏感性与第二阶段 gate",
            "",
            "| ticker | h | 正向k数 | Holm通过k数 | 同时满足方向与Holm的k数 | 达到第二阶段门槛 |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.horizon} | {int(row.k_values_positive)}/3 | "
            f"{int(row.k_values_passing_holm)}/3 | "
            f"{int(row.k_values_passing_direction_and_holm)}/3 | "
            f"{'是' if row.robust_enough_for_stage2 else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"Holm校正后显著的单项组合：{gate['n_holm_significant_comparisons']}/{gate['holm_family_size']}。",
            f"通过阈值稳健性 gate 的 ticker-horizon 组：{gate['n_robust_ticker_horizon_groups']}/14。",
            (
                "第一阶段通过预设 gate；在用户明确批准前仍不实现或运行第二阶段。"
                if gate["stage2_allowed"]
                else "信号诊断阶段即未发现满足预设稳健性门槛的显著关系；第二阶段不应继续。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _update_readme(report: str) -> None:
    path = EXPERIMENT_DIR / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise RuntimeError("README confirmed-turn result markers are missing")
    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    path.write_text(text[:start] + README_START + "\n" + report + README_END + text[end:], encoding="utf-8")


def run_signal_diagnostic(config_path: Path) -> dict[str, Any]:
    config = load_confirmed_turns_config(config_path)
    observations = []
    for ticker in config["tickers"]:
        for fold, configured_split in config["folds"].items():
            split = resolve_effective_split(
                ticker, fold, configured_split, config["tsla_min_train_start"]
            )
            for k_multiplier in config["confirmed_turn_feature"]["k_multipliers"]:
                feat = _build_features(ticker, split, config, float(k_multiplier))
                for horizon in config["reward_horizons_days"]:
                    observations.append(
                        _fold_observations(
                            feat,
                            split,
                            ticker,
                            fold,
                            int(horizon),
                            float(k_multiplier),
                        )
                    )
    all_observations = pd.concat(observations, ignore_index=True)
    folds = _fold_summary(all_observations)
    stitched, distributions = _stitched_tests(all_observations, config)
    holm, sensitivity, gate = _apply_holm_and_gate(stitched, folds, config)
    report = _format_report(config, holm, sensitivity, gate)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(all_observations, RESULTS_ROOT / "observations.csv")
    _atomic_csv(folds, RESULTS_ROOT / "fold_summary.csv")
    _atomic_csv(stitched, RESULTS_ROOT / "stitched_significance.csv")
    _atomic_csv(holm, RESULTS_ROOT / "holm_summary.csv")
    _atomic_csv(sensitivity, RESULTS_ROOT / "k_sensitivity.csv")
    np.savez_compressed(RESULTS_ROOT / "bootstrap_distributions.npz", **distributions)
    _write_json(RESULTS_ROOT / "stage2_gate.json", gate)
    _atomic_text(report, RESULTS_ROOT / "report.md")
    _update_readme(report)
    _write_json(
        RESULTS_ROOT / "_COMPLETE.json",
        {
            "status": "complete",
            "config_digest": _config_digest(config),
            "n_observations": len(all_observations),
            "n_fold_rows": len(folds),
            "n_parallel_tests": len(holm),
            "stage2_allowed": gate["stage2_allowed"],
            "completed_at_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with _single_instance_lock(RESULTS_ROOT, "confirmed_turn_signal_diagnostic"):
        gate = run_signal_diagnostic(args.config)
    print(f"Signal diagnostic complete: {RESULTS_ROOT / 'report.md'}")
    print(f"Stage 2 allowed: {gate['stage2_allowed']}")


if __name__ == "__main__":
    main()
