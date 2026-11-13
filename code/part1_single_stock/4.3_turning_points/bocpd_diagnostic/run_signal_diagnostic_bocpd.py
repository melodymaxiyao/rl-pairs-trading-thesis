"""Stage-1 signal diagnostic for the causal BOCPD turn detector.

No policy is trained.  The complete 42-test family is evaluated and Holm
corrected in one run; stage 2 remains subject to the preregistered gate and an
explicit user decision.
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

from experiments.single_stock.features_bocpd import (
    CONFIRMED_PEAK,
    CONFIRMED_TROUGH,
    NO_CONFIRMED_TURN,
    TURN_STATE_LABELS,
    build_bocpd_features,
    installed_dependency_version,
)
from experiments.single_stock.holm_correct_zslope import holm_step_down
from experiments.single_stock.run_signal_diagnostic_confirmed_turns import (
    moving_block_bootstrap_turn_spread,
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


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_bocpd.json"
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "bocpd" / "signal_diagnostic"
README_START = "<!-- BOCPD_SIGNAL_RESULTS_START -->"
README_END = "<!-- BOCPD_SIGNAL_RESULTS_END -->"
EXPECTED_FOLDS = tuple(f"fold_{number:02d}" for number in range(1, 5))
EXPECTED_TICKERS = ("SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL")
EXPECTED_HORIZONS = (5, 10)
EXPECTED_HAZARDS = (50.0, 100.0, 200.0)
EXPECTED_THRESHOLDS = (0.5,)
EXPECTED_FAMILY_SIZE = 42


def load_bocpd_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    feature = config["bocpd_feature"]
    dependency = config["dependency"]
    diagnostic = config["signal_diagnostic"]
    if dependency != {
        "package": "sdt-python",
        "version": "20.1.4",
        "api": "sdt.changepoint.BayesOnline",
        "engine": "python",
        "pin": "sdt-python==20.1.4",
    }:
        raise ValueError("The BOCPD third-party dependency must remain exactly pinned")
    if installed_dependency_version() != dependency["version"]:
        raise RuntimeError(
            f"Install {dependency['pin']} before running this experiment"
        )
    if feature["input_series"] != "z20" or feature["hazard_model"] != "constant":
        raise ValueError("BOCPD must use z20 and the preregistered constant hazard")
    if tuple(float(x) for x in feature["hazard_rate_lambdas"]) != EXPECTED_HAZARDS:
        raise ValueError(f"hazard grid must be exactly {EXPECTED_HAZARDS}")
    if tuple(float(x) for x in feature["changepoint_prob_threshold_grid"]) != EXPECTED_THRESHOLDS:
        raise ValueError(f"threshold grid must be exactly {EXPECTED_THRESHOLDS}")
    if int(feature["confirmation_lag_days"]) < 1:
        raise ValueError("confirmation_lag_days must be positive")
    if int(feature["direction_pre_window_days"]) < 1:
        raise ValueError("direction_pre_window_days must be positive")
    model = feature["observation_model"]
    if model["name"] != "student_t" or model["conjugate_prior"] != "normal_inverse_gamma":
        raise ValueError("Observation model must be the configured Student-t/NIG model")
    if tuple(config["folds"]) != EXPECTED_FOLDS:
        raise ValueError(f"folds must be exactly {EXPECTED_FOLDS}")
    if tuple(config["tickers"]) != EXPECTED_TICKERS:
        raise ValueError(f"tickers must be exactly {EXPECTED_TICKERS}")
    if tuple(config["reward_horizons_days"]) != EXPECTED_HORIZONS:
        raise ValueError(f"horizons must be exactly {EXPECTED_HORIZONS}")
    if int(config["bootstrap"]["n_repetitions"]) != 5000:
        raise ValueError("The diagnostic requires 5,000 bootstrap repetitions")
    if int(diagnostic["minimum_events_per_turn_state"]) < 1:
        raise ValueError("minimum_events_per_turn_state must be positive")
    if not 0.0 < float(diagnostic["holm_alpha"]) < 1.0:
        raise ValueError("holm_alpha must be in (0, 1)")
    if config["cost_grid"] != [0.0, 0.0005, 0.001, 0.002]:
        raise ValueError("Experiment-2 cost grid must remain recorded in config")
    if config["seeds"] != list(range(10)) or int(config["episodes"]) != 100:
        raise ValueError("Experiment-2 seeds and episodes must remain recorded in config")
    validate_non_overlapping_test_periods(config["folds"])
    family_size = (
        len(config["tickers"])
        * len(config["reward_horizons_days"])
        * len(feature["hazard_rate_lambdas"])
        * len(feature["changepoint_prob_threshold_grid"])
    )
    if family_size != EXPECTED_FAMILY_SIZE:
        raise ValueError(f"Signal diagnostic family must contain {EXPECTED_FAMILY_SIZE} tests")
    return config


def _load_base_features(
    ticker: str, split: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    requested_raw_start = pd.Timestamp(split["actual_train_start"]) - pd.Timedelta(
        days=int(config["feature_warmup_calendar_days"])
    )
    cache_path = REPO_ROOT / "data" / "raw" / f"{ticker.lower()}_daily_adjusted_ohlcv.csv"
    raw_start = requested_raw_start
    if cache_path.exists():
        first_cached = pd.Timestamp(
            pd.read_csv(cache_path, usecols=[0], nrows=1).iloc[0, 0]
        )
        if requested_raw_start < first_cached < pd.Timestamp(split["actual_train_start"]):
            raw_start = first_cached
    ohlcv = load_ohlcv(ticker, raw_start.strftime("%Y-%m-%d"), split["test_end"])
    return build_feature_frame(ohlcv)


def _build_features(
    base: pd.DataFrame,
    config: dict[str, Any],
    hazard_lambda: float,
    probability_threshold: float,
) -> pd.DataFrame:
    feature = config["bocpd_feature"]
    model = feature["observation_model"]
    turns = build_bocpd_features(
        base["z20"],
        hazard_lambda=hazard_lambda,
        probability_threshold=probability_threshold,
        confirmation_lag_days=int(feature["confirmation_lag_days"]),
        direction_pre_window_days=int(feature["direction_pre_window_days"]),
        observation_model={
            "name": model["name"],
            "mu": model["mu"],
            "kappa": model["kappa"],
            "alpha": model["alpha"],
            "beta": model["beta"],
        },
        dependency_engine=config["dependency"]["engine"],
        expected_dependency_version=config["dependency"]["version"],
    )
    return pd.concat([base, turns], axis=1)


def _fold_observations(
    feat: pd.DataFrame,
    split: dict[str, Any],
    ticker: str,
    fold: str,
    horizon: int,
    hazard_lambda: float,
    probability_threshold: float,
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
        reward, _, _ = single_step_execution(feat, loc, end_loc, 0.0, 1.0, 0.0)
        state = int(feat["confirmed_turn_state"].iloc[loc])
        rows.append(
            {
                "date": decision_date,
                "ticker": ticker,
                "fold": fold,
                "horizon": horizon,
                "hazard_lambda": hazard_lambda,
                "probability_threshold": probability_threshold,
                "confirmed_turn_state": state,
                "confirmed_turn_label": TURN_STATE_LABELS[state],
                "confirmed_turn_source_date": feat["bocpd_turn_source_date"].iloc[loc],
                "changepoint_probability": float(
                    feat["bocpd_changepoint_probability"].iloc[loc]
                ),
                "run_length_mode": float(feat["bocpd_run_length_mode"].iloc[loc]),
                "future_log_return": float(reward),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No diagnostic observations for {ticker}/{fold}/h{horizon}")
    return result


def _fold_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["ticker", "horizon", "hazard_lambda", "probability_threshold", "fold"]
    for keys, group in observations.groupby(columns, sort=False):
        ticker, horizon, hazard_lambda, threshold, fold = keys
        groups = {
            state: group.loc[group["confirmed_turn_state"] == state, "future_log_return"]
            for state in TURN_STATE_LABELS
        }
        peak, trough = groups[CONFIRMED_PEAK], groups[CONFIRMED_TROUGH]
        event_count = len(peak) + len(trough)
        rows.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "hazard_lambda": hazard_lambda,
                "probability_threshold": threshold,
                "fold": fold,
                "n_observations": len(group),
                "n_no_confirmed_turn": len(groups[NO_CONFIRMED_TURN]),
                "n_confirmed_peak": len(peak),
                "n_confirmed_trough": len(trough),
                "turns_per_100_trading_days": 100.0 * event_count / len(group),
                "mean_return_no_confirmed_turn": float(groups[NO_CONFIRMED_TURN].mean()),
                "mean_return_confirmed_peak": float(peak.mean()) if len(peak) else np.nan,
                "mean_return_confirmed_trough": float(trough.mean()) if len(trough) else np.nan,
                "trough_minus_peak_mean_return": (
                    float(trough.mean() - peak.mean()) if len(peak) and len(trough) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _stitched_tests(
    observations: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    bootstrap = BootstrapConfig(**config["bootstrap"])
    minimum = int(config["signal_diagnostic"]["minimum_events_per_turn_state"])
    rows, distributions = [], {}
    feature = config["bocpd_feature"]
    for ticker in config["tickers"]:
        for horizon in config["reward_horizons_days"]:
            for hazard in feature["hazard_rate_lambdas"]:
                for threshold in feature["changepoint_prob_threshold_grid"]:
                    selected = observations[
                        (observations["ticker"] == ticker)
                        & (observations["horizon"] == horizon)
                        & (observations["hazard_lambda"] == float(hazard))
                        & (observations["probability_threshold"] == float(threshold))
                    ].sort_values("date")
                    indexed = selected.set_index("date")
                    result = moving_block_bootstrap_turn_spread(
                        indexed["future_log_return"],
                        indexed["confirmed_turn_state"],
                        bootstrap,
                        minimum,
                    )
                    key = f"{ticker}__h{horizon}__lambda{hazard:g}__p{threshold:g}"
                    distributions[key] = result.pop("bootstrap_samples")
                    peak = indexed["confirmed_turn_state"] == CONFIRMED_PEAK
                    trough = indexed["confirmed_turn_state"] == CONFIRMED_TROUGH
                    event_count = int(peak.sum() + trough.sum())
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": int(horizon),
                            "hazard_lambda": float(hazard),
                            "probability_threshold": float(threshold),
                            "mean_return_confirmed_peak": float(indexed.loc[peak, "future_log_return"].mean()),
                            "mean_return_confirmed_trough": float(indexed.loc[trough, "future_log_return"].mean()),
                            "trough_minus_peak_mean_return": result["point_estimate"],
                            "ci_lower_95": result["ci_lower_95"],
                            "ci_upper_95": result["ci_upper_95"],
                            "raw_p_value": result["raw_p_value"],
                            "turns_per_100_trading_days": 100.0 * event_count / len(indexed),
                            "n_folds": len(config["folds"]),
                            **{
                                name: value
                                for name, value in result.items()
                                if name not in {"point_estimate", "ci_lower_95", "ci_upper_95", "raw_p_value"}
                            },
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_FAMILY_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_FAMILY_SIZE} tests, found {len(frame)}")
    return frame, distributions


def _apply_holm_and_gate(
    stitched: pd.DataFrame, folds: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    alpha = float(config["signal_diagnostic"]["holm_alpha"])
    correction = holm_step_down(stitched["raw_p_value"], alpha=alpha)
    holm = stitched.iloc[correction["original_index"].to_numpy()].reset_index(drop=True)
    holm = pd.concat(
        [
            holm,
            correction[["holm_rank", "holm_adjusted_threshold", "passes_holm_correction"]].reset_index(drop=True),
        ],
        axis=1,
    )
    direction = (
        folds.assign(positive=folds["trough_minus_peak_mean_return"].fillna(-np.inf) > 0.0)
        .groupby(["ticker", "horizon", "hazard_lambda", "probability_threshold"], as_index=False)["positive"]
        .sum()
        .rename(columns={"positive": "positive_folds"})
    )
    keys = ["ticker", "horizon", "hazard_lambda", "probability_threshold"]
    holm = holm.merge(direction, on=keys, how="left")
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
            hazard_values_tested=("hazard_lambda", "size"),
            hazard_values_positive=("trough_minus_peak_mean_return", lambda x: int((x > 0).sum())),
            hazard_values_passing_holm=("passes_holm_correction", "sum"),
            hazard_values_passing_direction_and_holm=("passes_direction_and_holm", "sum"),
            min_turns_per_100_days=("turns_per_100_trading_days", "min"),
            max_turns_per_100_days=("turns_per_100_trading_days", "max"),
        )
    )
    minimum_hazards = int(gate_config["minimum_hazard_rates_passing_per_ticker_horizon"])
    sensitivity["robust_enough_for_stage2"] = (
        sensitivity["hazard_values_passing_direction_and_holm"] >= minimum_hazards
    )
    passing = sensitivity[sensitivity["robust_enough_for_stage2"]]
    gate_passed = not passing.empty
    gate = {
        "status": "passed" if gate_passed else "failed",
        "stage2_allowed": gate_passed,
        "holm_family_size": len(holm),
        "holm_alpha": alpha,
        "n_holm_significant_comparisons": int(holm["passes_holm_correction"].sum()),
        "n_direction_and_holm_comparisons": int(holm["passes_direction_and_holm"].sum()),
        "n_robust_ticker_horizon_groups": len(passing),
        "minimum_positive_folds": minimum_positive_folds,
        "minimum_hazard_rates_passing": minimum_hazards,
        "passing_ticker_horizon_groups": passing.to_dict("records"),
        "decision": (
            "BOCPD signal diagnostic passed the preregistered gate; wait for explicit user approval before implementing stage 2."
            if gate_passed
            else "BOCPD signal diagnostic found no hazard-robust Holm-significant relationship; do not implement or run stage 2."
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
    feature = config["bocpd_feature"]
    lines = [
        "# BOCPD confirmed-turn signal diagnostic",
        "",
        "本阶段只检验 BOCPD confirmed-turn 标签与未来收益的关系，没有训练或回测 Q-learning 模型。",
        f"第三方实现固定为 `{config['dependency']['pin']}`；输入为z20，Student-t/NIG先验为 "
        f"mu={feature['observation_model']['mu']}, kappa={feature['observation_model']['kappa']}, "
        f"alpha={feature['observation_model']['alpha']}, beta={feature['observation_model']['beta']}。",
        f"因 constant hazard 下当期P(r_t=0)=1/lambda，本实验以确认滞后 d={feature['confirmation_lag_days']} "
        "读取P(r_t=d)，即在决策时点t确认发生于t-d的变点；全过程只使用z20_0:t。",
        f"扫描lambda={feature['hazard_rate_lambdas']}、概率门槛={feature['changepoint_prob_threshold_grid']}。"
        "统计量为波谷标签后平均未来log return减波峰标签后平均未来log return，正值符合预期。",
        "全部42项检验在同一次运行内复用holm_correct_zslope.py的Holm-Bonferroni实现。",
        "",
        "## 完整 Holm 排序",
        "",
        "| rank | ticker | h | lambda | threshold | trough-peak mean | 95% CI | raw p | Holm门槛 | +folds | turns/100d | 通过Holm |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in holm.itertuples(index=False):
        finite_ci = np.isfinite(row.ci_lower_95) and np.isfinite(row.ci_upper_95)
        ci = f"[{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}]" if finite_ci else "insufficient events"
        point = f"{row.trough_minus_peak_mean_return:.6f}" if np.isfinite(row.trough_minus_peak_mean_return) else "NA"
        lines.append(
            f"| {row.holm_rank} | {row.ticker} | {row.horizon} | {row.hazard_lambda:g} | "
            f"{row.probability_threshold:g} | {point} | {ci} | {row.raw_p_value:.6f} | "
            f"{row.holm_adjusted_threshold:.8f} | {int(row.positive_folds)}/4 | "
            f"{row.turns_per_100_trading_days:.3f} | {'是' if row.passes_holm_correction else '否'} |"
        )
    lines.extend(
        [
            "",
            "## hazard-rate敏感性与第二阶段gate",
            "",
            "| ticker | h | 正向lambda数 | Holm通过lambda数 | 方向与Holm均通过 | turns/100d范围 | 达到第二阶段门槛 |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {row.horizon} | {int(row.hazard_values_positive)}/3 | "
            f"{int(row.hazard_values_passing_holm)}/3 | "
            f"{int(row.hazard_values_passing_direction_and_holm)}/3 | "
            f"{row.min_turns_per_100_days:.3f}–{row.max_turns_per_100_days:.3f} | "
            f"{'是' if row.robust_enough_for_stage2 else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"Holm校正后显著的单项组合：{gate['n_holm_significant_comparisons']}/{gate['holm_family_size']}。",
            f"通过hazard稳健性gate的ticker-horizon组：{gate['n_robust_ticker_horizon_groups']}/14。",
            (
                "第一阶段通过预设gate；在用户明确批准前仍不实现或运行第二阶段。"
                if gate["stage2_allowed"]
                else "信号诊断阶段即未发现满足预设稳健性门槛的显著关系；第二阶段不应继续。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _update_readme(report: str) -> None:
    path = EXPERIMENT_DIR / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise RuntimeError("README BOCPD result markers are missing")
    start = text.index(README_START)
    end = text.index(README_END) + len(README_END)
    path.write_text(
        text[:start] + README_START + "\n" + report + README_END + text[end:],
        encoding="utf-8",
    )


def run_signal_diagnostic(config_path: Path) -> dict[str, Any]:
    config = load_bocpd_config(config_path)
    observations = []
    feature = config["bocpd_feature"]
    for ticker in config["tickers"]:
        for fold, configured_split in config["folds"].items():
            split = resolve_effective_split(
                ticker, fold, configured_split, config["tsla_min_train_start"]
            )
            base = _load_base_features(ticker, split, config)
            for hazard in feature["hazard_rate_lambdas"]:
                for threshold in feature["changepoint_prob_threshold_grid"]:
                    feat = _build_features(base, config, float(hazard), float(threshold))
                    for horizon in config["reward_horizons_days"]:
                        observations.append(
                            _fold_observations(
                                feat,
                                split,
                                ticker,
                                fold,
                                int(horizon),
                                float(hazard),
                                float(threshold),
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
    _atomic_csv(sensitivity, RESULTS_ROOT / "hazard_sensitivity.csv")
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
    with _single_instance_lock(RESULTS_ROOT, "bocpd_signal_diagnostic"):
        gate = run_signal_diagnostic(args.config)
    print(f"Signal diagnostic complete: {RESULTS_ROOT / 'report.md'}")
    print(f"Stage 2 allowed: {gate['stage2_allowed']}")


if __name__ == "__main__":
    main()
