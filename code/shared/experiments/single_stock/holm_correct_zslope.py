"""Holm-Bonferroni correction for the 28 stitched z-slope comparisons.

This script never trains a policy and never writes into an existing run
directory.  It reads the completed stitched-OOS return paths and captures the
exact bootstrap sample passed to ``np.percentile`` by the established
``bootstrap_mean_seed_sharpe`` function.  The captured sample therefore drives
both the reproduced confidence interval and the raw two-sided p-value.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import fyp_rl.audit_statistics as audit_statistics
from fyp_rl.config import BootstrapConfig


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_zslope.json"
ZSLOPE_RESULTS_ROOT = EXPERIMENT_DIR / "results" / "zslope"
PARENT_RESULTS_ROOT = EXPERIMENT_DIR / "results" / "walkforward"
DEFAULT_STITCHED = ZSLOPE_RESULTS_ROOT / "summary" / "stitched_significance.csv"
DEFAULT_SUMMARY = ZSLOPE_RESULTS_ROOT / "holm_correction_summary.csv"
DEFAULT_REPORT = ZSLOPE_RESULTS_ROOT / "holm_correction_report.md"
EXPECTED_MODELS = ("tabular_q_6state_zslope", "tabular_q_12state_zslope")
EXPECTED_HORIZONS = (5, 10)
EXPECTED_FOLDS = tuple(f"fold_{number:02d}" for number in range(1, 5))
EXPECTED_COMPARISONS = 28
EXPECTED_METHOD = "paired_time_blocks_mean_of_seed_sharpes"
TARGET_TICKER = "TSLA"
TARGET_MODEL = "tabular_q_6state_zslope"
TARGET_HORIZON = 10
CI_ATOL = 1e-12


def two_sided_bootstrap_p_value(samples: Iterable[float]) -> float:
    """Return the requested two-sided sign-tail bootstrap p-value."""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap samples must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap samples must all be finite")
    lower_tail = float(np.mean(values <= 0.0))
    upper_tail = float(np.mean(values >= 0.0))
    return min(2.0 * min(lower_tail, upper_tail), 1.0)


def holm_step_down(p_values: Iterable[float], alpha: float = 0.05) -> pd.DataFrame:
    """Apply the standard Holm step-down family-wise error procedure.

    The returned ``original_index`` maps each sorted row back to its input
    position.  Threshold comparisons are strict, as specified for this audit.
    """
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Every p-value must be finite and in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    order = np.argsort(values, kind="stable")
    family_size = len(values)
    continue_rejecting = True
    rows: list[dict[str, Any]] = []
    for rank, original_index in enumerate(order, start=1):
        threshold = alpha / (family_size - rank + 1)
        individually_below_threshold = bool(values[original_index] < threshold)
        passes = bool(continue_rejecting and individually_below_threshold)
        if not passes:
            continue_rejecting = False
        rows.append(
            {
                "original_index": int(original_index),
                "raw_p_value": float(values[original_index]),
                "holm_rank": rank,
                "holm_adjusted_threshold": float(threshold),
                "passes_holm_correction": passes,
            }
        )
    return pd.DataFrame(rows)


def capture_established_bootstrap_distribution(
    strategies: pd.DataFrame,
    benchmark: pd.Series,
    config: BootstrapConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    """Call the established bootstrap and capture its discarded sample array.

    ``bootstrap_mean_seed_sharpe`` computes its interval with exactly one call
    to ``np.percentile(samples, [2.5, 97.5])``.  Temporarily wrapping that call
    exposes the same in-memory samples without modifying or reimplementing the
    bootstrap's block generation, paired indexing, Sharpe statistic, or RNG.
    """
    original_percentile = np.percentile
    captured: dict[str, np.ndarray] = {}

    def capture_percentile(values, q, *args, **kwargs):
        sample_array = np.asarray(values, dtype=np.float64)
        if list(q) == [2.5, 97.5]:
            if "samples" in captured:
                raise RuntimeError("Established bootstrap made multiple CI percentile calls")
            captured["samples"] = sample_array.copy()
        return original_percentile(values, q, *args, **kwargs)

    with patch.object(audit_statistics.np, "percentile", new=capture_percentile):
        result = audit_statistics.bootstrap_mean_seed_sharpe(
            strategies, benchmark, config
        )

    if "samples" not in captured:
        raise RuntimeError("Could not capture the established bootstrap distribution")
    samples = captured["samples"]
    if len(samples) != config.n_repetitions:
        raise RuntimeError(
            f"Captured {len(samples)} bootstrap draws; expected {config.n_repetitions}"
        )
    reproduced = original_percentile(samples, [2.5, 97.5])
    if not np.allclose(
        reproduced,
        [result["ci_lower_95"], result["ci_upper_95"]],
        rtol=0.0,
        atol=CI_ATOL,
    ):
        raise RuntimeError("Captured distribution does not reproduce its returned CI")
    return result, samples


def _cost_name(cost: float) -> str:
    return f"cost_{cost:g}"


def _load_daily_returns(path: Path) -> pd.Series:
    if not path.exists():
        raise RuntimeError(f"Required completed return path is missing: {path}")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "daily_return" not in frame:
        raise RuntimeError(f"daily_return column is missing: {path}")
    result = frame["daily_return"]
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise RuntimeError(f"Return dates are duplicated or unordered: {path}")
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite return found: {path}")
    return result


def _zslope_returns(
    ticker: str,
    model: str,
    fold: str,
    horizon: int,
    cost: float,
    seed: int,
) -> pd.Series:
    path = (
        ZSLOPE_RESULTS_ROOT
        / ticker.lower()
        / model
        / fold
        / f"horizon_{horizon}"
        / _cost_name(cost)
        / f"seed_{seed}"
        / "daily_returns.csv"
    )
    return _load_daily_returns(path)


def _buy_hold_returns(
    ticker: str, fold: str, horizon: int, cost: float
) -> pd.Series:
    path = (
        PARENT_RESULTS_ROOT
        / ticker.lower()
        / "buy_hold"
        / fold
        / f"horizon_{horizon}"
        / _cost_name(cost)
        / "seed_NA"
        / "daily_returns.csv"
    )
    return _load_daily_returns(path)


def _stitch(parts: list[pd.Series], label: str) -> pd.Series:
    result = pd.concat(parts)
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise RuntimeError(f"Stitched dates overlap or are unordered: {label}")
    return result


def load_stitched_inputs(
    ticker: str,
    model: str,
    horizon: int,
    cost: float,
    folds: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load exactly the policy and buy-hold series used by the z-slope summary."""
    benchmark = _stitch(
        [_buy_hold_returns(ticker, fold, horizon, cost) for fold in folds],
        f"{ticker}/buy_hold/horizon_{horizon}",
    )
    policies = pd.DataFrame(
        {
            str(seed): _stitch(
                [
                    _zslope_returns(ticker, model, fold, horizon, cost, seed)
                    for fold in folds
                ],
                f"{ticker}/{model}/horizon_{horizon}/seed_{seed}",
            )
            for seed in seeds
        }
    )
    if not policies.index.equals(benchmark.index):
        raise RuntimeError("Policy and buy-hold stitched dates are not identical")
    return policies, benchmark


def load_analysis_design(
    config_path: Path, stitched_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, BootstrapConfig]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stitched = pd.read_csv(stitched_path)
    required_columns = {
        "ticker",
        "model",
        "horizon",
        "cost",
        "folds",
        "n_folds",
        "mean_delta_sharpe",
        "ci_lower_95",
        "ci_upper_95",
        "n_seeds",
        "n_observations",
        "method",
    }
    missing = required_columns - set(stitched.columns)
    if missing:
        raise RuntimeError(f"stitched_significance.csv is missing columns: {sorted(missing)}")
    if len(stitched) != EXPECTED_COMPARISONS:
        raise RuntimeError(
            f"Holm family must contain exactly {EXPECTED_COMPARISONS} rows; found {len(stitched)}"
        )
    identities = stitched[["ticker", "model", "horizon"]]
    if identities.duplicated().any():
        raise RuntimeError("Duplicate (ticker, model, horizon) comparisons found")
    if set(stitched["model"]) != set(EXPECTED_MODELS):
        raise RuntimeError("Unexpected model in the 28-comparison family")
    if set(stitched["horizon"].astype(int)) != set(EXPECTED_HORIZONS):
        raise RuntimeError("Unexpected horizon in the 28-comparison family")
    if set(stitched["ticker"]) != set(config["tickers"]):
        raise RuntimeError("Ticker family differs between config and stitched summary")
    if set(stitched["method"]) != {EXPECTED_METHOD}:
        raise RuntimeError("Existing summary was not produced by the expected paired method")
    if set(stitched["cost"].astype(float)) != {float(config["primary_cost"])}:
        raise RuntimeError("Existing summary cost differs from config primary_cost")
    expected_fold_text = ",".join(EXPECTED_FOLDS)
    if set(stitched["folds"]) != {expected_fold_text} or set(stitched["n_folds"]) != {4}:
        raise RuntimeError("Existing summary does not use the approved four stitched folds")
    if set(stitched["n_seeds"]) != {len(config["seeds"])}:
        raise RuntimeError("Existing summary seed count differs from config")

    bootstrap_keys = {"block_length", "n_repetitions", "seed"}
    if set(config["bootstrap"]) != bootstrap_keys:
        raise RuntimeError("Bootstrap config must contain exactly block_length/n_repetitions/seed")
    bootstrap = BootstrapConfig(**config["bootstrap"])
    if bootstrap.n_repetitions != 5000:
        raise RuntimeError("This audit requires the original 5,000 bootstrap repetitions")
    return config, stitched, bootstrap


def compute_raw_p_values(
    config: dict[str, Any], stitched: pd.DataFrame, bootstrap: BootstrapConfig
) -> pd.DataFrame:
    """Reproduce every existing CI and retain each corresponding distribution."""
    rows = []
    folds = list(config["folds"])
    seeds = [int(seed) for seed in config["seeds"]]
    for existing in stitched.itertuples(index=False):
        policies, benchmark = load_stitched_inputs(
            str(existing.ticker),
            str(existing.model),
            int(existing.horizon),
            float(existing.cost),
            folds,
            seeds,
        )
        result, samples = capture_established_bootstrap_distribution(
            policies, benchmark, bootstrap
        )
        expected = np.asarray(
            [
                float(existing.mean_delta_sharpe),
                float(existing.ci_lower_95),
                float(existing.ci_upper_95),
            ]
        )
        actual = np.asarray(
            [
                float(result["mean_delta_sharpe"]),
                float(result["ci_lower_95"]),
                float(result["ci_upper_95"]),
            ]
        )
        if not np.allclose(actual, expected, rtol=0.0, atol=CI_ATOL):
            raise RuntimeError(
                "Bootstrap reproduction differs from stitched_significance.csv for "
                f"{existing.ticker}/{existing.model}/horizon_{existing.horizon}: "
                f"existing={expected.tolist()}, reproduced={actual.tolist()}"
            )
        if int(result["n_observations"]) != int(existing.n_observations):
            raise RuntimeError("Reproduced observation count differs from existing summary")
        rows.append(
            {
                "ticker": str(existing.ticker),
                "model": str(existing.model),
                "horizon": int(existing.horizon),
                "mean_delta_sharpe": float(existing.mean_delta_sharpe),
                "ci_lower_95": float(existing.ci_lower_95),
                "ci_upper_95": float(existing.ci_upper_95),
                "raw_p_value": two_sided_bootstrap_p_value(samples),
            }
        )
    return pd.DataFrame(rows)


def apply_holm_to_results(raw: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    correction = holm_step_down(raw["raw_p_value"], alpha=alpha)
    sorted_raw = raw.iloc[correction["original_index"].to_numpy()].reset_index(drop=True)
    result = pd.concat(
        [
            sorted_raw,
            correction[
                ["holm_rank", "holm_adjusted_threshold", "passes_holm_correction"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    return result[
        [
            "ticker",
            "model",
            "horizon",
            "mean_delta_sharpe",
            "ci_lower_95",
            "ci_upper_95",
            "raw_p_value",
            "holm_rank",
            "holm_adjusted_threshold",
            "passes_holm_correction",
        ]
    ]


def format_report(results: pd.DataFrame, alpha: float = 0.05) -> str:
    target = results[
        (results["ticker"] == TARGET_TICKER)
        & (results["model"] == TARGET_MODEL)
        & (results["horizon"] == TARGET_HORIZON)
    ]
    if len(target) != 1:
        raise RuntimeError("The requested TSLA six-state horizon-10 row is not unique")
    target_row = target.iloc[0]
    passed_count = int(results["passes_holm_correction"].sum())
    lines = [
        "# z-slope Holm-Bonferroni 多重比较校正",
        "",
        f"检验族包含全部 {len(results)} 个 stitched z-slope vs buy-and-hold 比较；"
        f"族错误率 alpha = {alpha:g}。",
        "原始 p-value 使用指定的双尾符号比例公式，并直接取自复现各行既有 95% CI 的"
        "同一份 5,000 次 paired-time-block 重抽样分布。",
        "",
        "## 28 个原始 p-value 的完整排序",
        "",
        "| 排名 | ticker | model | horizon | mean ΔSharpe | 95% CI | raw p-value | Holm 门槛 | 通过 |",
        "|---:|---|---|---:|---:|---|---:|---:|:---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.holm_rank} | {row.ticker} | {row.model} | {row.horizon} | "
            f"{row.mean_delta_sharpe:.6f} | [{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}] | "
            f"{row.raw_p_value:.6f} | {row.holm_adjusted_threshold:.8f} | "
            f"{'是' if row.passes_holm_correction else '否'} |"
        )
    lines.extend(
        [
            "",
            "## TSLA 最终判定",
            "",
            f"`{TARGET_TICKER}` / `{TARGET_MODEL}` / horizon={TARGET_HORIZON} 在 "
            f"{len(results)} 项中排第 {int(target_row['holm_rank'])}；raw p-value = "
            f"{float(target_row['raw_p_value']):.6f}，对应 Holm 门槛 = "
            f"{float(target_row['holm_adjusted_threshold']):.8f}。该结果"
            f"{'通过' if bool(target_row['passes_holm_correction']) else '未通过'} "
            "Holm-Bonferroni 校正。",
            "",
            "## 校正后的总体结果",
            "",
            f"Holm-Bonferroni 校正后仍显著的组合共有 {passed_count}/{len(results)} 个。"
            if passed_count
            else f"Holm-Bonferroni 校正后仍显著的组合为 0/{len(results)} 个。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_analysis(
    config_path: Path,
    stitched_path: Path,
    summary_path: Path,
    report_path: Path,
    alpha: float = 0.05,
) -> pd.DataFrame:
    config, stitched, bootstrap = load_analysis_design(config_path, stitched_path)
    raw = compute_raw_p_values(config, stitched, bootstrap)
    corrected = apply_holm_to_results(raw, alpha=alpha)
    _atomic_write_csv(corrected, summary_path)
    _atomic_write_text(format_report(corrected, alpha=alpha), report_path)
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stitched", type=Path, default=DEFAULT_STITCHED)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    results = run_analysis(
        args.config,
        args.stitched,
        args.summary_output,
        args.report_output,
        alpha=args.alpha,
    )
    print(f"Wrote {len(results)} Holm-corrected rows to {args.summary_output}")
    print(f"Wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
