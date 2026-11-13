"""Visual validation plots for the two causal turn detectors.

This is a post-signal-diagnostic, pre-stage-2 inspection aid.  It never trains
an agent and never participates in the statistical gate.  Triangle markers are
drawn at the inferred turn's source date; the source becomes observable to the
strategy only ``confirmation_lag_days`` trading rows later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.single_stock.features_bocpd import (
    CONFIRMED_PEAK,
    CONFIRMED_TROUGH,
    build_bocpd_features,
)
from experiments.single_stock.features_confirmed_turns import (
    build_confirmed_turn_features,
)
from experiments.single_stock.run_signal_diagnostic_bocpd import (
    _load_base_features,
    load_bocpd_config,
)
from experiments.single_stock.run_signal_diagnostic_confirmed_turns import (
    load_confirmed_turns_config,
)
from experiments.single_stock.run_single_stock_walkforward import (
    resolve_effective_split,
)


LO_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_confirmed_turns.json"
BOCPD_CONFIG = EXPERIMENT_DIR / "configs" / "single_stock_bocpd.json"
LO_OUTPUT = EXPERIMENT_DIR / "confirmed_turns" / "diagnostic_plots"
BOCPD_OUTPUT = EXPERIMENT_DIR / "bocpd" / "diagnostic_plots"
TEST_CONTEXT_CALENDAR_DAYS = 45
REPORT_START = "<!-- TURN_VISUAL_VALIDATION_START -->"
REPORT_END = "<!-- TURN_VISUAL_VALIDATION_END -->"
README_START = "<!-- TURN_VISUAL_VALIDATION_OVERVIEW_START -->"
README_END = "<!-- TURN_VISUAL_VALIDATION_OVERVIEW_END -->"


def _price_equivalent(base: pd.DataFrame, z_signal: pd.Series) -> pd.Series:
    """Map a z20-scale signal back to the contemporaneous price scale."""
    if not z_signal.index.equals(base.index):
        raise ValueError("z_signal and base features must share an index")
    return (base["sma20"] + z_signal * base["std20"]).rename("price_equivalent")


def causal_map_segment_mean(z20: pd.Series, run_length_mode: pd.Series) -> pd.Series:
    """Causal mean of the MAP run segment ending at every observation.

    In sdt's posterior, a modal run length of 1 after an update means that the
    current observation is the first member of the new segment.  Therefore the
    segment start is ``t - run_length + 1``.  Only rows through ``t`` are read.
    """
    if not z20.index.equals(run_length_mode.index):
        raise ValueError("z20 and run_length_mode must share an index")
    values = z20.to_numpy(dtype=np.float64)
    modes = run_length_mode.to_numpy(dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for loc, (value, mode) in enumerate(zip(values, modes)):
        if not np.isfinite(value) or not np.isfinite(mode):
            continue
        run_length = max(1, int(round(mode)))
        start = max(0, loc - run_length + 1)
        segment = values[start : loc + 1]
        finite = segment[np.isfinite(segment)]
        if len(finite):
            result[loc] = float(finite.mean())
    return pd.Series(result, index=z20.index, name="bocpd_map_segment_mean_z20")


def _event_points(
    base: pd.DataFrame,
    feature: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    source_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_mask = (
        (feature.index >= period_start)
        & (feature.index <= period_end)
        & feature["confirmed_turn_state"].isin([CONFIRMED_PEAK, CONFIRMED_TROUGH])
    )
    events = feature.loc[decision_mask, ["confirmed_turn_state", source_column]].copy()
    events["decision_date"] = events.index
    events = events.dropna(subset=[source_column])
    events["source_date"] = pd.to_datetime(events[source_column])
    events["source_close"] = events["source_date"].map(base["Close"])
    events = events.dropna(subset=["source_close"])
    peaks = events[events["confirmed_turn_state"] == CONFIRMED_PEAK]
    troughs = events[events["confirmed_turn_state"] == CONFIRMED_TROUGH]
    return peaks, troughs


def render_turn_plot(
    *,
    base: pd.DataFrame,
    feature: pd.DataFrame,
    smooth_price: pd.Series,
    ticker: str,
    method_label: str,
    parameter_label: str,
    fold: str,
    window_name: str,
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
    confirmation_lag_days: int,
    source_column: str,
    output_path: Path,
) -> Path:
    """Render one inspectable PNG and return its path."""
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    if window_name not in {"train", "test"}:
        raise ValueError("window_name must be train or test")
    display_start = (
        start - pd.Timedelta(days=TEST_CONTEXT_CALENDAR_DAYS)
        if window_name == "test"
        else start
    )
    display = base.loc[(base.index >= display_start) & (base.index <= end)]
    smooth_display = smooth_price.reindex(display.index)
    if display.empty or display["Close"].dropna().empty:
        raise RuntimeError("No price observations in requested plot window")
    peaks, troughs = _event_points(
        base, feature, start, end, source_column=source_column
    )

    fig, ax = plt.subplots(figsize=(16, 7.5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.84)
    ax.plot(
        display.index,
        display["Close"],
        color="#9AA3AF",
        linewidth=0.85,
        alpha=0.72,
        label="Adjusted close (raw price scale)",
        zorder=1,
    )
    ax.plot(
        smooth_display.index,
        smooth_display,
        color="#173B67",
        linewidth=1.9,
        alpha=0.95,
        label="Causal smoothed / segmented price-equivalent",
        zorder=2,
    )
    ax.scatter(
        peaks["source_date"],
        peaks["source_close"],
        marker="v",
        s=64,
        color="#C23B3B",
        edgecolor="#FFFFFF",
        linewidth=0.6,
        label=f"Confirmed peak source (known {confirmation_lag_days}d later)",
        zorder=4,
    )
    ax.scatter(
        troughs["source_date"],
        troughs["source_close"],
        marker="^",
        s=64,
        color="#16856B",
        edgecolor="#FFFFFF",
        linewidth=0.6,
        label=f"Confirmed trough source (known {confirmation_lag_days}d later)",
        zorder=4,
    )
    if window_name == "test":
        ax.axvspan(display_start, start, color="#E5E7EB", alpha=0.38, zorder=0)
        ax.axvline(
            start,
            color="#5B6470",
            linewidth=1.15,
            linestyle="--",
            label="Test window starts",
            zorder=3,
        )

    fig.suptitle(
        f"{ticker} — {method_label} | {fold} {window_name} | {parameter_label}",
        x=0.07,
        y=0.975,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.925,
        (
            f"Decision-window events: {len(peaks)} peaks, {len(troughs)} troughs. "
            "Triangles mark source dates, not earlier availability."
        ),
        fontsize=9.5,
        color="#4B5563",
        ha="left",
        va="top",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close / price-equivalent (USD)")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=1 if window_name == "test" else 2,
        fontsize=8.7 if window_name == "test" else 9,
    )
    ax.margins(x=0.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Plot was not written: {output_path}")
    return output_path


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_confirmed_turns_config(LO_CONFIG), load_bocpd_config(BOCPD_CONFIG)


def _base_and_split(
    ticker: str,
    fold: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if fold not in config["folds"]:
        raise ValueError(f"Unknown fold: {fold}")
    split = resolve_effective_split(
        ticker,
        fold,
        config["folds"][fold],
        config["tsla_min_train_start"],
    )
    base = _load_base_features(ticker, split, config)
    return base, split


def generate_lo_plots(
    *,
    ticker: str,
    fold: str,
    k_multiplier: float,
    windows: Iterable[str],
    output_dir: Path = LO_OUTPUT,
) -> list[Path]:
    config, _ = _load_configs()
    allowed = [float(x) for x in config["confirmed_turn_feature"]["k_multipliers"]]
    if float(k_multiplier) not in allowed:
        raise ValueError(f"k_multiplier must be one of {allowed}")
    base, split = _base_and_split(ticker, fold, config)
    feature_config = config["confirmed_turn_feature"]
    turns = build_confirmed_turn_features(
        base["z20"],
        smooth_window=int(feature_config["smooth_window"]),
        vol_lookback_days=int(feature_config["vol_lookback_days"]),
        k_multiplier=float(k_multiplier),
        confirmation_lag_days=int(feature_config["confirmation_lag_days"]),
        local_vol_ddof=int(feature_config["local_vol_ddof"]),
    )
    smooth_price = _price_equivalent(base, turns["z20_smooth"])
    paths = []
    for window in windows:
        start_key = "actual_train_start" if window == "train" else "test_start"
        end_key = "train_end" if window == "train" else "test_end"
        path = output_dir / f"{ticker.upper()}_k{k_multiplier:g}_{fold}_{window}.png"
        paths.append(
            render_turn_plot(
                base=base,
                feature=turns,
                smooth_price=smooth_price,
                ticker=ticker.upper(),
                method_label="Lo-style causal EMA turns",
                parameter_label=f"k={k_multiplier:g}",
                fold=fold,
                window_name=window,
                period_start=split[start_key],
                period_end=split[end_key],
                confirmation_lag_days=int(feature_config["confirmation_lag_days"]),
                source_column="confirmed_turn_source_date",
                output_path=path,
            )
        )
    return paths


def generate_bocpd_plots(
    *,
    ticker: str,
    fold: str,
    hazard_lambda: float,
    windows: Iterable[str],
    output_dir: Path = BOCPD_OUTPUT,
) -> list[Path]:
    _, config = _load_configs()
    feature_config = config["bocpd_feature"]
    allowed = [float(x) for x in feature_config["hazard_rate_lambdas"]]
    if float(hazard_lambda) not in allowed:
        raise ValueError(f"hazard_lambda must be one of {allowed}")
    base, split = _base_and_split(ticker, fold, config)
    model = feature_config["observation_model"]
    turns = build_bocpd_features(
        base["z20"],
        hazard_lambda=float(hazard_lambda),
        probability_threshold=float(
            feature_config["changepoint_prob_threshold_grid"][0]
        ),
        confirmation_lag_days=int(feature_config["confirmation_lag_days"]),
        direction_pre_window_days=int(feature_config["direction_pre_window_days"]),
        observation_model={
            key: model[key] for key in ("name", "mu", "kappa", "alpha", "beta")
        },
        dependency_engine=config["dependency"]["engine"],
        expected_dependency_version=config["dependency"]["version"],
    )
    map_mean = causal_map_segment_mean(base["z20"], turns["bocpd_run_length_mode"])
    smooth_price = _price_equivalent(base, map_mean)
    paths = []
    for window in windows:
        start_key = "actual_train_start" if window == "train" else "test_start"
        end_key = "train_end" if window == "train" else "test_end"
        path = output_dir / f"{ticker.upper()}_lambda{hazard_lambda:g}_{fold}_{window}.png"
        paths.append(
            render_turn_plot(
                base=base,
                feature=turns,
                smooth_price=smooth_price,
                ticker=ticker.upper(),
                method_label="BOCPD MAP-segment turns",
                parameter_label=f"lambda={hazard_lambda:g}, P(r_t=d)>0.5",
                fold=fold,
                window_name=window,
                period_start=split[start_key],
                period_end=split[end_key],
                confirmation_lag_days=int(feature_config["confirmation_lag_days"]),
                source_column="bocpd_turn_source_date",
                output_path=path,
            )
        )
    return paths


def _replace_or_append(text: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        return prefix + block + suffix
    return text.rstrip() + "\n\n" + block + "\n"


def update_reports(paths: list[Path], ticker: str, fold: str) -> None:
    grouped = {
        "confirmed_turns": [p for p in paths if p.parent.parent.name == "confirmed_turns"],
        "bocpd": [p for p in paths if p.parent.parent.name == "bocpd"],
    }
    for method, method_paths in grouped.items():
        if not method_paths:
            continue
        report = (
            EXPERIMENT_DIR / "results" / method / "signal_diagnostic" / "report.md"
        )
        if not report.exists():
            raise RuntimeError(f"Signal diagnostic report is missing: {report}")
        method_title = "Lo-style causal turns" if method == "confirmed_turns" else "BOCPD turns"
        lines = [
            "## 首轮转折点可视化验证",
            "",
            f"{ticker.upper()} / {fold}；本图仅供人工目视检查，不改变统计 gate。",
            "三角形位于转折来源日，但标签仅在三个交易日后的决策日可用。测试图左侧灰色区域为测试起点前的训练期上下文。",
            "",
        ]
        for path in method_paths:
            relative = Path("../../../") / method / "diagnostic_plots" / path.name
            window = "训练窗口" if path.stem.endswith("_train") else "测试窗口"
            lines.extend([f"### {method_title}：{window}", "", f"![{path.stem}]({relative.as_posix()})", ""])
        report.write_text(
            _replace_or_append(
                report.read_text(encoding="utf-8"),
                REPORT_START,
                REPORT_END,
                "\n".join(lines),
            ),
            encoding="utf-8",
        )

    readme = EXPERIMENT_DIR / "README.md"
    overview = [
        "## 确认转折点首轮可视化验证",
        "",
        f"首轮仅生成 {ticker.upper()} / {fold} 的默认参数图；本环节只供人工目视检查，不作为自动 gate，也未运行第二阶段。",
        "三角形标在转折来源日，实际信号在 d=3 个交易日后才可用。测试图左侧灰色带是测试起点前的训练期上下文，虚线是测试起点。",
        "",
    ]
    for path in paths:
        relative = path.relative_to(EXPERIMENT_DIR)
        overview.extend([f"### {path.stem}", "", f"![{path.stem}]({relative.as_posix()})", ""])
    readme.write_text(
        _replace_or_append(
            readme.read_text(encoding="utf-8"),
            README_START,
            README_END,
            "\n".join(overview),
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--fold", default="fold_02")
    parser.add_argument("--methods", nargs="+", choices=["lo", "bocpd"], default=["lo", "bocpd"])
    parser.add_argument("--windows", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--lo-k", type=float, default=1.5)
    parser.add_argument("--bocpd-lambda", type=float, default=100.0)
    parser.add_argument("--no-update-reports", action="store_true")
    args = parser.parse_args()

    paths: list[Path] = []
    if "lo" in args.methods:
        paths.extend(
            generate_lo_plots(
                ticker=args.ticker,
                fold=args.fold,
                k_multiplier=args.lo_k,
                windows=args.windows,
            )
        )
    if "bocpd" in args.methods:
        paths.extend(
            generate_bocpd_plots(
                ticker=args.ticker,
                fold=args.fold,
                hazard_lambda=args.bocpd_lambda,
                windows=args.windows,
            )
        )
    expected = len(args.methods) * len(args.windows)
    if len(paths) != expected or not all(path.exists() for path in paths):
        raise RuntimeError(f"Expected {expected} plots, created {len(paths)}")
    if not args.no_update_reports:
        update_reports(paths, args.ticker, args.fold)
    print(json.dumps({"plots": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
