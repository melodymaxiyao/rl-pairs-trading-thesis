"""Generate fold_02 TRAIN diagnostics for causal adjusted-log-price smoothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "causal_kernel_smoothing.json"
sys.path.insert(0, str(REPO_ROOT))

from experiments.single_stock.run_single_stock_walkforward import (  # noqa: E402
    resolve_effective_split,
)
from experiments.turning_point_detection.new_stage1.kernel_smoothing import (  # noqa: E402
    add_causal_kernel_smoothing,
    kernel_mean_lag,
)
from fyp_rl.data import load_ohlcv  # noqa: E402


RAW_COLOR = "#475569"
SMOOTH_COLOR = "#007F73"
RESIDUAL_COLOR = "#C2413B"
ZERO_COLOR = "#94A3B8"


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("method") != "causal_kernel_regression":
        raise ValueError("Expected method=causal_kernel_regression")
    if config.get("kernel") != "gaussian":
        raise ValueError("Only the Gaussian kernel is implemented")
    if config.get("input_series") != "log_adjusted_close":
        raise ValueError("Expected input_series=log_adjusted_close")
    return config


def _load_train_frame(ticker: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_config = json.loads(
        _repo_path(config["split_source"]).read_text(encoding="utf-8")
    )
    fold_name = config["diagnostic_fold"]
    split = resolve_effective_split(
        ticker,
        fold_name,
        split_config["folds"][fold_name],
        split_config["tsla_min_train_start"],
    )
    raw_start = (
        pd.Timestamp(split["actual_train_start"])
        - pd.Timedelta(days=int(config["feature_warmup_calendar_days"]))
    ).strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(
        ticker,
        raw_start,
        split["train_end"],
        raw_dir=_repo_path(config["raw_dir"]),
    )
    train = pd.DataFrame({"log_adjusted_close": np.log(ohlcv["Close"])})
    train = train.loc[split["actual_train_start"] : split["train_end"]]
    train = train.dropna(subset=["log_adjusted_close"])
    if train.empty:
        raise ValueError(f"No finite TRAIN adjusted-log-price observations for {ticker}")
    return train, split


def estimate_smooth_lag_days(frame: pd.DataFrame, *, max_lag_days: int) -> int:
    """Find the peak overlap-normalized cross-correlation of price levels.

    Search integer trading-day lags within the kernel's finite support,
    ``[-(W-1), W-1]``. At positive lag L, compare p[t] with smooth[t+L],
    so positive values mean the smooth lags the raw series. Each overlapping
    pair is separately centered and normalized (Pearson correlation).
    Ties favor the smallest absolute lag. No differencing is applied.
    """
    raw = frame["log_adjusted_close"].to_numpy(dtype=float)
    smooth = frame["p_smooth"].to_numpy(dtype=float)
    correlations = []
    for lag in range(-max_lag_days, max_lag_days + 1):
        if lag > 0:
            left, right = raw[:-lag], smooth[lag:]
        elif lag < 0:
            left, right = raw[-lag:], smooth[:lag]
        else:
            left, right = raw, smooth
        finite = np.isfinite(left) & np.isfinite(right)
        left, right = left[finite], right[finite]
        if len(left) < 2:
            continue
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator > 0:
            correlations.append((float(np.dot(left, right) / denominator), lag))
    if not correlations:
        raise ValueError("Lag estimation requires nonconstant overlapping series")
    return max(correlations, key=lambda pair: (pair[0], -abs(pair[1])))[1]


def summarize_smoothing(
    ticker: str, bandwidth_h: float, frame: pd.DataFrame, *, window_multiplier: int
) -> dict[str, Any]:
    residual = frame["smoothing_residual"]
    return {
        "ticker": ticker,
        "h": bandwidth_h,
        "residual_std": float(residual.std(ddof=1)),
        "residual_lag1_autocorr": float(residual.autocorr(lag=1)),
        "smooth_lag_days_estimate": estimate_smooth_lag_days(
            frame, max_lag_days=int(window_multiplier * bandwidth_h) - 1
        ),
        "kernel_mean_lag": kernel_mean_lag(bandwidth_h, window_multiplier),
    }


def _plot_group(
    tickers: list[str],
    *,
    frames: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    bandwidth_h: float,
    window_multiplier: int,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(tickers),
        3,
        figsize=(18, 3.2 * len(tickers)),
        sharex="col",
        sharey=False,
        squeeze=False,
    )
    for row, ticker in enumerate(tickers):
        frame, split = frames[ticker]
        series = (
            ("log_adjusted_close", RAW_COLOR, "Log adjusted close + causal smooth"),
            ("p_smooth", SMOOTH_COLOR, "Causal Gaussian smooth"),
            ("smoothing_residual", RESIDUAL_COLOR, "Residual"),
        )
        for column, (field, color, title) in enumerate(series):
            axis = axes[row, column]
            axis.plot(frame.index, frame[field], color=color, linewidth=0.9)
            if field == "log_adjusted_close":
                axis.lines[0].set_label("Log adjusted close")
                axis.plot(
                    frame.index, frame["p_smooth"], color=SMOOTH_COLOR,
                    linewidth=0.65, label="Causal smooth",
                )
                axis.legend(loc="lower right", fontsize=7, frameon=False)
            if field == "smoothing_residual":
                axis.axhline(0.0, color=ZERO_COLOR, linewidth=0.8, linestyle="--")
                axis.text(
                    0.02, 0.96,
                    f"Residual SD = {frame[field].std(ddof=1):.5f} log units",
                    transform=axis.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
            axis.grid(axis="y", color="#E2E8F0", linewidth=0.55)
            axis.set_title(title, fontsize=10)
            axis.set_ylabel(
                f"{ticker}\nLog-price deviation (log units)"
                if field == "smoothing_residual"
                else f"{ticker}\nLog adjusted price (log units)"
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.xaxis.set_major_locator(mdates.YearLocator(2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[row, 0].text(
            0.01,
            0.96,
            f"TRAIN {split['actual_train_start']} to {split['train_end']}",
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=8,
            color="#64748B",
        )
    fig.suptitle(
        "Strictly causal one-sided Gaussian smoothing of log adjusted close\n"
        f"h={bandwidth_h:g}, trailing window={window_multiplier * bandwidth_h:g} trading days; "
        "each estimate uses t and earlier observations only",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "All panels use log-price units; a 0.01 log-price deviation is approximately 1%. "
        "Each panel has an independent y-axis scale.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.035, 1.0, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Plot was not written: {output_path}")
    return output_path


def main() -> None:
    config = load_config()
    output_dir = _repo_path(config["output_dir"])
    for group in (1, 2):
        original = output_dir / f"smoothing_diagnostic_group{group}.png"
        preserved = output_dir / f"smoothing_diagnostic_z20_group{group}.png"
        if original.exists():
            if preserved.exists():
                raise FileExistsError(f"Refusing to overwrite preserved z20 plot: {preserved}")
            original.rename(preserved)
    tickers = list(config["tickers_group1"]) + list(config["tickers_group2"])
    raw_frames = {ticker: _load_train_frame(ticker, config) for ticker in tickers}
    outputs = []
    summary = []
    for bandwidth_h in config["bandwidth_grid"]:
        frames = {
            ticker: (
                add_causal_kernel_smoothing(
                    frame, bandwidth_h=bandwidth_h,
                    window_multiplier=config["window_multiplier"],
                ),
                split,
            )
            for ticker, (frame, split) in raw_frames.items()
        }
        for ticker, (frame, _) in frames.items():
            summary.append(summarize_smoothing(
                ticker, bandwidth_h, frame,
                window_multiplier=config["window_multiplier"],
            ))
        for group in (1, 2):
            outputs.append(_plot_group(
                list(config[f"tickers_group{group}"]),
                frames=frames, bandwidth_h=bandwidth_h,
                window_multiplier=config["window_multiplier"],
                output_path=output_dir / f"smoothing_diagnostic_logprice_h{bandwidth_h:g}_group{group}.png",
            ))
    summary_path = output_dir.parent / "tables" / "smoothing_diagnostic_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(summary_path, index=False, float_format="%.12g")
    outputs.append(summary_path)
    for output in outputs:
        print(output.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
