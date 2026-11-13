"""Strictly causal one-sided kernel regression smoothing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_gaussian_kernel_smooth(
    values: pd.Series,
    *,
    bandwidth_h: float,
    window_days: int,
) -> pd.Series:
    """Smooth each value using only the current and trailing observations.

    At position ``t``, the estimate is the Gaussian-kernel weighted average
    over ``s in [t-window_days+1, t]`` with lag ``t-s``. Missing observations
    are omitted from both numerator and denominator.
    """
    if not np.isfinite(bandwidth_h) or bandwidth_h <= 0.0:
        raise ValueError("bandwidth_h must be finite and positive")
    if int(window_days) != window_days or window_days < 1:
        raise ValueError("window_days must be a positive integer")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    source = numeric.to_numpy(dtype=float)
    smoothed = np.full(len(source), np.nan, dtype=float)
    for position in range(len(source)):
        start = max(0, position - int(window_days) + 1)
        window = source[start : position + 1]
        finite = np.isfinite(window)
        if not finite.any():
            continue
        lags = position - np.arange(start, position + 1, dtype=float)
        weights = np.exp(-0.5 * np.square(lags / float(bandwidth_h)))
        smoothed[position] = float(
            np.dot(weights[finite], window[finite]) / weights[finite].sum()
        )
    return pd.Series(smoothed, index=values.index, name="p_smooth")


def add_causal_kernel_smoothing(
    frame: pd.DataFrame,
    *,
    bandwidth_h: float,
    window_multiplier: int = 3,
) -> pd.DataFrame:
    """Append a causal smooth and residual in adjusted-log-price units.

    The trailing window is derived as ``window_multiplier * bandwidth_h``.
    Startup estimates use the available prefix until the window is full.
    """
    if "log_adjusted_close" not in frame.columns:
        raise ValueError("Input frame must contain log_adjusted_close")
    if not np.isfinite(window_multiplier) or window_multiplier <= 0:
        raise ValueError("window_multiplier must be finite and positive")
    window_days = window_multiplier * bandwidth_h
    result = frame.copy()
    result["p_smooth"] = causal_gaussian_kernel_smooth(
        result["log_adjusted_close"],
        bandwidth_h=bandwidth_h,
        window_days=window_days,
    )
    result["smoothing_residual"] = result["log_adjusted_close"] - result["p_smooth"]
    return result


def kernel_mean_lag(bandwidth_h: float, window_multiplier: int) -> float:
    """Mean lag of the existing one-sided local-constant Gaussian kernel."""
    window = window_multiplier * bandwidth_h
    if bandwidth_h <= 0 or int(window) != window or window < 1:
        raise ValueError("Positive bandwidth and integral positive window required")
    j = np.arange(int(window), dtype=float)
    weights = np.exp(-0.5 * (j / bandwidth_h) ** 2)
    return float(np.dot(j, weights) / weights.sum())


def local_linear_slope_weights(
    bandwidth_h: float, *, side: str, window_multiplier: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return offsets x and fixed WLS slope weights l, in offset order.

    Causal: x = 0, -1, ..., -(W-1), W = multiplier*h.
    Bilateral: x = -radius, ..., radius, radius = multiplier*h.
    b = sum(l*p[t+x]); sum(l)=0 and sum(l*x)=1.
    """
    extent = window_multiplier * bandwidth_h
    if not np.isfinite(bandwidth_h) or bandwidth_h <= 0:
        raise ValueError("bandwidth_h must be finite and positive")
    if not np.isfinite(extent) or int(extent) != extent or extent < 1:
        raise ValueError("Window extent must be a positive integer")
    if side == "causal":
        x = -np.arange(int(extent), dtype=float)
    elif side == "bilateral":
        x = np.arange(-int(extent), int(extent) + 1, dtype=float)
    else:
        raise ValueError("side must be causal or bilateral")
    w = np.exp(-0.5 * (x / bandwidth_h) ** 2)
    xbar = np.dot(w, x) / w.sum()
    denominator = np.dot(w, (x - xbar) ** 2)
    if denominator <= 0:
        raise ValueError("At least two distinct offsets required for slope estimation")
    return x, w * (x - xbar) / denominator


def _fixed_weight_slope(values: pd.Series, weights: np.ndarray, *, center: int) -> pd.Series:
    source = values.to_numpy(dtype=float)
    result = np.full(len(source), np.nan)
    width = len(weights)
    if len(source) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(source, width)
        # Subtract a window value to preserve constant-series zero exactly and
        # reduce cancellation. This is algebraically the same fixed-weight WLS.
        differences = windows - windows[:, center, None]
        slopes = np.einsum("ij,j->i", differences, weights, optimize=False)
        slopes[~np.isfinite(windows).all(axis=1)] = np.nan
        result[center: center + len(slopes)] = slopes
    return pd.Series(result, index=values.index, name="slope")


def causal_local_linear_slope(
    values: pd.Series, *, bandwidth_h: float, window_multiplier: int
) -> pd.Series:
    """Strictly causal WLS slope; incomplete trailing windows produce NaN."""
    _, weights = local_linear_slope_weights(
        bandwidth_h, side="causal", window_multiplier=window_multiplier
    )
    return _fixed_weight_slope(values, weights[::-1], center=len(weights) - 1)


def bilateral_local_linear_slope(
    values: pd.Series, *, bandwidth_h: float, radius_multiplier: int
) -> pd.Series:
    """Retrospective evaluation reference ONLY; incomplete windows are NaN."""
    _, weights = local_linear_slope_weights(
        bandwidth_h, side="bilateral", window_multiplier=radius_multiplier
    )
    return _fixed_weight_slope(values, weights, center=len(weights) // 2)
