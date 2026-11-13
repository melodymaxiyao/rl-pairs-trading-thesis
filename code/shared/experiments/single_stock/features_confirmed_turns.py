"""Causal, volatility-adaptive confirmed turning-point features.

The smoother and local volatility at row t use rows <= t only.  A geometric
extremum at source row c is exposed to the decision process only at row
t = c + confirmation_lag_days, after c+1 is observable.  No label is
back-filled onto c, which would create look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NO_CONFIRMED_TURN = 0
CONFIRMED_PEAK = 1
CONFIRMED_TROUGH = 2
TURN_STATE_LABELS = {
    NO_CONFIRMED_TURN: "no_confirmed_turn",
    CONFIRMED_PEAK: "at_confirmed_peak",
    CONFIRMED_TROUGH: "at_confirmed_trough",
}


def causal_ema(z20: pd.Series, smooth_window: int) -> pd.Series:
    """Single-sided EMA with a full-window warm-up requirement."""
    if smooth_window <= 1:
        raise ValueError("smooth_window must be greater than 1")
    values = pd.to_numeric(z20, errors="coerce")
    return values.ewm(
        span=smooth_window,
        adjust=False,
        min_periods=smooth_window,
    ).mean()


def rolling_local_volatility(
    z20: pd.Series, vol_lookback_days: int, ddof: int = 0
) -> pd.Series:
    """Causal rolling standard deviation of z20."""
    if vol_lookback_days <= 1:
        raise ValueError("vol_lookback_days must be greater than 1")
    if ddof < 0 or ddof >= vol_lookback_days:
        raise ValueError("ddof must be in [0, vol_lookback_days)")
    values = pd.to_numeric(z20, errors="coerce")
    return values.rolling(
        vol_lookback_days, min_periods=vol_lookback_days
    ).std(ddof=ddof)


def _candidate_type(previous: float, current: float, following: float) -> int:
    if not np.isfinite([previous, current, following]).all():
        return NO_CONFIRMED_TURN
    if current > previous and current > following:
        return CONFIRMED_PEAK
    if current < previous and current < following:
        return CONFIRMED_TROUGH
    return NO_CONFIRMED_TURN


def confirm_turns_from_smoothed(
    z20_smooth: pd.Series,
    swing_threshold: pd.Series,
    confirmation_lag_days: int,
) -> pd.DataFrame:
    """Confirm alternating extrema causally and emit labels at decision time.

    The first geometric extremum initializes an internal opposite-extremum
    anchor but is not emitted because no prior opposite anchor exists against
    which to validate its swing.  A failed opposite candidate is ignored.  A
    later, more extreme candidate of the same type may replace the anchor;
    this is the causal zig-zag convention needed to avoid a deadlocked first
    turn while still applying the adaptive threshold to every emitted label.
    """
    if confirmation_lag_days < 1:
        raise ValueError("confirmation_lag_days must be at least 1")
    if not z20_smooth.index.equals(swing_threshold.index):
        raise ValueError("z20_smooth and swing_threshold must share an index")
    if not z20_smooth.index.is_monotonic_increasing or z20_smooth.index.has_duplicates:
        raise ValueError("feature index must be unique and increasing")

    smooth = z20_smooth.to_numpy(dtype=np.float64)
    thresholds = swing_threshold.to_numpy(dtype=np.float64)
    n = len(smooth)
    states = np.full(n, NO_CONFIRMED_TURN, dtype=np.int8)
    source_locs = np.full(n, -1, dtype=np.int64)
    swing_amplitudes = np.full(n, np.nan, dtype=np.float64)
    applied_thresholds = np.full(n, np.nan, dtype=np.float64)

    anchor_type = NO_CONFIRMED_TURN
    anchor_value = np.nan
    anchor_loc = -1

    first_decision_loc = confirmation_lag_days + 1
    for decision_loc in range(first_decision_loc, n):
        source_loc = decision_loc - confirmation_lag_days
        if source_loc + 1 >= n:
            continue
        candidate_type = _candidate_type(
            smooth[source_loc - 1], smooth[source_loc], smooth[source_loc + 1]
        )
        threshold = thresholds[source_loc]
        if candidate_type == NO_CONFIRMED_TURN or not np.isfinite(threshold):
            continue
        if threshold < 0.0:
            raise ValueError("swing_threshold must be non-negative")
        candidate_value = smooth[source_loc]

        if anchor_type == NO_CONFIRMED_TURN:
            anchor_type = candidate_type
            anchor_value = candidate_value
            anchor_loc = source_loc
            continue

        if candidate_type == anchor_type:
            more_extreme = (
                candidate_type == CONFIRMED_PEAK and candidate_value > anchor_value
            ) or (
                candidate_type == CONFIRMED_TROUGH and candidate_value < anchor_value
            )
            if more_extreme:
                anchor_value = candidate_value
                anchor_loc = source_loc
            continue

        amplitude = abs(candidate_value - anchor_value)
        if amplitude <= threshold:
            continue

        states[decision_loc] = candidate_type
        source_locs[decision_loc] = source_loc
        swing_amplitudes[decision_loc] = amplitude
        applied_thresholds[decision_loc] = threshold
        anchor_type = candidate_type
        anchor_value = candidate_value
        anchor_loc = source_loc

    source_dates = pd.Series(pd.NaT, index=z20_smooth.index, dtype="datetime64[ns]")
    emitted = source_locs >= 0
    if emitted.any():
        source_dates.iloc[np.flatnonzero(emitted)] = z20_smooth.index[
            source_locs[emitted]
        ].to_numpy()

    return pd.DataFrame(
        {
            "confirmed_turn_state": states,
            "confirmed_turn_label": pd.Series(states, index=z20_smooth.index).map(
                TURN_STATE_LABELS
            ),
            "confirmed_turn_source_date": source_dates,
            "confirmed_turn_swing_amplitude": swing_amplitudes,
            "confirmed_turn_applied_threshold": applied_thresholds,
        },
        index=z20_smooth.index,
    )


def build_confirmed_turn_features(
    z20: pd.Series,
    *,
    smooth_window: int,
    vol_lookback_days: int,
    k_multiplier: float,
    confirmation_lag_days: int,
    local_vol_ddof: int = 0,
) -> pd.DataFrame:
    """Build all causal confirmed-turn columns for one threshold setting."""
    if not np.isfinite(k_multiplier) or k_multiplier <= 0.0:
        raise ValueError("k_multiplier must be finite and positive")
    if not z20.index.is_monotonic_increasing or z20.index.has_duplicates:
        raise ValueError("z20 index must be unique and increasing")
    smooth = causal_ema(z20, smooth_window)
    local_vol = rolling_local_volatility(z20, vol_lookback_days, local_vol_ddof)
    threshold = k_multiplier * local_vol
    confirmed = confirm_turns_from_smoothed(
        smooth, threshold, confirmation_lag_days
    )
    return pd.concat(
        [
            smooth.rename("z20_smooth"),
            local_vol.rename("confirmed_turn_local_vol"),
            threshold.rename("confirmed_turn_swing_threshold"),
            confirmed,
        ],
        axis=1,
    )
