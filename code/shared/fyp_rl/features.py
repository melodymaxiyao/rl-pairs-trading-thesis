"""Causal feature engineering shared by all models. No feature uses information
from after the close of the day it is indexed on."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DQN_WINDOW, FEATURE_COLUMNS


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the shared feature frame from adjusted OHLCV data.

    Every column at row t is computed using data from rows <= t only
    (rolling windows include the current row, never future rows).
    """
    df = ohlcv.copy().sort_index()
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    sma200 = close.rolling(200).mean()
    log_return = np.log(close / close.shift(1))
    volume_mean20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std(ddof=0)

    feat = pd.DataFrame(index=df.index)
    feat["Open"] = open_
    feat["High"] = high
    feat["Low"] = low
    feat["Close"] = close
    feat["Volume"] = volume
    feat["sma20"] = sma20
    feat["std20"] = std20
    feat["sma200"] = sma200
    feat["z20"] = (close - sma20) / std20.replace(0.0, np.nan)
    feat["trend"] = (close >= sma200).astype(float)
    feat["log_return"] = log_return
    feat["overnight_gap"] = np.log(open_ / close.shift(1))
    feat["intraday_body"] = np.log(close / open_)
    feat["daily_range"] = np.log(high / low)
    feat["sma200_distance"] = close / sma200 - 1.0
    feat["volatility20"] = log_return.rolling(20).std(ddof=0)
    feat["volume_z20"] = (volume - volume_mean20) / volume_std20.replace(0.0, np.nan)

    return feat


def z_bucket(z20: float) -> int:
    """0 = oversold (z<-1), 1 = neutral, 2 = overbought (z>1)."""
    if z20 < -1.0:
        return 0
    if z20 > 1.0:
        return 2
    return 1


def bandit_state(z20: float) -> int:
    return z_bucket(z20)


def tabular_state(z20: float, trend: float, position: int) -> int:
    """12-state encoding: z_bucket(3) x trend(2) x position(2)."""
    zb = z_bucket(z20)
    tr = int(round(trend))
    pos = int(position)
    return zb * 4 + tr * 2 + pos


def tabular_state_labels() -> dict[int, str]:
    labels = {}
    z_names = {0: "oversold", 1: "neutral", 2: "overbought"}
    for zb in range(3):
        for tr in range(2):
            for pos in range(2):
                s = zb * 4 + tr * 2 + pos
                labels[s] = (
                    f"{z_names[zb]}/trend={'up' if tr else 'down'}/"
                    f"pos={'long' if pos else 'cash'}"
                )
    return labels


def dqn_observation(feat: pd.DataFrame, t_loc: int, position: int, window: int = DQN_WINDOW) -> np.ndarray:
    """Flatten the trailing `window` rows of FEATURE_COLUMNS ending at t_loc (inclusive),
    then append the current position. Raises if insufficient history is available."""
    if t_loc - window + 1 < 0:
        raise ValueError("Not enough history to build the observation window")
    block = feat.iloc[t_loc - window + 1 : t_loc + 1][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if np.isnan(block).any():
        raise ValueError("NaN encountered inside the observation window (insufficient warm-up)")
    return np.concatenate([block.flatten(), [float(position)]])
