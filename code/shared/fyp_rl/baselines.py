"""Deterministic non-RL baselines sharing the unified decision/execution timing.

All baselines decide only every `decision_interval` trading days (locs
0, interval, 2*interval, ... relative to the start of the supplied feature
slice) and hold their target position unchanged in between. Loc 0 is expected
to be the last available pre-test trading day so its decision executes at the
open of the first in-sample/test trading day, per the experiment 12 spec.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import DECISION_INTERVAL, Z_ENTRY


def build_decision_series(
    feat: pd.DataFrame,
    rule_fn: Callable[[pd.Series, float], float],
    interval: int = DECISION_INTERVAL,
    initial_target: float = 0.0,
) -> pd.Series:
    n = len(feat)
    targets = np.empty(n, dtype=np.float64)
    current = initial_target
    for i in range(n):
        row = feat.iloc[i]
        if i % interval == 0:
            current = rule_fn(row, current)
        targets[i] = current
    return pd.Series(targets, index=feat.index, name="target_position")


def cash_baseline(feat: pd.DataFrame, interval: int = DECISION_INTERVAL) -> pd.Series:
    return build_decision_series(feat, lambda row, prev: 0.0, interval=interval, initial_target=0.0)


def buy_and_hold_baseline(feat: pd.DataFrame, interval: int = DECISION_INTERVAL) -> pd.Series:
    return build_decision_series(feat, lambda row, prev: 1.0, interval=interval, initial_target=0.0)


def zscore_rule_baseline(
    feat: pd.DataFrame, interval: int = DECISION_INTERVAL, z_entry: float = Z_ENTRY
) -> pd.Series:
    def rule(row: pd.Series, prev: float) -> float:
        z = row["z20"]
        if pd.isna(z):
            return prev
        if z < -z_entry:
            return 1.0
        if z > z_entry:
            return 0.0
        return prev

    return build_decision_series(feat, rule, interval=interval, initial_target=0.0)


def sma200_rule_baseline(feat: pd.DataFrame, interval: int = DECISION_INTERVAL) -> pd.Series:
    def rule(row: pd.Series, prev: float) -> float:
        if pd.isna(row["sma200"]):
            return prev
        return 1.0 if row["Close"] >= row["sma200"] else 0.0

    return build_decision_series(feat, rule, interval=interval, initial_target=0.0)
