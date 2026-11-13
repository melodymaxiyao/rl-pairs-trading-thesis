"""Tabular agents for the causal short-term z-score slope experiment.

The slope at row t is z20[t] - z20[t-lookback].  Both the lookback and the
dead-zone threshold are supplied by the experiment config; neither is an
implicit agent constant.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from fyp_rl.config import TabularQConfig
from fyp_rl.execution import single_step_execution
from fyp_rl.features import z_bucket


N_ACTIONS = 2
SIX_STATE_COUNT = 6
TWELVE_STATE_COUNT = 12


def add_z_slope_feature(feat: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Return a copy with a causal z_slope column computed from z20 rows."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if "z20" not in feat:
        raise ValueError("z20 is required to compute z_slope")
    enriched = feat.copy()
    enriched["z_slope"] = enriched["z20"] - enriched["z20"].shift(lookback_days)
    return enriched


def z_slope_bucket(z_slope: float, epsilon_deadzone: float) -> int:
    """Encode falling/flat as 0 and rising as 1 using the configured dead zone."""
    if epsilon_deadzone < 0:
        raise ValueError("epsilon_deadzone must be non-negative")
    if not np.isfinite(z_slope):
        raise ValueError("z_slope must be finite")
    return int(z_slope > epsilon_deadzone)


def six_state_zslope_from_buckets(z_bucket_value: int, slope_bucket: int) -> int:
    """Encode z_bucket(3) x z_slope_bucket(2)."""
    if z_bucket_value not in (0, 1, 2):
        raise ValueError("z_bucket_value must be 0, 1, or 2")
    if slope_bucket not in (0, 1):
        raise ValueError("slope_bucket must be 0 or 1")
    return z_bucket_value * 2 + slope_bucket


def twelve_state_zslope_from_buckets(
    z_bucket_value: int, slope_bucket: int, position: int
) -> int:
    """Encode z_bucket(3) x z_slope_bucket(2) x position(2)."""
    if position not in (0, 1):
        raise ValueError("position must be 0 or 1")
    return six_state_zslope_from_buckets(z_bucket_value, slope_bucket) * 2 + position


def six_state_zslope(z20: float, z_slope: float, epsilon_deadzone: float) -> int:
    return six_state_zslope_from_buckets(
        z_bucket(z20), z_slope_bucket(z_slope, epsilon_deadzone)
    )


def twelve_state_zslope(
    z20: float, z_slope: float, position: int, epsilon_deadzone: float
) -> int:
    return twelve_state_zslope_from_buckets(
        z_bucket(z20), z_slope_bucket(z_slope, epsilon_deadzone), position
    )


def _state_labels(include_position: bool) -> dict[int, str]:
    z_names = {0: "oversold", 1: "neutral", 2: "overbought"}
    slope_names = {0: "falling_or_flat", 1: "rising"}
    labels: dict[int, str] = {}
    for zb in range(3):
        for slope in range(2):
            prefix = f"{z_names[zb]}/z_slope={slope_names[slope]}"
            if include_position:
                for position in range(2):
                    state = twelve_state_zslope_from_buckets(zb, slope, position)
                    labels[state] = f"{prefix}/pos={'long' if position else 'cash'}"
            else:
                labels[six_state_zslope_from_buckets(zb, slope)] = prefix
    return labels


class _ZSlopeTabularAgent:
    def __init__(
        self,
        config: TabularQConfig,
        epsilon_deadzone: float,
        include_position: bool,
    ):
        if epsilon_deadzone < 0:
            raise ValueError("epsilon_deadzone must be non-negative")
        self.config = config
        self.epsilon_deadzone = epsilon_deadzone
        self.include_position = include_position
        self.n_states = TWELVE_STATE_COUNT if include_position else SIX_STATE_COUNT
        self.q = np.zeros((self.n_states, N_ACTIONS), dtype=np.float64)
        self.counts = np.zeros((self.n_states, N_ACTIONS), dtype=np.int64)

    def _state(self, z20: float, slope: float, position: int) -> int:
        if self.include_position:
            return twelve_state_zslope(z20, slope, position, self.epsilon_deadzone)
        return six_state_zslope(z20, slope, self.epsilon_deadzone)

    def train(self, feat_train: pd.DataFrame, cost: float, seed: int) -> None:
        rng = random.Random(seed)
        cfg = self.config
        decision_locs = list(
            range(0, len(feat_train) - cfg.decision_interval, cfg.decision_interval)
        )
        z20 = feat_train["z20"].to_numpy(dtype=np.float64)
        slopes = feat_train["z_slope"].to_numpy(dtype=np.float64)
        valid_locs = [
            loc for loc in decision_locs if np.isfinite(z20[loc]) and np.isfinite(slopes[loc])
        ]
        if not valid_locs:
            raise ValueError("No valid decision points for z-slope Q training")

        for episode in range(cfg.episodes):
            fraction = episode / max(1, cfg.episodes - 1)
            epsilon = cfg.epsilon_start * (cfg.epsilon_end / cfg.epsilon_start) ** fraction
            position = 0
            for index, loc in enumerate(valid_locs):
                state = self._state(float(z20[loc]), float(slopes[loc]), position)
                if rng.random() < epsilon:
                    action = rng.randint(0, 1)
                else:
                    action = int(np.argmax(self.q[state]))

                reward, _, _ = single_step_execution(
                    feat_train,
                    loc,
                    loc + cfg.decision_interval,
                    float(position),
                    float(action),
                    cost,
                )
                if index + 1 < len(valid_locs):
                    next_loc = valid_locs[index + 1]
                    next_state = self._state(
                        float(z20[next_loc]), float(slopes[next_loc]), action
                    )
                    best_next_q = float(np.max(self.q[next_state]))
                else:
                    best_next_q = 0.0

                self.counts[state, action] += 1
                alpha = 1.0 / self.counts[state, action]
                td_target = reward + cfg.gamma * best_next_q
                self.q[state, action] += alpha * (td_target - self.q[state, action])
                position = action

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def rule_fn(self):
        def rule(row: pd.Series, previous_position: float) -> float:
            if pd.isna(row["z20"]) or pd.isna(row["z_slope"]):
                return previous_position
            state = self._state(
                float(row["z20"]), float(row["z_slope"]), int(round(previous_position))
            )
            return float(self.greedy_action(state))

        return rule

    def summary(self) -> dict[str, Any]:
        labels = _state_labels(self.include_position)
        return {
            "q_table": self.q.tolist(),
            "visit_counts": self.counts.tolist(),
            "state_labels": labels,
            "greedy_policy": {
                state: self.greedy_action(state) for state in range(self.n_states)
            },
        }


class SixStateZSlopeAgent(_ZSlopeTabularAgent):
    """Gamma-zero z_bucket x z_slope signal diagnostic (no position state)."""

    def __init__(self, config: TabularQConfig, epsilon_deadzone: float):
        if config.gamma != 0.0:
            raise ValueError("SixStateZSlopeAgent requires gamma=0")
        super().__init__(config, epsilon_deadzone, include_position=False)


class TwelveStateZSlopeAgent(_ZSlopeTabularAgent):
    """Sequential z_bucket x z_slope x current-position Q agent."""

    def __init__(self, config: TabularQConfig, epsilon_deadzone: float):
        super().__init__(config, epsilon_deadzone, include_position=True)
