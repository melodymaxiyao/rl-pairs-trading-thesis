"""Six-state tabular Q-learning for the state/horizon comparison.

State: z_bucket(3) x position(2). Action: TargetCash(0) or TargetLong(1).
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from fyp_rl.config import TabularQConfig
from fyp_rl.execution import single_step_execution
from fyp_rl.features import z_bucket


N_STATES = 6
N_ACTIONS = 2


def six_state_from_bucket(z_bucket_value: int, position: int) -> int:
    """Encode z_bucket(3) x position(2) as one of six state IDs."""
    if z_bucket_value not in (0, 1, 2):
        raise ValueError("z_bucket_value must be 0, 1, or 2")
    if position not in (0, 1):
        raise ValueError("position must be 0 or 1")
    return z_bucket_value * 2 + position


def six_state(z20: float, position: int) -> int:
    return six_state_from_bucket(z_bucket(z20), position)


def six_state_labels() -> dict[int, str]:
    z_names = {0: "oversold", 1: "neutral", 2: "overbought"}
    return {
        six_state_from_bucket(zb, position): (
            f"{z_names[zb]}/pos={'long' if position else 'cash'}"
        )
        for zb in range(3)
        for position in range(2)
    }


class SixStateTabularQAgent:
    def __init__(self, config: TabularQConfig):
        self.config = config
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)

    def train(self, feat_train: pd.DataFrame, cost: float, seed: int) -> None:
        rng = random.Random(seed)
        cfg = self.config
        decision_locs = list(range(0, len(feat_train) - cfg.decision_interval, cfg.decision_interval))
        z20 = feat_train["z20"].to_numpy(dtype=np.float64)
        valid_locs = [loc for loc in decision_locs if not np.isnan(z20[loc])]
        if not valid_locs:
            raise ValueError("No valid decision points for six-state Q training")

        for episode in range(cfg.episodes):
            fraction = episode / max(1, cfg.episodes - 1)
            epsilon = cfg.epsilon_start * (cfg.epsilon_end / cfg.epsilon_start) ** fraction
            position = 0
            for index, loc in enumerate(valid_locs):
                end_loc = loc + cfg.decision_interval
                state = six_state(float(z20[loc]), position)
                if rng.random() < epsilon:
                    action = rng.randint(0, 1)
                else:
                    action = int(np.argmax(self.q[state]))

                reward, _, _ = single_step_execution(
                    feat_train,
                    loc,
                    end_loc,
                    float(position),
                    float(action),
                    cost,
                )

                if index + 1 < len(valid_locs):
                    next_loc = valid_locs[index + 1]
                    next_state = six_state(float(z20[next_loc]), action)
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
            if pd.isna(row["z20"]):
                return previous_position
            state = six_state(float(row["z20"]), int(round(previous_position)))
            return float(self.greedy_action(state))

        return rule

    def summary(self) -> dict:
        labels = six_state_labels()
        return {
            "q_table": self.q.tolist(),
            "visit_counts": self.counts.tolist(),
            "state_labels": labels,
            "greedy_policy": {state: self.greedy_action(state) for state in range(N_STATES)},
        }