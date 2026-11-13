"""Model 2: regime-aware 12-state tabular Q-learning.

State: (z_bucket in {0,1,2}) x (trend in {0,1}) x (position in {0,1}) -> 12 states.
Action: 0 = TargetCash, 1 = TargetLong (a target position, not an order).
Reward: net log-wealth growth from the unified execution engine, over one
decision interval (default 5 trading days), including transaction cost.
Update: Q(s,a) += alpha_t(s,a) * [r + gamma * max_a' Q(s',a') - Q(s,a)],
alpha_t(s,a) = 1/N(s,a), epsilon decays geometrically from start to end.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ..config import TabularQConfig
from ..execution import single_step_execution
from ..features import tabular_state, tabular_state_labels, z_bucket


N_STATES = 12
N_ACTIONS = 2


class TabularQAgent:
    def __init__(self, config: TabularQConfig | None = None):
        self.config = config or TabularQConfig()
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)

    def _decision_locs(self, n: int) -> list[int]:
        interval = self.config.decision_interval
        locs = list(range(0, n - interval, interval))
        return locs

    def train(self, feat_train: pd.DataFrame, cost: float, seed: int) -> None:
        rng = random.Random(seed)
        cfg = self.config
        decision_locs = self._decision_locs(len(feat_train))
        # rows with valid z20/trend only
        z20 = feat_train["z20"].to_numpy()
        trend = feat_train["trend"].to_numpy()
        valid = [i for i in decision_locs if not (np.isnan(z20[i]) or np.isnan(trend[i]))]
        if not valid:
            raise ValueError("No valid decision points for tabular Q training (check warm-up)")

        for episode in range(cfg.episodes):
            frac = episode / max(1, cfg.episodes - 1)
            epsilon = cfg.epsilon_start * (cfg.epsilon_end / cfg.epsilon_start) ** frac
            position = 0
            for idx, loc in enumerate(valid):
                end_loc = loc + cfg.decision_interval
                if end_loc >= len(feat_train):
                    break
                s = tabular_state(float(z20[loc]), float(trend[loc]), position)
                if rng.random() < epsilon:
                    a = rng.randint(0, 1)
                else:
                    a = int(np.argmax(self.q[s]))

                reward, _, _ = single_step_execution(feat_train, loc, end_loc, float(position), float(a), cost)

                next_position = a
                if idx + 1 < len(valid):
                    next_loc = valid[idx + 1]
                    s_next = tabular_state(float(z20[next_loc]), float(trend[next_loc]), next_position)
                    best_next_q = float(np.max(self.q[s_next]))
                else:
                    best_next_q = 0.0

                self.counts[s, a] += 1
                alpha = 1.0 / self.counts[s, a]
                td_target = reward + cfg.gamma * best_next_q
                self.q[s, a] += alpha * (td_target - self.q[s, a])

                position = next_position

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def summary(self) -> dict:
        labels = tabular_state_labels()
        return {
            "q_table": self.q.tolist(),
            "visit_counts": self.counts.tolist(),
            "state_labels": labels,
            "greedy_policy": {s: self.greedy_action(s) for s in range(N_STATES)},
        }

    def rule_fn(self):
        def rule(row: pd.Series, prev: float) -> float:
            z, tr = row["z20"], row["trend"]
            if pd.isna(z) or pd.isna(tr):
                return prev
            position = int(round(prev))
            s = tabular_state(float(z), float(tr), position)
            return float(self.greedy_action(s))

        return rule
