"""Model 1: 3-state contextual bandit (pure signal diagnostic).

State: z_bucket(z20) in {0=oversold, 1=neutral, 2=overbought}.
Action: 0 = TargetCash, 1 = TargetLong.
Reward: R_t(a) = a * log(Close_{t+H} / Close_t), H=30 trading days.
gamma = 0 (bandit, no bootstrapping). Q is updated with a visit-count sample
average, matching Q_{n+1} = Q_n + (1/N)(R - Q_n).
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ..config import BanditConfig
from ..features import bandit_state


class ContextualBanditAgent:
    def __init__(self, config: BanditConfig | None = None):
        self.config = config or BanditConfig()
        self.q = np.zeros((3, 2), dtype=np.float64)
        self.counts = np.zeros((3, 2), dtype=np.int64)

    def train(self, feat_train: pd.DataFrame, seed: int, n_passes: int = 50,
              epsilon_start: float = 1.0, epsilon_end: float = 0.05) -> None:
        rng = random.Random(seed)
        horizon = self.config.horizon
        close = feat_train["Close"]
        z20 = feat_train["z20"]

        valid_locs = [
            i for i in range(len(feat_train) - horizon)
            if not pd.isna(z20.iloc[i])
        ]
        if not valid_locs:
            raise ValueError("No valid training rows for the contextual bandit (check horizon/warmup)")

        for episode in range(n_passes):
            frac = episode / max(1, n_passes - 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            order = valid_locs[:]
            rng.shuffle(order)
            for i in order:
                s = bandit_state(float(z20.iloc[i]))
                if rng.random() < epsilon:
                    a = rng.randint(0, 1)
                else:
                    a = int(np.argmax(self.q[s]))
                reward = a * float(np.log(close.iloc[i + horizon] / close.iloc[i]))
                self.counts[s, a] += 1
                alpha = 1.0 / self.counts[s, a]
                self.q[s, a] += alpha * (reward - self.q[s, a])

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def greedy_policy(self) -> dict:
        return {s: self.greedy_action(s) for s in range(3)}

    def summary(self) -> dict:
        return {
            "q_table": self.q.tolist(),
            "visit_counts": self.counts.tolist(),
            "greedy_policy": self.greedy_policy(),
            "q_long_by_state": {int(s): float(self.q[s, 1]) for s in range(3)},
            "s0_minus_s2_long": float(self.q[0, 1] - self.q[2, 1]),
        }

    def rule_fn(self):
        def rule(row: pd.Series, prev: float) -> float:
            z = row["z20"]
            if pd.isna(z):
                return prev
            s = bandit_state(float(z))
            return float(self.greedy_action(s))

        return rule
