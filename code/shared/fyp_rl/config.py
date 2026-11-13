"""Shared configuration dataclasses for fyp_rl experiments."""
from __future__ import annotations

from dataclasses import dataclass, field


DECISION_INTERVAL = 5  # trading days between decisions, shared by all models/baselines
Z_ENTRY = 1.0  # |z20| threshold used by the hysteresis z-score rule
FEATURE_COLUMNS = [
    "log_return",
    "overnight_gap",
    "intraday_body",
    "daily_range",
    "z20",
    "sma200_distance",
    "volatility20",
    "volume_z20",
]
DQN_WINDOW = 20
DQN_INPUT_DIM = DQN_WINDOW * len(FEATURE_COLUMNS) + 1  # + current position -> 161


@dataclass
class DataSplit:
    """A named train/test date range."""

    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class ExecutionConfig:
    transaction_cost: float = 0.0
    decision_interval: int = DECISION_INTERVAL


@dataclass
class BanditConfig:
    horizon: int = 30
    gamma: float = 0.0


@dataclass
class TabularQConfig:
    gamma: float = 0.90
    episodes: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decision_interval: int = DECISION_INTERVAL


@dataclass
class DQNConfig:
    gamma: float = 0.90
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 50_000
    replay_warmup: int = 500
    target_update_interval: int = 250  # gradient steps
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    episodes: int = 100
    decision_interval: int = DECISION_INTERVAL
    hidden_dim: int = 64
    double_dqn: bool = False


@dataclass
class BootstrapConfig:
    block_length: int = 20
    n_repetitions: int = 5000
    seed: int = 12345


COST_GRID = (0.0, 0.0005, 0.001, 0.002)
SEEDS = tuple(range(10))
