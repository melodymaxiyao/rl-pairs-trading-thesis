"""Unified next-open execution engine shared by all baselines and RL models.

Timing convention (see experiment 12 README for the full derivation):
  - The state/decision at "day t" is computed using information available at
    the close of day t.
  - The resulting target position w_new is executed at the OPEN of day t+1.
  - w_old is the position that was already held going into day t+1 (i.e. the
    target decided at the close of day t-1, or the initial position).

Wealth recursion (V indexed by close-of-day):
  V_open_before = V_t * (1 + w_old * (Open_{t+1}/Close_t - 1))
  V_open_after  = V_open_before * (1 - c * |w_new - w_old|)
  V_{t+1}       = V_open_after * (1 + w_new * (Close_{t+1}/Open_{t+1} - 1))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    daily: pd.DataFrame  # index=date, columns: w_old, w_new, cost_paid, V, daily_return
    equity_curve: pd.Series
    daily_returns: pd.Series
    turnover: float
    total_cost_paid: float
    num_entries: int
    num_exits: int
    exposure: float
    avg_holding_period: float
    trades: pd.DataFrame  # holding_days per completed round trip


def simulate_positions(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,
    cost: float,
    initial_position: float = 0.0,
) -> ExecutionResult:
    """Simulate the unified execution model.

    Parameters
    ----------
    ohlcv: DataFrame with Open/Close columns, sorted ascending, one row per trading day.
    target_position: Series aligned to ohlcv.index. target_position[t] is the position
        decided using information available at close of day t (w_new to be executed
        at the open of day t+1). Non-decision days must already be forward-filled by
        the caller so that "holding position unchanged" falls out naturally.
    cost: one-way proportional transaction cost, charged on |w_new - w_old| at the
        post-gap-move wealth level.
    initial_position: position held before the first row of ohlcv (w_old for day 1).
    """
    if not ohlcv.index.equals(target_position.index):
        raise ValueError("target_position must be aligned to ohlcv.index")
    if len(ohlcv) < 2:
        raise ValueError("Need at least two trading days to simulate execution")

    dates = ohlcv.index
    opens = ohlcv["Open"].to_numpy(dtype=np.float64)
    closes = ohlcv["Close"].to_numpy(dtype=np.float64)
    targets = target_position.to_numpy(dtype=np.float64)

    n = len(dates)
    V = 1.0
    w_old = float(initial_position)

    records = []
    entries = 0
    exits = 0
    total_cost = 0.0
    total_turnover = 0.0
    holding_days = []
    current_entry_idx = None

    for i in range(1, n):
        w_new = targets[i - 1]
        open_move = opens[i] / closes[i - 1] - 1.0
        V_open_before = V * (1.0 + w_old * open_move)
        turnover_step = abs(w_new - w_old)
        cost_paid = V_open_before * cost * turnover_step
        V_open_after = V_open_before - cost_paid
        intraday_move = closes[i] / opens[i] - 1.0
        V_next = V_open_after * (1.0 + w_new * intraday_move)

        if w_old == 0.0 and w_new == 1.0:
            entries += 1
            current_entry_idx = i
        if w_old == 1.0 and w_new == 0.0:
            exits += 1
            if current_entry_idx is not None:
                holding_days.append(i - current_entry_idx)
                current_entry_idx = None

        daily_return = V_next / V - 1.0
        records.append(
            {
                "date": dates[i],
                "w_old": w_old,
                "w_new": w_new,
                "cost_paid": cost_paid,
                "V": V_next,
                "daily_return": daily_return,
            }
        )
        total_cost += cost_paid
        total_turnover += turnover_step
        V = V_next
        w_old = w_new

    daily_df = pd.DataFrame.from_records(records).set_index("date")
    equity_curve = pd.concat([pd.Series([1.0], index=[dates[0]]), daily_df["V"]])
    daily_returns = daily_df["daily_return"]

    trades = pd.DataFrame({"holding_days": holding_days})
    avg_holding_period = float(np.mean(holding_days)) if holding_days else 0.0
    exposure = float(daily_df["w_old"].mean()) if not daily_df.empty else 0.0

    return ExecutionResult(
        daily=daily_df,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        turnover=total_turnover,
        total_cost_paid=total_cost,
        num_entries=entries,
        num_exits=exits,
        exposure=exposure,
        avg_holding_period=avg_holding_period,
        trades=trades,
    )


def wealth_between(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,
    start_loc: int,
    end_loc: int,
    w_old_at_start: float,
    cost: float,
) -> tuple[float, float, float]:
    """Compute (log-wealth-growth reward, final wealth ratio, cost paid) for the
    segment (start_loc, end_loc] using the same recursion as simulate_positions,
    with V starting at 1.0 at start_loc. Used by RL agents to score one decision
    step at a time without re-running the full-history simulation."""
    opens = ohlcv["Open"].to_numpy(dtype=np.float64)
    closes = ohlcv["Close"].to_numpy(dtype=np.float64)
    targets = target_position.to_numpy(dtype=np.float64)

    V = 1.0
    w_old = w_old_at_start
    total_cost = 0.0
    for i in range(start_loc + 1, end_loc + 1):
        w_new = targets[i - 1]
        open_move = opens[i] / closes[i - 1] - 1.0
        V_open_before = V * (1.0 + w_old * open_move)
        cost_paid = V_open_before * cost * abs(w_new - w_old)
        V_open_after = V_open_before - cost_paid
        intraday_move = closes[i] / opens[i] - 1.0
        V = V_open_after * (1.0 + w_new * intraday_move)
        total_cost += cost_paid
        w_old = w_new

    reward = float(np.log(V))
    return reward, float(V), total_cost


def single_step_execution(
    ohlcv: pd.DataFrame,
    start_loc: int,
    end_loc: int,
    w_old: float,
    w_new: float,
    cost: float,
) -> tuple[float, float, float]:
    """Same recursion as wealth_between, but the target position is a single
    constant decided once at start_loc and held through end_loc (used by RL
    agents whose decision interval spans multiple trading days)."""
    opens = ohlcv["Open"].to_numpy(dtype=np.float64)
    closes = ohlcv["Close"].to_numpy(dtype=np.float64)

    V = 1.0
    cur = w_old
    total_cost = 0.0
    for i in range(start_loc + 1, end_loc + 1):
        open_move = opens[i] / closes[i - 1] - 1.0
        V_open_before = V * (1.0 + cur * open_move)
        cost_paid = V_open_before * cost * abs(w_new - cur)
        V_open_after = V_open_before - cost_paid
        intraday_move = closes[i] / opens[i] - 1.0
        V = V_open_after * (1.0 + w_new * intraday_move)
        total_cost += cost_paid
        cur = w_new

    reward = float(np.log(V))
    return reward, float(V), total_cost
