"""Causal pair features, explicit two-leg accounting, and 15-state Q pilot.

Adjusted OHLC are synthetic total-return units, not actual historical shares.
The 15-state observation is deliberately coarse and not claimed fully Markov.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ACTIONS = (0, -1, 1)  # Cash wins exact greedy ties; never default to a short.


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject bad snapshots rather than silently deleting/forward-filling bars."""
    df = frame.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        raise ValueError("Full adjusted OHLCV required")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    values = df[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Invalid OHLCV values")
    if (df.High + 1e-8 < df[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High below another OHLC price")
    if (df.Low - 1e-8 > df[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low above another OHLC price")
    return df[required]


def align_pair(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a, b = validate_prices(a), validate_prices(b)
    common = a.index.intersection(b.index)
    if len(common) < 2:
        raise ValueError("Insufficient common trading dates")
    df = pd.concat([a.loc[common].add_suffix("_a"), b.loc[common].add_suffix("_b")], axis=1)
    return df, {"common_rows": len(common), "a_only_dates": [str(x.date()) for x in a.index.difference(b.index)],
                "b_only_dates": [str(x.date()) for x in b.index.difference(a.index)]}


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, float(p[index]) * (len(p) - rank))
        adjusted[index] = min(1.0, previous)
    return adjusted.tolist()


def screen_pairs(frames: dict[str, pd.DataFrame], fold: dict, min_rows=1800, alpha=0.05) -> list[dict]:
    """All fit/test operations use the training slice, never future dates.

    I(1) diagnostic gate: ADF level p >= alpha and difference p < alpha
    for both legs. Non-rejection alone is not proof of a unit root.
    Holm family includes all candidates, including any failed I(1) gates.
    """
    from statsmodels.tsa.stattools import adfuller, coint

    records = []
    for name, frame in frames.items():
        train = frame.loc[fold["train_start"]:fold["train_end"]]
        if len(train) < min_rows:
            raise ValueError(f"{name}: insufficient training coverage")
        logs = np.log(train[["Close_a", "Close_b"]].to_numpy(dtype=float))
        alpha_fit, beta = np.linalg.lstsq(np.column_stack([np.ones(len(train)), logs[:, 1]]), logs[:, 0], rcond=None)[0]
        level_p = [float(adfuller(logs[:, i], regression="c", autolag="AIC")[1]) for i in (0, 1)]
        diff_p = [float(adfuller(np.diff(logs[:, i]), regression="c", autolag="AIC")[1]) for i in (0, 1)]
        stat, p, _ = coint(logs[:, 0], logs[:, 1], trend="c", autolag="AIC")
        records.append({"pair": name, "train_first": str(train.index[0].date()),
                        "train_last": str(train.index[-1].date()), "train_rows": len(train),
                        "alpha_fit": float(alpha_fit), "beta": float(beta),
                        "adf_level_p": level_p, "adf_difference_p": diff_p,
                        "i1_diagnostic_pass": all(x >= alpha for x in level_p) and all(x < alpha for x in diff_p),
                        "coint_stat": float(stat), "coint_p": float(p)})
    adjusted = holm_adjust([x["coint_p"] for x in records])
    for row, p in zip(records, adjusted):
        row["coint_holm_p"] = p
        row["eligible"] = row["i1_diagnostic_pass"] and row["beta"] > 0 and p < alpha
        row["reasons"] = (["I(1) diagnostic gate failed"] if not row["i1_diagnostic_pass"] else []) + (
            ["non-positive hedge beta"] if row["beta"] <= 0 else []) + (
            ["Holm-adjusted cointegration gate failed"] if p >= alpha else [])
    return records


def pair_features(frame: pd.DataFrame, alpha_fit: float, beta: float, window=60) -> pd.DataFrame:
    out = frame.copy()
    out["spread"] = np.log(out.Close_a) - alpha_fit - beta * np.log(out.Close_b)
    rolling = out.spread.rolling(window, min_periods=window)
    std = rolling.std(ddof=1)
    out["z"] = ((out.spread - rolling.mean()) / std).where(std > 1e-12)
    return out


def state_id(z: float, position: int) -> int:
    if not np.isfinite(z) or position not in ACTIONS:
        raise ValueError("Invalid pair observation")
    bucket = 0 if z < -2 else 1 if z < -1 else 2 if z <= 1 else 3 if z <= 2 else 4
    return bucket * 3 + (position + 1)


def threshold_action(z: float, previous: int) -> int:
    if not np.isfinite(z):
        return 0
    if previous == 1:
        return 0 if z >= 0 else 1
    if previous == -1:
        return 0 if z <= 0 else -1
    return 1 if z < -1 else -1 if z > 1 else 0


def target_weights(position: int, beta: float) -> np.ndarray:
    if position not in ACTIONS or not np.isfinite(beta) or beta <= 0:
        raise ValueError("Positive finite beta and valid action required")
    return np.array([position / (1 + beta), -position * beta / (1 + beta)])


@dataclass
class PairBook:
    cash: float = 100000.0
    shares_a: float = 0.0
    shares_b: float = 0.0

    def equity(self, prices) -> float:
        return float(self.cash + np.dot([self.shares_a, self.shares_b], prices))

    def step(self, previous_close, next_open, next_close, weights, cost: float,
             annual_borrow: float, liquidate=False) -> dict:
        """Close t decision -> open t+1 trades -> close t+1 valuation.

        Targets use pre-fee open equity. Borrow is annual/252 times short
        liability at previous close (one-trading-day proxy, including an
        overnight position exited next open). No separate dividend credits.
        End-of-episode liquidation executes at the final close and pays fees.
        """
        prev, op, cl, w = [np.asarray(x, dtype=float) for x in (previous_close, next_open, next_close, weights)]
        if any(x.shape != (2,) or not np.isfinite(x).all() for x in (prev, op, cl, w)):
            raise ValueError("Two finite values required")
        if any((x <= 0).any() for x in (prev, op, cl)) or not 0 <= cost < 1 or not 0 <= annual_borrow < 1:
            raise ValueError("Invalid price or cost")
        if np.abs(w).sum() > 1 + 1e-12:
            raise ValueError("Gross target exposure must not exceed one")
        old = np.array([self.shares_a, self.shares_b])
        before = self.equity(prev)
        at_open = self.equity(op)
        if min(before, at_open) <= 0:
            raise ValueError("Insolvent book: do not silently clip returns")
        new = w * at_open / op
        delta = new - old
        traded = delta * op
        turnover = float(np.abs(traded).sum())
        trade_fee = cost * turnover
        borrow_fee = annual_borrow / 252 * float(np.maximum(-old * prev, 0).sum())
        self.cash -= float(traded.sum()) + trade_fee + borrow_fee
        self.shares_a, self.shares_b = map(float, new)
        marked = new * cl
        pre_liquidation_equity = self.equity(cl)
        if pre_liquidation_equity <= 0:
            raise ValueError("Insolvent closing book")
        liquidation_fee = cost * float(np.abs(marked).sum()) if liquidate else 0.0
        events = [{"leg": leg, "execution": "open", "delta_shares": float(ds),
                   "price": float(price), "notional": float(ds * price), "fee": float(cost * abs(ds * price))}
                  for leg, ds, price in zip(("a", "b"), delta, op) if abs(ds * price) > 1e-10]
        if liquidate:
            events.extend({"leg": leg, "execution": "terminal_close", "delta_shares": float(-sh),
                           "price": float(price), "notional": float(-sh * price), "fee": float(cost * abs(sh * price))}
                          for leg, sh, price in zip(("a", "b"), new, cl) if abs(sh * price) > 1e-10)
            self.cash += float(marked.sum()) - liquidation_fee
            self.shares_a = self.shares_b = 0.0
        after = self.equity(cl)
        if after <= 0:
            raise ValueError("Insolvent liquidation")
        overnight_pnl = float(np.dot(old, op - prev))
        intraday_pnl = float(np.dot(new, cl - op))
        assert np.isclose(after, before + overnight_pnl + intraday_pnl - trade_fee - borrow_fee - liquidation_fee,
                          rtol=1e-11, atol=1e-8), "Self-financing reconciliation failed"
        return {"equity": after, "daily_return": after / before - 1, "reward": float(np.log(after / before)),
                "cash": self.cash, "shares_a": self.shares_a, "shares_b": self.shares_b,
                "overnight_pnl": overnight_pnl, "intraday_pnl": intraday_pnl,
                "trade_fee": trade_fee, "borrow_fee": borrow_fee, "liquidation_fee": liquidation_fee,
                "turnover_usd": turnover + (float(np.abs(marked).sum()) if liquidate else 0),
                "gross_exposure_before_liquidation": float(np.abs(marked).sum() / pre_liquidation_equity),
                "net_exposure_before_liquidation": float(marked.sum() / pre_liquidation_equity),
                "events": events}


class PairQAgent:
    def __init__(self):
        self.q = np.zeros((15, 3), dtype=float)
        self.counts = np.zeros((15, 3), dtype=np.int64)

    def action(self, z, position):
        return ACTIONS[int(np.argmax(self.q[state_id(z, position)]))]

    def update(self, state, action_index, reward, next_state, done, gamma):
        self.counts[state, action_index] += 1
        alpha = 1.0 / self.counts[state, action_index]
        target = reward + (0.0 if done else gamma * float(self.q[next_state].max()))
        self.q[state, action_index] += alpha * (target - self.q[state, action_index])

    def train(self, train: pd.DataFrame, beta: float, cfg: dict, cost: float, borrow: float, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        # No skip across unknown z observations; first warm-up is sliced once.
        first = train.z.first_valid_index()
        if first is None:
            raise ValueError("No finite training features")
        df = train.loc[first:]
        if len(df) < 2 or not np.isfinite(df.z).all():
            raise ValueError("Incomplete training features")
        op = df[["Open_a", "Open_b"]].to_numpy()
        cl = df[["Close_a", "Close_b"]].to_numpy()
        z = df.z.to_numpy()
        transitions = 0
        for episode in range(cfg["episodes"]):
            frac = episode / max(1, cfg["episodes"] - 1)
            epsilon = cfg["epsilon_start"] * (cfg["epsilon_end"] / cfg["epsilon_start"]) ** frac
            book, position = PairBook(), 0
            for t in range(len(df) - 1):
                state = state_id(z[t], position)
                ai = int(rng.integers(3)) if rng.random() < epsilon else int(np.argmax(self.q[state]))
                action = ACTIONS[ai]
                done = t == len(df) - 2
                outcome = book.step(cl[t], op[t + 1], cl[t + 1], target_weights(action, beta), cost, borrow, done)
                next_position = 0 if done else action
                self.update(state, ai, outcome["reward"], state_id(z[t + 1], next_position), done, cfg["gamma"])
                position = next_position
                transitions += 1
        return {"episodes": cfg["episodes"], "seed": seed, "transitions": transitions,
                "train_first_state_date": str(df.index[0].date()), "train_last_reward_date": str(df.index[-1].date()),
                "q_table": self.q.tolist(), "visit_counts": self.counts.tolist(),
                "visited_state_actions": int((self.counts > 0).sum()), "action_order": list(ACTIONS)}


def run_policy(features: pd.DataFrame, start: str, end: str, beta: float, policy,
               cost: float, borrow: float, initial=100000.0, equal_long=False):
    execution_dates = features.loc[start:end].index
    if len(execution_dates) < 2:
        raise ValueError("Insufficient evaluation rows")
    first_loc = features.index.get_loc(execution_dates[0])
    if first_loc == 0:
        raise ValueError("Prior close required for first test-open execution")
    book, position = PairBook(initial), 0
    daily, actions, trades = [], [], []
    for date in execution_dates:
        loc = features.index.get_loc(date)
        prev, current = features.iloc[loc - 1], features.iloc[loc]
        z = float(prev.z)
        action = policy(z, position)
        weights = np.array([0.5, 0.5]) if equal_long else target_weights(action, beta)
        final = date == execution_dates[-1]
        result = book.step(prev[["Close_a", "Close_b"]], current[["Open_a", "Open_b"]],
                           current[["Close_a", "Close_b"]], weights, cost, borrow, final)
        for event in result.pop("events"):
            trades.append({"date": str(date.date()), **event})
        actions.append({"signal_date": str(features.index[loc - 1].date()), "execution_date": str(date.date()),
                        "z": z, "previous_position": position, "target_position": action,
                        "terminal_liquidation": final})
        daily.append({"date": date, **result})
        position = 0 if final else action
    daily = pd.DataFrame(daily).set_index("date")
    return daily, pd.DataFrame(actions), pd.DataFrame(trades, columns=["date", "leg", "execution", "delta_shares", "price", "notional", "fee"])
