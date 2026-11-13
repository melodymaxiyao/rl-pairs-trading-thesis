"""Performance metrics computed from daily return / equity series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=0) == 0 or daily_returns.empty:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / daily_returns.std(ddof=0))


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_days = len(equity_curve) - 1
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return float(cagr(equity_curve) / abs(mdd))


def cumulative_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def beta_vs_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)
    var_bm = cov[1, 1]
    if var_bm == 0:
        return 0.0
    return float(cov[0, 1] / var_bm)


def full_metrics(equity_curve: pd.Series, daily_returns: pd.Series, benchmark_returns: pd.Series | None = None) -> dict:
    out = {
        "cumulative_return": cumulative_return(equity_curve),
        "cagr": cagr(equity_curve),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve),
    }
    if benchmark_returns is not None:
        out["beta_vs_buy_and_hold"] = beta_vs_benchmark(daily_returns, benchmark_returns)
    return out
