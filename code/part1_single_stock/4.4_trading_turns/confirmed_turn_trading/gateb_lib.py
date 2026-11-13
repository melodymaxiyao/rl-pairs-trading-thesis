"""gateb_v1 library. Definitions: preregistration.md (SHA-256 in preregistration.sha256).

Reuses the restored FYP code read-only: fyp_rl.execution / statistics / metrics / data and the
new_stage1 detector primitives. Nothing here edits restored code; the only runtime change is a
guard that makes yfinance downloads raise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESTORED = Path("/Users/maxiyao/single_stock_runs/inputs_restored_v1")
if str(RESTORED) not in sys.path:
    sys.path.insert(0, str(RESTORED))

import fyp_rl.data as fdata  # noqa: E402


def _blocked_download(*args, **kwargs):
    raise RuntimeError("gateb_v1: yfinance download blocked; inputs must come from the restored cache")


fdata.yf.download = _blocked_download

from fyp_rl.data import load_ohlcv  # noqa: E402
from fyp_rl.execution import simulate_positions  # noqa: E402
from fyp_rl.metrics import sharpe_ratio, cagr, annualized_volatility, max_drawdown  # noqa: E402
from experiments.turning_point_detection.new_stage1.kernel_smoothing import causal_local_linear_slope  # noqa: E402
from experiments.turning_point_detection.new_stage1.slope_accuracy import detect_hysteresis_events  # noqa: E402
from experiments.turning_point_detection.new_stage1.run_slope_accuracy import load_prices  # noqa: E402

TYPE_SIGN = {"peak": -1, "trough": 1}
SIGN_TYPE = {-1: "peak", 1: "trough"}


# ---------------------------------------------------------------- inputs
def load_config(out_dir: Path) -> dict:
    return json.loads((out_dir / "gateb_config.json").read_text())


def load_detector_config(cfg: dict) -> dict:
    return json.loads((RESTORED / cfg["detector_config_file"]).read_text())


def load_tables(cfg: dict) -> dict:
    sel = pd.read_csv(RESTORED / cfg["selected_params_file"])
    sel = sel[(sel.source == "real") & (sel.repeat == -1)]
    if len(sel) != 48 or sel.duplicated(["h_ref", "fold_id"]).any():
        raise ValueError("unexpected selected params layout")
    cal = pd.read_csv(RESTORED / cfg["fold_calendar_file"])
    return {"selected": sel, "calendar": cal}


def load_ticker_prices(ticker: str, det_cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """log(Close) exactly as new_stage1 loads it, plus Open/Close for execution from the same loader."""
    logp = load_prices(ticker, det_cfg)
    ohlcv = load_ohlcv(ticker, str(logp.index[0].date()), str(logp.index[-1].date()),
                       raw_dir=RESTORED / det_cfg["raw_dir"])
    if not ohlcv.index.equals(logp.index):
        raise ValueError(f"{ticker}: price index mismatch between load_prices and load_ohlcv")
    return logp, ohlcv


def selected_params(sel: pd.DataFrame, fold_id: str, h_ref: int) -> tuple[int, int, float]:
    row = sel[(sel.fold_id == fold_id) & (sel.h_ref == h_ref)]
    if len(row) != 1:
        raise ValueError(f"no unique selected params for {fold_id} h_ref={h_ref}")
    r = row.iloc[0]
    return int(r.h), int(r.d), float(r.k)


# ---------------------------------------------------------------- detector
def fold_events(logp: pd.Series, fold: dict, h: int, d: int, k: float, det_cfg: dict) -> pd.DataFrame:
    """Events of one fold's detector run, computed exactly as evaluate_fold does."""
    index = logp.index
    start = index.searchsorted(pd.Timestamp(fold["train_start"]))
    stop = index.searchsorted(pd.Timestamp(fold["test_end"]), side="right")
    local = logp.iloc[start:stop]
    local_train = local.index <= pd.Timestamp(fold["train_end"])
    b = causal_local_linear_slope(local, bandwidth_h=h, window_multiplier=det_cfg["window_multiplier"])
    sigma = float(b.loc[local_train].std(ddof=det_cfg["std_ddof"]))
    if not np.isfinite(sigma):
        raise ValueError("Insufficient finite TRAIN slopes")
    ev = detect_hysteresis_events(b.to_numpy(), tau=k * sigma, d=d)
    dates = local.index[ev[:, 0]]
    out = pd.DataFrame({"date": dates, "type": [SIGN_TYPE[int(s)] for s in ev[:, 1]]})
    out["segment"] = np.where(out.date <= pd.Timestamp(fold["train_end"]), "train", "test")
    return out


def compare_with_detector_events(events: pd.DataFrame, stored: pd.DataFrame) -> dict:
    a = events[["date", "type"]].assign(date=lambda x: pd.to_datetime(x.date)).reset_index(drop=True)
    b = stored[["date", "type"]].assign(date=lambda x: pd.to_datetime(x.date)).sort_values("date").reset_index(drop=True)
    same = len(a) == len(b) and bool((a.date.values == b.date.values).all()) and bool((a.type.values == b.type.values).all())
    return {"n_recomputed": len(a), "n_stored": len(b), "identical": same}


def init_state(events: pd.DataFrame, before: pd.Timestamp) -> int:
    """Direction state at the close of `before`: last event trough -> 1; peak or none -> 0."""
    prior = events[events.date <= before]
    if prior.empty:
        return 0
    return 1 if prior.iloc[-1]["type"] == "trough" else 0


# ---------------------------------------------------------------- targets and backtest
def build_targets(ohlcv: pd.DataFrame, years: list[dict], init: str) -> tuple[pd.Series, dict]:
    """years: list of {"fold": fold_row, "events": DataFrame} sorted by test year.

    Returns the target series on [day0] + test dates and bookkeeping.
    """
    dates = ohlcv.index
    first_start = pd.Timestamp(years[0]["fold"]["test_start"])
    last_end = pd.Timestamp(years[-1]["fold"]["test_end"])
    test_mask = (dates >= first_start) & (dates <= last_end)
    test_dates = dates[test_mask]
    day0_pos = dates.searchsorted(first_start) - 1
    if day0_pos < 0:
        raise ValueError("no trading day before first test year")
    sim_index = dates[day0_pos:][: len(test_dates) + 1]
    if not sim_index[1:].equals(test_dates):
        raise ValueError("test dates are not contiguous")
    target = pd.Series(np.nan, index=sim_index)
    n_events, inits = 0, {}
    for y in years:
        f = y["fold"]
        t0, t1 = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        year_dates = sim_index[(sim_index >= t0) & (sim_index <= t1)]
        prev_close = sim_index[sim_index.get_loc(year_dates[0]) - 1]
        if init == "state":
            w0 = init_state(y["events"], prev_close)
        elif init == "cash":
            w0 = 0
        else:
            raise ValueError(init)
        inits[f["fold_id"]] = w0
        ev = y["events"][(y["events"].date >= t0) & (y["events"].date <= t1)]
        n_events += len(ev)
        sig = pd.Series(np.nan, index=year_dates)
        for dt, typ in zip(ev.date, ev["type"]):
            sig.loc[dt] = 1.0 if typ == "trough" else 0.0
        sig = sig.ffill().fillna(w0)
        target.loc[year_dates] = sig.values
        target.loc[prev_close] = w0  # year Y's initial position, set at the close before Y
    if target.isna().any():
        raise ValueError("unfilled target")
    return target, {"n_events": n_events, "inits": inits, "n_test_days": len(test_dates)}


def backtest(ohlcv: pd.DataFrame, target: pd.Series, cost: float):
    px = ohlcv.loc[target.index, ["Open", "Close"]]
    rule = simulate_positions(px, target, cost=cost, initial_position=0.0)
    bh = simulate_positions(px, pd.Series(1.0, index=target.index), cost=cost, initial_position=0.0)
    return rule, bh


def describe(res, n_test_days: int) -> dict:
    years = n_test_days / 252.0
    return {"ann_return": cagr(res.equity_curve), "ann_vol": annualized_volatility(res.daily_returns),
            "sharpe": sharpe_ratio(res.daily_returns), "max_drawdown": max_drawdown(res.equity_curve),
            "round_trips_per_year": len(res.trades) / years,
            "holding_day_share": float(res.daily["w_new"].mean()),
            "avg_holding_days": res.avg_holding_period}


def by_year(rule, bh) -> pd.DataFrame:
    rows = []
    for name, res in (("rule", rule), ("bh", bh)):
        r = res.daily_returns
        for y, ry in r.groupby(r.index.year):
            eq = np.r_[1.0, np.cumprod(1.0 + ry.to_numpy())]
            dd = eq / np.maximum.accumulate(eq) - 1.0
            rows.append({"strategy": name, "year": int(y), "annual_return": float(eq[-1] - 1.0),
                         "max_drawdown_in_year": float(dd.min())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
def _sharpe_rows(samples: np.ndarray) -> np.ndarray:
    """Exactly the fyp_rl.statistics resampled Sharpe (ddof=0, sqrt(252), 0 when std == 0)."""
    m = samples.mean(axis=1)
    s = samples.std(axis=1, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, np.sqrt(252.0) * m / s, 0.0)


def _masked_sharpe_rows(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.empty(samples.shape[0])
    for i in range(samples.shape[0]):
        x = samples[i][mask[i]]
        if len(x) == 0:
            out[i] = 0.0
            continue
        s = x.std(ddof=0)
        out[i] = np.sqrt(252.0) * x.mean() / s if s > 0 else 0.0
    return out


def joint_block_bootstrap(pairs: dict[str, tuple[pd.Series, pd.Series]], *, block_length: int,
                          n_repetitions: int, seed: int, chunk: int = 250) -> dict:
    """Joint moving-block bootstrap of ΔSharpe on the union calendar.

    Index generation is identical to fyp_rl.statistics.moving_block_bootstrap_sharpe_diff.
    Returns per-ticker Δ* arrays (n_repetitions,) and the equal-weighted Δ*.
    """
    union = sorted(set().union(*(set(r.index) for r, _ in pairs.values())))
    union = pd.DatetimeIndex(union)
    n = len(union)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    starts = rng.integers(0, max_start + 1, size=(n_repetitions, n_blocks))
    offsets = np.arange(block_length)
    out = {}
    for ticker, (rule, bh) in pairs.items():
        aligned = pd.concat([rule.rename("s"), bh.rename("b")], axis=1, join="inner").dropna()
        s = aligned["s"].reindex(union).to_numpy()
        b = aligned["b"].reindex(union).to_numpy()
        mask_full = ~np.isnan(s)
        full = bool(mask_full.all())
        diffs = np.empty(n_repetitions)
        for c0 in range(0, n_repetitions, chunk):
            idx = (starts[c0:c0 + chunk, :, None] + offsets[None, None, :]).reshape(len(starts[c0:c0 + chunk]), -1)[:, :n]
            if full:
                diffs[c0:c0 + chunk] = _sharpe_rows(s[idx]) - _sharpe_rows(b[idx])
            else:
                m = mask_full[idx]
                diffs[c0:c0 + chunk] = _masked_sharpe_rows(s[idx], m) - _masked_sharpe_rows(b[idx], m)
        out[ticker] = diffs
    ew = np.mean(np.vstack([out[t] for t in pairs]), axis=0)
    return {"per_ticker": out, "ew": ew, "n_union": n}


def two_sided_p(draws: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0))))


def percentile_ci(draws: np.ndarray, ci=(2.5, 97.5)) -> tuple[float, float]:
    lo, hi = np.percentile(draws, list(ci))
    return float(lo), float(hi)


def holm(pvalues: dict[str, float], alpha: float) -> pd.DataFrame:
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    rows, still, running = [], True, 0.0
    for rank, (name, p) in enumerate(items, start=1):
        thr = alpha / (m - rank + 1)
        running = max(running, min(1.0, (m - rank + 1) * p))
        reject = still and p <= thr
        still = reject
        rows.append({"ticker": name, "p": p, "holm_rank": rank, "holm_threshold": thr,
                     "holm_adjusted_p": running, "holm_reject": reject})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- forward-return diagnostic
def forward_returns(ohlcv: pd.DataFrame, events: pd.DataFrame, horizons) -> pd.DataFrame:
    pos = ohlcv.index.get_indexer(pd.to_datetime(events.date))
    o, c = ohlcv["Open"].to_numpy(), ohlcv["Close"].to_numpy()
    rows = []
    for p, (dt, typ) in zip(pos, zip(events.date, events["type"])):
        for H in horizons:
            ok = p + H < len(c)
            rows.append({"date": pd.Timestamp(dt), "year": pd.Timestamp(dt).year, "type": typ, "H": H,
                         "fwd_log_return": float(np.log(c[p + H] / o[p + 1])) if ok else np.nan})
    return pd.DataFrame(rows)


def d_stat(df: pd.DataFrame) -> float:
    t = df.loc[df.type == "trough", "fwd_log_return"].dropna()
    p = df.loc[df.type == "peak", "fwd_log_return"].dropna()
    if t.empty or p.empty:
        return np.nan
    return float(t.mean() - p.mean())


def year_block_bootstrap_d(df: pd.DataFrame, years: list[int], *, n_repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {y: g for y, g in df.groupby("year")}
    out = np.empty(n_repetitions)
    for i in range(n_repetitions):
        draw = rng.choice(years, size=len(years), replace=True)
        parts = [groups[y] for y in draw if y in groups]
        out[i] = d_stat(pd.concat(parts)) if parts else np.nan
    return out
