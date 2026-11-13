"""Adjusted daily OHLCV data loading with validation and cache-merge semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _raw_path(ticker: str, raw_dir: str | Path) -> Path:
    return Path(raw_dir) / f"{ticker.lower()}_daily_adjusted_ohlcv.csv"


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce schema invariants: sorted unique dates, positive OHLC, no missing OHLC."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in OHLCV data")

    ohlc_cols = ["Open", "High", "Low", "Close"]
    df = df.dropna(subset=ohlc_cols)
    non_positive = (df[ohlc_cols] <= 0).any(axis=1)
    if non_positive.any():
        df = df.loc[~non_positive]

    if df["Volume"].isna().any():
        df["Volume"] = df["Volume"].fillna(0.0)

    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index is not sorted ascending after cleaning")
    if df.index.has_duplicates:
        raise ValueError("OHLCV index contains duplicate dates after cleaning")

    return df[REQUIRED_COLUMNS]


def _covers_request(cached: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    if cached.empty:
        return False
    return cached.index.min() <= start_ts and cached.index.max() >= end_ts


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load adjusted OHLCV data for `ticker` covering [start, end], using a local cache.

    Falls back to yfinance (auto_adjust=True) if the cache is missing or does not
    fully cover the requested range. Newly downloaded data is merged with any
    existing cache (dedup + sort), never overwritten wholesale. Raises on
    download failure instead of fabricating data.
    """
    raw_path = _raw_path(ticker, raw_dir)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    today_ts = pd.Timestamp.today().normalize()
    effective_end = min(end_ts, today_ts)

    cached = None
    if raw_path.exists():
        cached = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        cached = _validate_schema(cached)
        if _covers_request(cached, start_ts, effective_end):
            return cached.loc[(cached.index >= start_ts) & (cached.index <= effective_end)].copy()

    try:
        downloaded = yf.download(
            ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(effective_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to download OHLCV data for {ticker}: {exc}") from exc

    if downloaded is None or downloaded.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} in range {start} to {effective_end.date()}; "
            "refusing to fabricate data."
        )

    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    downloaded = downloaded[REQUIRED_COLUMNS].copy()
    downloaded = _validate_schema(downloaded)

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, downloaded])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = downloaded

    merged = _validate_schema(merged)
    merged.to_csv(raw_path, date_format="%Y-%m-%d")

    result = merged.loc[(merged.index >= start_ts) & (merged.index <= effective_end)].copy()
    if result.empty:
        raise RuntimeError(
            f"No OHLCV rows for {ticker} after filtering to {start}..{effective_end.date()}"
        )
    return result
