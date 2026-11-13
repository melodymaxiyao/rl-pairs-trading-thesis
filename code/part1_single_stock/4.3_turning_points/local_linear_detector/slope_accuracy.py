"""Non-trading slope-turn accuracy evaluation; references never drive events."""
from __future__ import annotations

import itertools
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from experiments.single_stock.run_single_stock_walkforward import resolve_effective_split
from experiments.turning_point_detection.new_stage1.kernel_smoothing import (
    causal_local_linear_slope, bilateral_local_linear_slope,
)
from experiments.turning_point_detection.shared.evaluation_metrics import precision_recall_f1


TURN_TYPES = {"peak": -1, "trough": 1}


def make_fold_calendar(
    ticker: str, index: pd.DatetimeIndex, *, train_years: int, test_years: int,
    step_years: int, as_of: str, tsla_min_train_start: str, market_calendar: str = "XNYS",
) -> pd.DataFrame:
    """Complete calendar years; fold IDs identify the shared test calendar year."""
    columns = ["ticker", "fold_id", "train_start", "train_end", "test_start", "test_end"]
    if index.empty:
        return pd.DataFrame(columns=columns)
    if min(train_years, test_years, step_years) < 1:
        raise ValueError("Year counts must be positive")
    earliest = max(index.min(), pd.Timestamp(tsla_min_train_start)) if ticker == "TSLA" else index.min()
    last_year = min(pd.Timestamp(as_of).year - 1, index.max().year)
    calendar = xcals.get_calendar(
        market_calendar, start=f"{earliest.year}-01-01", end=f"{index.max().year}-12-31"
    )
    first_sessions = calendar.sessions[calendar.sessions.year == earliest.year]
    first_year = earliest.year + int(earliest > first_sessions[0].tz_localize(None))
    last_sessions = calendar.sessions[calendar.sessions.year == last_year]
    if index.max() < last_sessions[-1].tz_localize(None):
        last_year -= 1
    rows = []
    for test_year in range(first_year + train_years, last_year - test_years + 2, step_years):
        split = {
            "train_start": f"{test_year-train_years}-01-01", "train_end": f"{test_year-1}-12-31",
            "test_start": f"{test_year}-01-01", "test_end": f"{test_year+test_years-1}-12-31",
        }
        resolved = resolve_effective_split(ticker, f"test_{test_year}", split, tsla_min_train_start)
        if resolved["actual_train_start"] != split["train_start"]:
            continue  # Never silently shorten a required complete three-year TRAIN.
        rows.append({"ticker": ticker, "fold_id": resolved["name"], **split})
    return pd.DataFrame(rows, columns=columns)


def detect_hysteresis_events(slopes: np.ndarray, *, tau: float, d: int) -> np.ndarray:
    """Return [confirmation_position, type_sign]; initialize without an event."""
    if not np.isfinite(tau) or tau < 0 or int(d) != d or d < 1:
        raise ValueError("Finite nonnegative tau and positive integral d required")
    signs = np.zeros(len(slopes), dtype=int)
    signs[slopes > tau] = 1
    signs[slopes < -tau] = -1
    if len(signs) < d:
        return np.empty((0, 2), dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(signs, d)
    eligible = (windows[:, 0] != 0) & np.all(windows == windows[:, :1], axis=1)
    positions = np.flatnonzero(eligible) + d - 1
    if not len(positions):
        return np.empty((0, 2), dtype=int)
    candidate_signs = signs[positions]
    changes = np.r_[False, candidate_signs[1:] != candidate_signs[:-1]]
    return np.column_stack((positions[changes], candidate_signs[changes])).astype(int)


def reference_events(slopes: np.ndarray) -> np.ndarray:
    """Unfiltered sign crossings; zeros retain the prior sign, NaNs break it."""
    result = []
    previous = 0
    for position, value in enumerate(slopes):
        if not np.isfinite(value):
            previous = 0
            continue
        sign = int(np.sign(value))
        if sign == 0:
            continue
        if previous and sign != previous:
            result.append((position, sign))
        previous = sign
    return np.asarray(result, dtype=int).reshape(-1, 2)


def greedy_match_events(
    detected: np.ndarray, reference: np.ndarray, *, l_pre: int, max_delay: int
) -> np.ndarray:
    """Chronological detection-first greedy match, same type, nearest |delay|.

    Equal-distance ties choose the earlier reference. Unlike the existing
    Lo-style dynamic-programming matcher, windows are asymmetric and there
    is no global cardinality optimization or monotonicity constraint.
    """
    if l_pre < 0 or max_delay < 0:
        raise ValueError("Matching limits must be nonnegative")
    detected = detected[np.argsort(detected[:, 0], kind="stable")]
    reference = reference[np.argsort(reference[:, 0], kind="stable")]
    used = np.zeros(len(reference), dtype=bool)
    matches = []
    for position, kind in detected:
        delta = position - reference[:, 0]
        candidates = np.flatnonzero(
            (~used) & (reference[:, 1] == kind) & (delta >= -l_pre) & (delta <= max_delay)
        )
        if len(candidates):
            best = candidates[np.argmin(np.abs(delta[candidates]))]
            used[best] = True
            matches.append((position, reference[best, 0], kind))
    return np.asarray(matches, dtype=int).reshape(-1, 3)


def stitch_scores(scores: list[dict], *, quantiles: list[float]) -> dict[str, Any]:
    """Aggregate matched counts/delays, never average annual F1s."""
    n_ref = sum(s["n_ref"] for s in scores)
    n_det = sum(s["n_det"] for s in scores)
    n_match = sum(s["n_match"] for s in scores)
    delays = [delay for score in scores for delay in score["delays"]]
    metrics = precision_recall_f1(n_match, n_det-n_match, n_ref-n_match)
    q = np.quantile(delays, quantiles) if delays else [np.nan] * len(quantiles)
    return {
        "n_ref": n_ref, "n_det": n_det, "n_match": n_match,
        "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["F1"],
        "delay_q25": float(q[0]), "delay_median": float(q[1]), "delay_q75": float(q[2]),
        "delays": delays,
    }


def score_events(detected, reference, *, l_pre, max_delay, quantiles):
    matches = greedy_match_events(detected, reference, l_pre=l_pre, max_delay=max_delay)
    result = {}
    for name, sign in {**TURN_TYPES, "combined": None}.items():
        chosen = matches if sign is None else matches[matches[:, 2] == sign]
        score = {
            "n_ref": len(reference) if sign is None else int(np.sum(reference[:, 1] == sign)),
            "n_det": len(detected) if sign is None else int(np.sum(detected[:, 1] == sign)),
            "n_match": len(chosen), "delays": (chosen[:, 0] - chosen[:, 1]).tolist(),
        }
        result[name] = stitch_scores([score], quantiles=quantiles)
    return result


def select_parameters(macro_train_scores: pd.DataFrame) -> tuple[int, int, float]:
    """F1 descending, then signed median delay, h, d, k ascending."""
    rows = macro_train_scores.copy()
    rows["f1"] = rows["f1"].fillna(0)
    rows["delay_median"] = rows["delay_median"].fillna(np.inf)
    best = rows.sort_values(["f1", "delay_median", "h", "d", "k"],
                            ascending=[False, True, True, True, True], kind="stable").iloc[0]
    return int(best.h), int(best.d), float(best.k)


def evaluate_fold(prices: pd.Series, fold: dict, config: dict) -> dict:
    """Calibrate on TRAIN, carry detector state continuously into TEST.

    Reference slopes use the full price index; evaluation-end trimming ensures
    retained TRAIN references cannot use TEST prices. Detector slopes reset at
    TRAIN start, hence no pre-TRAIN observations in a detector window.
    """
    index = prices.index
    start = index.searchsorted(pd.Timestamp(fold["train_start"]))
    stop = index.searchsorted(pd.Timestamp(fold["test_end"]), side="right")
    local = prices.iloc[start:stop]
    local_train = local.index <= pd.Timestamp(fold["train_end"])
    result = {"slopes": {}, "sigma_b": {}, "events": {}, "references": {},
              "reference_slopes": {}, "train_scores": {}, "test_scores": {}, "bounds": {}}
    for h in config["h_grid"]:
        b = causal_local_linear_slope(local, bandwidth_h=h, window_multiplier=config["window_multiplier"])
        sigma = float(b.loc[local_train].std(ddof=config["std_ddof"]))
        if not np.isfinite(sigma):
            raise ValueError("Insufficient finite TRAIN slopes")
        full = np.full(len(prices), np.nan)
        full[start:stop] = b.to_numpy()
        result["slopes"][h] = full
        result["sigma_b"][h] = sigma
        for d, k in itertools.product(config["d_grid"], config["k_grid"]):
            events = detect_hysteresis_events(b.to_numpy(), tau=k*sigma, d=d)
            events[:, 0] += start
            result["events"][(h, d, k)] = events
    for h_ref in config["h_ref_grid"]:
        radius = int(config["reference_radius_multiplier"] * h_ref)
        b_ref = bilateral_local_linear_slope(prices, bandwidth_h=h_ref,
                                             radius_multiplier=config["reference_radius_multiplier"])
        result["reference_slopes"][h_ref] = b_ref.to_numpy()
        all_ref = reference_events(b_ref.to_numpy())
        for segment in ("train", "test"):
            begin = index.searchsorted(pd.Timestamp(fold[f"{segment}_start"]))
            end = index.searchsorted(pd.Timestamp(fold[f"{segment}_end"]), side="right") - 1 - radius
            if end < begin:
                raise ValueError("Segment too short after reference-tail exclusion")
            result["bounds"][(h_ref, segment)] = (int(begin), int(end))
            refs = all_ref[(all_ref[:, 0] >= begin) & (all_ref[:, 0] <= end)]
            result["references"][(h_ref, segment)] = refs
            for (h, d, k), events in result["events"].items():
                det = events[(events[:, 0] >= begin) & (events[:, 0] <= end)]
                scores = score_events(det, refs, l_pre=config["match_l_pre"],
                                      max_delay=int(config["match_m_multiplier"]*h_ref),
                                      quantiles=config["delay_quantiles"])
                for kind, metrics in scores.items():
                    result[f"{segment}_scores"][(h_ref, h, d, k, kind)] = metrics
    return result


def generate_random_walk(prices: pd.Series, fold: dict, *, seed: int, ddof: int):
    """Full-length i.i.d. path, calibrated only to this fold's TRAIN returns.

    Each fold has its own full-length path because overlapping folds generally
    have different calibrated means/variances. No TEST return is calibrated.
    """
    train = prices.loc[fold["train_start"]:fold["train_end"]]
    returns = train.diff().dropna()
    mean, std = float(returns.mean()), float(returns.std(ddof=ddof))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Insufficient TRAIN returns for random-walk calibration")
    innovations = np.random.default_rng(seed).normal(mean, std, len(prices)-1)
    path = np.r_[prices.iloc[0], prices.iloc[0] + np.cumsum(innovations)]
    return pd.Series(path, index=prices.index, name=prices.name), {
        "train_return_mean": mean, "train_return_std": std, "n_prices": len(prices),
        "n_train_returns": len(returns), "seed": seed,
    }
