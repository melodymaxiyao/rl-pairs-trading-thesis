"""Supplemental accuracy diagnostics; unchanged detector/selection primitives."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from experiments.turning_point_detection.new_stage1.slope_accuracy import (
    evaluate_fold, reference_events, score_events, generate_random_walk,
)
from experiments.turning_point_detection.new_stage1.run_slope_accuracy import (
    select_fold, scalar_score, macro_score, add_stitched, aggregate_selected,
)


def evaluate_fold_v2(prices, fold, config):
    """Only replace TEST cropping/scoring; TRAIN and detector are unchanged."""
    result = evaluate_fold(prices, fold, config)
    result["test_scores_v1"] = result["test_scores"].copy()
    result["bounds_v1"] = result["bounds"].copy()
    index = prices.index
    for h_ref in config["h_ref_grid"]:
        radius = int(config["reference_radius_multiplier"] * h_ref)
        begin = int(index.searchsorted(pd.Timestamp(fold["test_start"])))
        segment_end = int(index.searchsorted(pd.Timestamp(fold["test_end"]), side="right") - 1)
        end = min(segment_end, len(prices) - 1 - radius)
        if end < begin:
            raise ValueError("No valid test interval with available retrospective context")
        result["bounds"][(h_ref, "test")] = (begin, end)
        all_ref = reference_events(result["reference_slopes"][h_ref])
        refs = all_ref[(all_ref[:, 0] >= begin) & (all_ref[:, 0] <= end)]
        result["references"][(h_ref, "test")] = refs
        for params, events in result["events"].items():
            det = events[(events[:, 0] >= begin) & (events[:, 0] <= end)]
            scores = score_events(det, refs, l_pre=config["match_l_pre"],
                                  max_delay=int(config["match_m_multiplier"] * h_ref),
                                  quantiles=config["delay_quantiles"])
            for kind, score in scores.items():
                result["test_scores"][(h_ref, *params, kind)] = score
    return result


def null_seed(base_seed, source, repeat, ticker, fold_id=None):
    if source == "B0":
        # Byte-identical seed derivation to the original iid-normal baseline.
        identity = f"{base_seed}:{repeat}:{ticker}:{fold_id}"
    elif source in ("B1", "B2"):
        identity = f"{base_seed}:{source}:{repeat}:{ticker}"
    else:
        raise ValueError(source)
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")


def sign_flip_returns(returns, seed):
    """Centered magnitudes are preserved bit for bit, before path integration."""
    returns = pd.Series(returns, copy=True)
    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Finite observed returns required")
    mu = float(returns.mean())
    centered = returns.to_numpy() - mu
    signs = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), len(returns))
    transformed = signs * centered
    output = pd.Series(mu + transformed, index=returns.index, name=returns.name)
    return output, {"mu": mu, "signs": signs, "original_centered": centered, "surrogate_centered": transformed}


def shuffle_returns_within_year(returns, seed):
    returns = pd.Series(returns, copy=True)
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.empty or not np.isfinite(returns).all():
        raise ValueError("Finite returns on a daily DatetimeIndex required")
    rng = np.random.default_rng(seed)
    shuffled = returns.copy()
    for year in sorted(set(returns.index.year)):
        positions = np.flatnonzero(returns.index.year == year)
        shuffled.iloc[positions] = rng.permutation(returns.iloc[positions].to_numpy())
    return shuffled


def empirical_null_path(prices, source, seed):
    returns = prices.diff().iloc[1:]
    if source == "B1":
        transformed, details = sign_flip_returns(returns, seed)
        mean = details["mu"]
    elif source == "B2":
        transformed = shuffle_returns_within_year(returns, seed)
        mean = float(returns.mean())
    else:
        raise ValueError("Empirical null must be B1 or B2")
    values = np.r_[prices.iloc[0], prices.iloc[0] + np.cumsum(transformed.to_numpy())]
    path = pd.Series(values, index=prices.index, name=prices.name)
    audit = {"source": source, "seed": seed, "return_mean_full_series": mean,
             "realized_surrogate_return_mean": float(transformed.mean()),
             "n_prices": len(prices), "n_returns": len(returns),
             "return_scope_start": str(returns.index[0].date()), "return_scope_end": str(returns.index[-1].date())}
    return path, audit


def evaluate_selected(price_map, calendar, config, *, source="real", repeat=-1,
                      collect_v1=False, collect_results=False, collect_train_grid=False):
    """Select solely on unchanged TRAIN, freeze to revised TEST, then stitch."""
    stitched, old_stitched, selected_rows, bounds_rows, train_rows, retained = {}, {}, [], [], [], {}
    from experiments.turning_point_detection.new_stage1.run_slope_accuracy import gather_grid_rows
    for fold_id, group in calendar.groupby("fold_id", sort=True):
        folds = {row["ticker"]: row for row in group.to_dict("records")}
        current_prices = {ticker: price_map[(fold_id, ticker)] if (fold_id, ticker) in price_map else price_map[ticker] for ticker in folds}
        results = {ticker: evaluate_fold_v2(current_prices[ticker], fold, config) for ticker, fold in folds.items()}
        if collect_results:
            retained[fold_id] = results
        if collect_train_grid:
            train_rows.extend(gather_grid_rows(results, folds, fold_id, config, "train"))
        for h_ref in config["h_ref_grid"]:
            params, chosen = select_fold(results, config, h_ref)
            test = macro_score([value["test_scores"][(h_ref, *params, "combined")] for value in results.values()], config["delay_quantiles"])
            selected_rows.append({"source": source, "repeat": repeat, "h_ref": h_ref, "fold_id": fold_id,
                                  "h": params[0], "d": params[1], "k": params[2],
                                  "train_f1": chosen["f1"], "test_f1": test["f1"],
                                  "train_minus_test_f1": chosen["f1"] - test["f1"],
                                  "train_delay_median": chosen["delay_median"], "test_delay_median": test["delay_median"],
                                  "n_tickers": len(results), "selection_tickers": ";".join(results)})
            for ticker, result in results.items():
                add_stitched(stitched, ticker, fold_id, result, h_ref, params)
                if collect_v1:
                    for kind in ("peak", "trough", "combined"):
                        score = result["test_scores_v1"][(h_ref, *params, kind)].copy()
                        score["fold_id"] = fold_id
                        old_stitched.setdefault((h_ref, ticker, kind), []).append(score)
                index = current_prices[ticker].index
                for segment in ("train", "test"):
                    begin, end = result["bounds"][(h_ref, segment)]
                    _, old_end = result["bounds_v1"][(h_ref, segment)]
                    bounds_rows.append({"source": source, "repeat": repeat, "ticker": ticker, "fold_id": fold_id,
                                        "h_ref": h_ref, "segment": segment, "evaluation_start": str(index[begin].date()),
                                        "evaluation_end_v1": str(index[old_end].date()), "evaluation_end_v2": str(index[end].date()),
                                        "evaluation_sessions_v1": old_end - begin + 1, "evaluation_sessions_v2": end - begin + 1,
                                        "added_sessions": end - old_end, "reference_context_last_date": str(index[end + int(config["reference_radius_multiplier"] * h_ref)].date())})
        print(f"{source} repeat={repeat} evaluated {fold_id}; tickers={','.join(results)}", flush=True)
    summary = aggregate_selected(stitched, config)
    summary["source"] = source; summary["repeat"] = repeat
    return {"summary": summary, "v1_summary": aggregate_selected(old_stitched, config) if collect_v1 else None,
            "selected": pd.DataFrame(selected_rows), "bounds": pd.DataFrame(bounds_rows),
            "results": retained, "train_grid": pd.DataFrame(train_rows)}


def compare_versions(v1, v2):
    keys = ["h_ref", "ticker", "type"]
    fields = ["n_ref", "n_det", "n_match", "precision", "recall", "f1", "delay_q25", "delay_median", "delay_q75", "n_test_years"]
    comparison = v1[keys + fields].merge(v2[keys + fields], on=keys, how="outer", validate="one_to_one", suffixes=("_v1", "_v2"))
    for field in fields:
        comparison[f"{field}_v2_minus_v1"] = comparison[f"{field}_v2"] - comparison[f"{field}_v1"]
    return comparison


def midrank_percentile(values, actual):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(actual):
        return np.nan
    return float((np.sum(values < actual) + .5 * np.sum(values == actual)) / len(values))


def compare_nulls(samples, real, config):
    rows = []
    for actual in real.to_dict("records"):
        row = {"h_ref": actual["h_ref"], "ticker": actual["ticker"], "type": actual["type"],
               "real_f1": actual["f1"], "real_delay_median": actual["delay_median"], "R": config["rw_repetitions"]}
        for source in ("B0", "B1", "B2"):
            group = samples.loc[(samples.source == source) & (samples.h_ref == actual["h_ref"]) & (samples.ticker == actual["ticker"]) & (samples.type == actual["type"])]
            if len(group) != config["rw_repetitions"]:
                raise AssertionError("Missing null repetitions")
            for field, real_field in (("f1", "f1"), ("delay", "delay_median")):
                values = group[real_field].dropna().to_numpy()
                qs = np.quantile(values, config["null_quantiles"]) if len(values) else [np.nan] * 3
                prefix = source.lower() + "_" + field
                for suffix, value in zip(("q05", "median", "q95"), qs):
                    row[prefix + "_" + suffix] = value
                row[f"real_{field}_{source.lower()}_percentile"] = midrank_percentile(values, actual[real_field])
                row[prefix + "_n_valid"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def train_test_gaps(selected):
    """Per-repeat fold macros, then equal-fold means, then repeat distribution."""
    values = selected.copy()
    values["aggregation"] = "fold_macro"
    values["n_folds"] = 1
    mean_rows = []
    for (source, repeat, h_ref), group in values.groupby(["source", "repeat", "h_ref"], sort=True):
        row = {"source": source, "repeat": repeat, "h_ref": h_ref, "fold_id": "across_fold_mean",
               "aggregation": "mean_of_fold_macros", "n_folds": len(group)}
        for field in ("train_f1", "test_f1", "train_minus_test_f1"):
            row[field] = float(group[field].mean())
        mean_rows.append(row)
    values = pd.concat([values, pd.DataFrame(mean_rows)], ignore_index=True)
    rows = []
    for (source, h_ref, fold_id, aggregation), group in values.groupby(["source", "h_ref", "fold_id", "aggregation"], sort=True):
        row = {"source": source, "h_ref": h_ref, "fold_id": fold_id, "aggregation": aggregation,
               "n_repetitions": group.repeat.nunique(), "n_folds": int(group.n_folds.iloc[0])}
        for field in ("train_f1", "test_f1", "train_minus_test_f1"):
            row[field + "_mean"] = float(group[field].mean())
            row[field + "_median"] = float(group[field].median())
            row[field + "_q05"] = float(group[field].quantile(.05))
            row[field + "_q95"] = float(group[field].quantile(.95))
        row["interpretation_note"] = "fold宏F1后跨折等权平均；不是拼接test F1；真实数据只有一次实现"
        rows.append(row)
    return pd.DataFrame(rows)
