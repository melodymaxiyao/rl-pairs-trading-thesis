"""Run accuracy-only evaluation. --smoke limits scope through JSON config."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fyp_rl.data import load_ohlcv  # noqa: E402
from experiments.turning_point_detection.new_stage1.kernel_smoothing import (  # noqa: E402
    kernel_mean_lag, local_linear_slope_weights, causal_local_linear_slope,
    bilateral_local_linear_slope,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy import (  # noqa: E402
    TURN_TYPES, make_fold_calendar, evaluate_fold, select_parameters, stitch_scores,
    generate_random_walk, reference_events, detect_hysteresis_events,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "slope_accuracy.json"
TRANSPARENCY_NOTE = "仅供透明度展示，不得用于选参"
METRICS = ["n_ref", "n_det", "n_match", "precision", "recall", "f1",
           "delay_q25", "delay_median", "delay_q75"]


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_accuracy_config(path=DEFAULT_CONFIG, *, smoke=False):
    config = json.loads(Path(path).read_text())
    if config["input_series"] != "log_adjusted_close":
        raise ValueError("Expected log_adjusted_close input")
    if config["selection_universe"] not in ("available_tickers", "all_tickers"):
        raise ValueError("Unknown selection_universe")
    if config["delay_quantiles"] != [0.25, 0.5, 0.75]:
        raise ValueError("Delay columns require quartiles and median")
    if config["null_quantiles"] != [0.05, 0.5, 0.95]:
        raise ValueError("Null columns require 5/50/95 percentiles")
    if config["percentile_ties"] != "midrank":
        raise ValueError("Only midrank empirical percentile is supported")
    if smoke:
        for field in ("tickers", "h_ref_grid", "rw_repetitions", "output_dir"):
            config[field] = config["smoke"][field]
        config["only_test_years"] = config["smoke"]["test_years"]
    return config


def load_prices(ticker, config):
    """Read cache bounds only; all prices come from the shared OHLCV loader."""
    raw_dir = repo_path(config["raw_dir"])
    cached_path = raw_dir / f"{ticker.lower()}_daily_adjusted_ohlcv.csv"
    if not cached_path.exists():
        raise FileNotFoundError(f"Need an existing data cache to determine actual calendar: {cached_path}")
    dates = pd.DatetimeIndex(pd.read_csv(cached_path, usecols=[0]).iloc[:, 0])
    start = max(dates.min(), pd.Timestamp(config["tsla_min_train_start"])) if ticker == "TSLA" else dates.min()
    end = min(dates.max(), pd.Timestamp(config["as_of"]))
    frame = load_ohlcv(ticker, str(start.date()), str(end.date()), raw_dir=raw_dir)
    return pd.Series(np.log(frame["Close"]), name="log_adjusted_close")


def scalar_score(score):
    return {field: score[field] for field in METRICS}


def macro_score(scores, quantiles):
    """Equal-ticker precision/recall/F1; pooled matched-event delay quantiles."""
    result = stitch_scores(scores, quantiles=quantiles)
    for field in ("precision", "recall", "f1"):
        # Undefined metrics are zero for macro scoring, retaining equal weights.
        result[field] = float(np.mean([s[field] if np.isfinite(s[field]) else 0 for s in scores]))
    return result


def select_fold(results, config, h_ref):
    rows = []
    for h, d, k in itertools.product(config["h_grid"], config["d_grid"], config["k_grid"]):
        scores = [r["train_scores"][(h_ref, h, d, k, "combined")] for r in results.values()]
        rows.append({"h": h, "d": d, "k": k,
                     **scalar_score(macro_score(scores, config["delay_quantiles"]))})
    frame = pd.DataFrame(rows)
    selected = select_parameters(frame)
    chosen = frame.loc[(frame.h == selected[0]) & (frame.d == selected[1]) & (frame.k == selected[2])].iloc[0]
    return selected, chosen.to_dict()


def gather_grid_rows(results, calendar_rows, fold_id, config, segment):
    rows = []
    participants = list(results)
    for h_ref, h, d, k, kind in itertools.product(
        config["h_ref_grid"], config["h_grid"], config["d_grid"], config["k_grid"],
        ["peak", "trough", "combined"],
    ):
        scores = []
        metadata = {"fold_id": fold_id, "h_ref": h_ref, "h": h, "d": d, "k": k, "type": kind}
        for ticker, result in results.items():
            score = result[f"{segment}_scores"][(h_ref, h, d, k, kind)]
            scores.append(score)
            rows.append({"ticker": ticker, **metadata, **scalar_score(score),
                         "n_tickers": 1, "selection_tickers": ticker,
                         "usage_note": "TRAIN selection only" if segment == "train" else TRANSPARENCY_NOTE})
        rows.append({"ticker": "equal_ticker_macro", **metadata,
                     **scalar_score(macro_score(scores, config["delay_quantiles"])),
                     "n_tickers": len(participants), "selection_tickers": ";".join(participants),
                     "usage_note": "TRAIN selection only" if segment == "train" else TRANSPARENCY_NOTE})
    return rows


def aggregate_selected(stitched, config):
    rows = []
    for (h_ref, ticker, kind), scores in stitched.items():
        score = stitch_scores(scores, quantiles=config["delay_quantiles"])
        rows.append({"h_ref": h_ref, "ticker": ticker, "type": kind,
                     **scalar_score(score), "n_test_years": len(scores)})
    for h_ref, kind in itertools.product(config["h_ref_grid"], ["peak", "trough", "combined"]):
        scores = []
        for ticker in config["tickers"]:
            key = (h_ref, ticker, kind)
            if key in stitched:
                scores.append(stitch_scores(stitched[key], quantiles=config["delay_quantiles"]))
        if scores:
            rows.append({"h_ref": h_ref, "ticker": "equal_ticker_macro", "type": kind,
                         **scalar_score(macro_score(scores, config["delay_quantiles"])),
                         "n_test_years": len({s["fold_id"] for v in stitched.values() for s in v})})
    return pd.DataFrame(rows)


def add_stitched(stitched, ticker, fold_id, result, h_ref, params):
    for kind in ("peak", "trough", "combined"):
        score = result["test_scores"][(h_ref, *params, kind)].copy()
        score["fold_id"] = fold_id
        stitched.setdefault((h_ref, ticker, kind), []).append(score)


def synthetic_v_table(config):
    """Vertex at position leg_days; then exactly leg_days positive increments."""
    leg, slope = config["synthetic_leg_days"], config["synthetic_slope"]
    increments = np.r_[np.full(leg, -slope), np.full(leg, slope)]
    p = pd.Series(np.r_[0.0, np.cumsum(increments)])
    rows = []
    for h in config["h_grid"]:
        b = causal_local_linear_slope(p, bandwidth_h=h, window_multiplier=config["window_multiplier"])
        crossing = reference_events(b.to_numpy())
        turns = crossing[crossing[:, 1] == 1]
        sign_day = int(turns[0, 0]) if len(turns) else None
        for d in config["d_grid"]:
            events = detect_hysteresis_events(b.to_numpy(), tau=config["synthetic_k"]*b.std(ddof=config["std_ddof"]), d=d)
            troughs = events[events[:, 1] == 1]
            confirmation = int(troughs[0, 0]) if len(troughs) else None
            rows.append({"h": h, "d": d, "k": config["synthetic_k"], "vertex_position": leg,
                         "sign_change_position": sign_day, "confirmation_position": confirmation,
                         "slope_sign_delay_days": sign_day-leg if sign_day is not None else np.nan,
                         "confirmation_delay_days": confirmation-leg if confirmation is not None else np.nan})
    return pd.DataFrame(rows)


def weight_audit(config):
    rows = []
    for side, grid, multiplier in (
        ("causal", config["h_grid"], config["window_multiplier"]),
        ("bilateral", config["h_ref_grid"], config["reference_radius_multiplier"]),
    ):
        for h in grid:
            x, weights = local_linear_slope_weights(h, side=side, window_multiplier=multiplier)
            for offset, weight in zip(x, weights):
                rows.append({"side": side, "h": h, "offset_days": int(offset), "slope_weight_l": weight})
    return pd.DataFrame(rows)


def extend_smoothing_summary(config):
    """Append only kernel_mean_lag; preserve every existing CSV field string."""
    path = repo_path(config["smoothing_summary"])
    smoothing_config = json.loads(repo_path(config["smoothing_config"]).read_text())
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["kernel_mean_lag"] = [format(kernel_mean_lag(float(h), smoothing_config["window_multiplier"]), ".12g")
                                for h in frame["h"]]
    frame.to_csv(path, index=False)


def null_summary(samples, actual, config):
    rows = []
    for real in actual.to_dict("records"):
        chosen = samples.loc[(samples.h_ref == real["h_ref"]) & (samples.ticker == real["ticker"])
                             & (samples.type == real["type"])]
        f1 = chosen.f1.dropna().to_numpy()
        delays = chosen.delay_median.dropna().to_numpy()
        fq = np.quantile(f1, config["null_quantiles"]) if len(f1) else [np.nan]*3
        dq = np.quantile(delays, config["null_quantiles"]) if len(delays) else [np.nan]*3
        position = (np.mean(f1 < real["f1"]) + .5*np.mean(f1 == real["f1"])) if len(f1) and np.isfinite(real["f1"]) else np.nan
        rows.append({"h_ref": real["h_ref"], "ticker": real["ticker"], "type": real["type"],
                     "real_f1": real["f1"], "real_delay_median": real["delay_median"],
                     "null_f1_q05": fq[0], "null_f1_median": fq[1], "null_f1_q95": fq[2],
                     "null_delay_q05": dq[0], "null_delay_median": dq[1], "null_delay_q95": dq[2],
                     "real_f1_null_percentile": position, "R": config["rw_repetitions"],
                     "n_valid_f1": len(f1), "n_valid_delay": len(delays)})
    return pd.DataFrame(rows)


def run(config, *, smoke=False):
    output = repo_path(config["output_dir"])
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    price_map = {ticker: load_prices(ticker, config) for ticker in config["tickers"]}
    calendars = [make_fold_calendar(
        ticker, p.index, train_years=config["train_years"], test_years=config["test_years"],
        step_years=config["step_years"], as_of=config["as_of"],
        tsla_min_train_start=config["tsla_min_train_start"], market_calendar=config["market_calendar"],
    ) for ticker, p in price_map.items()]
    calendar = pd.concat(calendars, ignore_index=True)
    if "only_test_years" in config:
        calendar = calendar.loc[pd.to_datetime(calendar.test_start).dt.year.isin(config["only_test_years"])]
    if config["selection_universe"] == "all_tickers":
        counts = calendar.groupby("fold_id").ticker.nunique()
        calendar = calendar.loc[calendar.fold_id.isin(counts[counts == len(config["tickers"])].index)]
    if calendar.empty:
        raise ValueError("No complete eligible folds in selected scope")
    calendar = calendar.sort_values(["fold_id", "ticker"]).reset_index(drop=True)
    table_frames = {"fold_calendar": calendar, "slope_weights": weight_audit(config),
                    "synthetic_v_delay": synthetic_v_table(config)}
    train_rows, test_rows, ref_rows, det_rows, selected_rows, sigma_rows = [], [], [], [], [], []
    stitched, result_map, selected_map = {}, {}, {}
    for fold_id, group in calendar.groupby("fold_id", sort=True):
        folds = {row["ticker"]: row for row in group.to_dict("records")}
        results = {ticker: evaluate_fold(price_map[ticker], fold, config) for ticker, fold in folds.items()}
        result_map[fold_id] = results
        train_rows.extend(gather_grid_rows(results, folds, fold_id, config, "train"))
        test_rows.extend(gather_grid_rows(results, folds, fold_id, config, "test"))
        for ticker, result in results.items():
            index = price_map[ticker].index
            for h, sigma in result["sigma_b"].items():
                sigma_rows.append({"ticker": ticker, "fold_id": fold_id, "h": h, "sigma_b": sigma})
            for (h_ref, segment), events in result["references"].items():
                for position, kind in events:
                    ref_rows.append({"ticker": ticker, "h_ref": h_ref, "date": index[position],
                                     "type": "peak" if kind == -1 else "trough", "segment": segment, "fold_id": fold_id})
            for (h, d, k), events in result["events"].items():
                for position, kind in events:
                    date = index[position]
                    segment = "train" if date <= pd.Timestamp(folds[ticker]["train_end"]) else "test"
                    det_rows.append({"ticker": ticker, "fold_id": fold_id, "h": h, "d": d, "k": k,
                                     "date": date, "type": "peak" if kind == -1 else "trough", "segment": segment})
        for h_ref in config["h_ref_grid"]:
            params, score = select_fold(results, config, h_ref)
            selected_map[(fold_id, h_ref)] = params
            selected_rows.append({"h_ref": h_ref, "fold_id": fold_id, **score,
                                  "n_tickers": len(results), "selection_tickers": ";".join(results)})
            for ticker, result in results.items():
                add_stitched(stitched, ticker, fold_id, result, h_ref, params)
        print(f"Actual data evaluated: {fold_id}; tickers={','.join(results)}", flush=True)
    density = []
    for ticker, p in price_map.items():
        for h_ref in config["h_ref_grid"]:
            b_ref = bilateral_local_linear_slope(p, bandwidth_h=h_ref,
                                                  radius_multiplier=config["reference_radius_multiplier"])
            events = reference_events(b_ref.to_numpy())
            for year in sorted(set(p.index.year)):
                positions = events[np.isin(p.index[events[:, 0]].year, [year])]
                density.append({"ticker": ticker, "h_ref": h_ref, "year": year,
                                "n_peak": int(np.sum(positions[:, 1] == -1)),
                                "n_trough": int(np.sum(positions[:, 1] == 1)), "n_ref": len(positions)})
    table_frames.update({
        "reference_turns": pd.DataFrame(ref_rows, columns=["ticker", "h_ref", "date", "type", "segment", "fold_id"]),
        "reference_density": pd.DataFrame(density),
        "detector_events": pd.DataFrame(det_rows, columns=["ticker", "fold_id", "h", "d", "k", "date", "type", "segment"]),
        "train_grid_scores": pd.DataFrame(train_rows), "test_grid_scores": pd.DataFrame(test_rows),
        "selected_params_by_fold": pd.DataFrame(selected_rows), "sigma_b_by_fold": pd.DataFrame(sigma_rows),
        "test_accuracy_summary": aggregate_selected(stitched, config),
    })
    null_rows, null_selected, null_calibration = [], [], []
    for repeat in range(config["rw_repetitions"]):
        null_stitched = {}
        for fold_id, group in calendar.groupby("fold_id", sort=True):
            results = {}
            for fold in group.to_dict("records"):
                ticker = fold["ticker"]
                # Stable identity-based child seed; invariant to ticker/fold subset or ordering.
                identity = f"{config['seed']}:{repeat}:{ticker}:{fold_id}".encode()
                seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
                p, calibration = generate_random_walk(price_map[ticker], fold, seed=seed, ddof=config["std_ddof"])
                results[ticker] = evaluate_fold(p, fold, config)
                null_calibration.append({"repeat": repeat, "ticker": ticker, "fold_id": fold_id, **calibration})
            for h_ref in config["h_ref_grid"]:
                params, score = select_fold(results, config, h_ref)
                null_selected.append({"repeat": repeat, "h_ref": h_ref, "fold_id": fold_id, **score,
                                      "n_tickers": len(results), "selection_tickers": ";".join(results)})
                for ticker, result in results.items():
                    add_stitched(null_stitched, ticker, fold_id, result, h_ref, params)
        summary = aggregate_selected(null_stitched, config)
        summary["repeat"] = repeat
        null_rows.extend(summary.to_dict("records"))
        print(f"Null repetition completed: {repeat+1}/{config['rw_repetitions']}", flush=True)
    samples = pd.DataFrame(null_rows)
    table_frames["rw_null_samples"] = samples
    table_frames["rw_null_selected_params"] = pd.DataFrame(null_selected)
    table_frames["rw_null_calibration"] = pd.DataFrame(null_calibration)
    table_frames["rw_null_accuracy"] = null_summary(samples, table_frames["test_accuracy_summary"], config)
    for name, frame in table_frames.items():
        frame.to_csv(tables / f"{name}.csv", index=False, float_format="%.12g")
    from experiments.turning_point_detection.new_stage1.plot_slope_accuracy import plot_accuracy
    plots, skips = plot_accuracy(price_map, result_map, selected_map, config, output / "accuracy_plots")
    extend_smoothing_summary(config)
    inventory = [{"file": f"tables/{name}.csv", "rows": len(frame), "columns": list(frame.columns)}
                 for name, frame in table_frames.items()]
    manifest = {"mode": "smoke" if smoke else "full", "config": config,
                "tables": inventory, "plots": [str(p.relative_to(output)) for p in plots],
                "plot_skips": skips, "trading_evaluations": 0,
                "methodology": "accuracy-only; retrospective references never enter detector or threshold"}
    (output / "accuracy_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(inventory, indent=2, ensure_ascii=False))
    return table_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_accuracy_config(args.config, smoke=args.smoke), smoke=args.smoke)


if __name__ == "__main__":
    main()
