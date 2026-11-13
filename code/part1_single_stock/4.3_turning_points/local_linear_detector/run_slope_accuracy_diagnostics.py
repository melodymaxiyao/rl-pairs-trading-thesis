"""Additive A–E diagnostics. Run --smoke only unless the user launches full."""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "supplementary_runtime/mpl_cache"))

import numpy as np
import pandas as pd

from experiments.turning_point_detection.new_stage1.run_slope_accuracy import (
    load_accuracy_config, load_prices, repo_path,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy import make_fold_calendar, generate_random_walk
from experiments.turning_point_detection.new_stage1.slope_accuracy_diagnostics import (
    evaluate_selected, compare_versions, compare_nulls, train_test_gaps, null_seed, empirical_null_path,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy_vol_diagnostics import volatility_rows, plot_volatility

DEFAULT_CONFIG = HERE / "configs/slope_accuracy_diagnostics.json"
TABLE_NAMES = (
    "test_accuracy_summary_v2", "test_accuracy_v1_vs_v2", "null_comparison", "train_vs_test_gap",
    "per_fold_test_vs_vol", "test_accuracy_kfine", "fold_calendar_diagnostics", "evaluation_bounds_v2",
    "selected_params_diagnostics", "null_samples_diagnostics", "null_selected_params_diagnostics",
    "null_generation_diagnostics", "selected_params_kfine", "train_grid_scores_kfine",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=DEFAULT_CONFIG, smoke=False):
    extension = json.loads(Path(path).read_text())
    config = load_accuracy_config(repo_path(extension["base_config"]), smoke=False)
    config["rw_repetitions"] = extension["null_repetitions"]
    config["seed"] = extension["seed"]
    config["output_dir"] = extension["output_dir"]
    config["k_fine_grid"] = extension["k_fine_grid"]
    config["sensitivity_note"] = extension["sensitivity_note"]
    config["b1_diagnostic_repeat"] = extension["b1_diagnostic_repeat"]
    if extension["null_sources"] != ["B0", "B1", "B2"]:
        raise ValueError("All three requested nulls must run")
    if config["b1_diagnostic_repeat"] != 0:
        raise ValueError("B1 diagnostic implementation is the fixed repeat0")
    if smoke:
        scope = extension["smoke"]
        config.update({"tickers": scope["tickers"], "h_ref_grid": scope["h_ref_grid"],
                       "only_test_years": scope["only_test_years"], "rw_repetitions": scope["null_repetitions"],
                       "output_dir": extension["smoke_output_dir"]})
    return config, extension


def check_existing_hashes(reference):
    changed = [filename for filename, digest in reference.items() if not (REPO_ROOT / filename).is_file() or sha(REPO_ROOT / filename) != digest]
    if changed:
        raise AssertionError("Protected existing files changed: " + ", ".join(changed))


def calendar_for_prices(price_map, config):
    frames = [make_fold_calendar(ticker, prices.index, train_years=config["train_years"],
              test_years=config["test_years"], step_years=config["step_years"], as_of=config["as_of"],
              tsla_min_train_start=config["tsla_min_train_start"], market_calendar=config["market_calendar"])
              for ticker, prices in price_map.items()]
    calendar = pd.concat(frames, ignore_index=True)
    if "only_test_years" in config:
        calendar = calendar.loc[pd.to_datetime(calendar.test_start).dt.year.isin(config["only_test_years"])]
    if config["selection_universe"] == "all_tickers":
        counts = calendar.groupby("fold_id").ticker.nunique()
        calendar = calendar.loc[calendar.fold_id.isin(counts[counts == len(config["tickers"])].index)]
    if calendar.empty:
        raise ValueError("No complete folds")
    return calendar.sort_values(["fold_id", "ticker"]).reset_index(drop=True)


def validate_reconstructed_v1(summary, config, smoke):
    baseline = HERE / ("smoke_outputs/tables/test_accuracy_summary.csv" if smoke else "tables/test_accuracy_summary.csv")
    old = pd.read_csv(baseline)
    keys = ["h_ref", "ticker", "type"]
    fields = ["n_ref", "n_det", "n_match", "precision", "recall", "f1", "delay_q25", "delay_median", "delay_q75", "n_test_years"]
    left = summary.set_index(keys).sort_index()
    right = old.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise AssertionError("Old saved baseline scope differs from requested comparison; refusing a misleading v1 comparison")
    np.testing.assert_allclose(left[fields], right[fields], rtol=2e-11, atol=2e-11, equal_nan=True)
    return {"file": str(baseline.relative_to(REPO_ROOT)), "sha256": sha(baseline), "rows_checked": len(left)}


def run(config, extension, *, smoke=False):
    output = repo_path(config["output_dir"])
    tables = output / "tables"; plot_dir = output / "diagnostic_plots"
    proposed = [tables / f"{name}.csv" for name in TABLE_NAMES]
    proposed.append(output / "supplementary_diagnostics_manifest.json")
    for source in ("real", "B1"):
        suffix = "" if source == "real" else "_B1_repeat0"
        proposed.extend(plot_dir / f"test_{metric}_vs_vol_ratio_href{h_ref}{suffix}.png"
                        for metric in ("f1", "precision", "recall") for h_ref in config["h_ref_grid"])
    conflicts = [str(p) for p in proposed if p.exists()]
    if conflicts:
        raise FileExistsError("Existing outputs are immutable; choose a fresh --output-dir: " + ", ".join(conflicts))
    tables.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)
    with (output / "supplementary_pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected = {str(p.relative_to(REPO_ROOT)): sha(p) for p in HERE.rglob("*") if p.is_file() and p not in proposed and "supplementary_runtime" not in p.parts and p.name != "supplementary_pipeline.lock"}
        for ticker in config["tickers"]:
            path = repo_path(config["raw_dir"]) / f"{ticker.lower()}_daily_adjusted_ohlcv.csv"
            protected[str(path.relative_to(REPO_ROOT))] = sha(path)
        # Bounds are exactly the existing cache's dates; shared loader returns
        # cache contents without a download or rewrite. Hashes enforce this.
        prices = {ticker: load_prices(ticker, config) for ticker in config["tickers"]}
        check_existing_hashes(protected)
        calendar = calendar_for_prices(prices, config)
        actual = evaluate_selected(prices, calendar, config, collect_v1=True, collect_results=True)
        baseline_check = validate_reconstructed_v1(actual["v1_summary"], config, smoke)
        vol = volatility_rows(prices, calendar, actual, config, "real", -1)
        frames = {"test_accuracy_summary_v2": actual["summary"].drop(columns=["source", "repeat"]),
                  "test_accuracy_v1_vs_v2": compare_versions(actual["v1_summary"], actual["summary"]),
                  "fold_calendar_diagnostics": calendar, "evaluation_bounds_v2": actual["bounds"],
                  "selected_params_diagnostics": actual["selected"]}
        samples, selected, generations = [], [actual["selected"]], []
        for source in ("B0", "B1", "B2"):
            for repeat in range(config["rw_repetitions"]):
                null_prices = {}
                if source == "B0":
                    for fold in calendar.to_dict("records"):
                        ticker, fold_id = fold["ticker"], fold["fold_id"]
                        seed = null_seed(config["seed"], source, repeat, ticker, fold_id)
                        path, details = generate_random_walk(prices[ticker], fold, seed=seed, ddof=config["std_ddof"])
                        null_prices[(fold_id, ticker)] = path
                        generations.append({"source": source, "repeat": repeat, "ticker": ticker, "fold_id": fold_id, **details})
                else:
                    for ticker, observed in prices.items():
                        seed = null_seed(config["seed"], source, repeat, ticker)
                        path, details = empirical_null_path(observed, source, seed)
                        null_prices[ticker] = path
                        generations.append({"repeat": repeat, "ticker": ticker, "fold_id": "one_path_all_folds", **details})
                diagnostic = source == "B1" and repeat == config["b1_diagnostic_repeat"]
                evaluated = evaluate_selected(null_prices, calendar, config, source=source, repeat=repeat, collect_results=diagnostic)
                samples.append(evaluated["summary"]); selected.append(evaluated["selected"])
                if diagnostic:
                    vol.extend(volatility_rows(null_prices, calendar, evaluated, config, source, repeat))
                print(f"{source}: repetition {repeat + 1}/{config['rw_repetitions']} finished", flush=True)
        null_samples = pd.concat(samples, ignore_index=True)
        choices = pd.concat(selected, ignore_index=True)
        frames.update({"null_comparison": compare_nulls(null_samples, actual["summary"], config),
                       "train_vs_test_gap": train_test_gaps(choices), "per_fold_test_vs_vol": pd.DataFrame(vol),
                       "null_samples_diagnostics": null_samples, "null_selected_params_diagnostics": choices.loc[choices.source != "real"].reset_index(drop=True),
                       "null_generation_diagnostics": pd.DataFrame(generations)})
        fine_config = {**config, "k_grid": config["k_fine_grid"]}
        fine = evaluate_selected(prices, calendar, fine_config, source="real_kfine", collect_train_grid=True)
        fine_summary = fine["summary"].drop(columns=["source", "repeat"])
        fine_summary["analysis_label"] = config["sensitivity_note"]
        frames["test_accuracy_kfine"] = fine_summary
        frames["selected_params_kfine"] = fine["selected"].assign(analysis_label=config["sensitivity_note"])
        frames["train_grid_scores_kfine"] = fine["train_grid"].assign(analysis_label=config["sensitivity_note"])
        plots = plot_volatility(frames["per_fold_test_vs_vol"], config, plot_dir)
        check_existing_hashes(protected)
        for name, frame in frames.items():
            with (tables / f"{name}.csv").open("x") as handle:
                frame.to_csv(handle, index=False, float_format="%.12g")
        inventory = [{"file": f"tables/{name}.csv", "rows": len(frame), "columns": list(frame.columns),
                      "sha256": sha(tables / f"{name}.csv")} for name, frame in frames.items()]
        manifest = {"mode": "smoke" if smoke else "full", "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                    "config": config, "extension": extension, "tables": inventory, "plots": plots,
                    "v1_baseline_reconciliation": baseline_check, "protected_reference_hashes": protected,
                    "protected_existing_files_unchanged": True,
                    "methodology": "A–E retrospective accuracy diagnostics; revised TEST context; unchanged TRAIN and detector; no trading evaluations"}
        with (output / "supplementary_diagnostics_manifest.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
        return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", help="Fresh output directory if requested names already exist")
    args = parser.parse_args()
    config, extension = load_config(args.config, smoke=args.smoke)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    run(config, extension, smoke=args.smoke)


if __name__ == "__main__":
    main()
