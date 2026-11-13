"""Binding consistency checks C1-C4 (preregistration §6). Writes logs/consistency.json; exit 2 on failure."""
import contextlib
import io
import json
import sys
import time

import numpy as np
import pandas as pd

import llt_lib as L
from experiments.turning_point_detection.new_stage1.run_slope_accuracy_diagnostics import load_config
from experiments.turning_point_detection.new_stage1.slope_accuracy_diagnostics import evaluate_selected
from experiments.turning_point_detection.new_stage1.kernel_smoothing import causal_local_linear_slope
from experiments.turning_point_detection.new_stage1.slope_accuracy import detect_hysteresis_events

TOL = L.CFG["consistency"]["tol"]
FIELDS = ["n_ref", "n_det", "n_match", "precision", "recall", "f1", "delay_q25", "delay_median", "delay_q75", "n_test_years"]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def c1(prices, cal, p2cfg):
    c = L.CFG["consistency"]["c1"]
    sel = pd.read_csv(L.TABLES / L.CFG["selected_params_file"])
    row = sel[(sel.h_ref == c["h_ref"]) & (sel.fold_id == c["fold_id"])].iloc[0]
    h, d, k = int(row.h), int(row.d), float(row.k)
    fold = cal[(cal.ticker == c["ticker"]) & (cal.fold_id == c["fold_id"])].iloc[0].to_dict()
    p = prices[c["ticker"]]
    local = p.loc[fold["train_start"]:fold["test_end"]]
    b = causal_local_linear_slope(local, bandwidth_h=h, window_multiplier=p2cfg["window_multiplier"])
    sigma = float(b.loc[local.index <= pd.Timestamp(fold["train_end"])].std(ddof=p2cfg["std_ddof"]))
    ev = detect_hysteresis_events(b.to_numpy(), tau=k * sigma, d=d)
    mine = pd.DataFrame({"date": local.index[ev[:, 0]].strftime("%Y-%m-%d"),
                         "type": np.where(ev[:, 1] == 1, "trough", "peak")})
    mine["segment"] = np.where(pd.to_datetime(mine.date) <= pd.Timestamp(fold["train_end"]), "train", "test")
    st = pd.read_csv(L.TABLES / L.CFG["detector_events_file"])
    st = st[(st.ticker == c["ticker"]) & (st.fold_id == c["fold_id"]) & (st.h == h) & (st.d == d) & (st.k == k)]
    st = st.assign(date=pd.to_datetime(st.date).dt.strftime("%Y-%m-%d")).sort_values("date")[["date", "type", "segment"]]
    same = len(mine) == len(st) and (mine.reset_index(drop=True) == st.reset_index(drop=True)).all().all()
    return {"ticker": c["ticker"], "fold_id": c["fold_id"], "h_ref": c["h_ref"], "h": h, "d": d, "k": k,
            "n_events": len(mine), "n_test_events": int((mine.segment == "test").sum()), "identical": bool(same)}


def c2(prices, cal, cfg):
    r = quiet(evaluate_selected, prices, cal, cfg)
    mine = r["summary"].drop(columns=["source", "repeat"]).set_index(["h_ref", "ticker", "type"]).sort_index()
    ref = pd.read_csv(L.TABLES / L.CFG["table51_file"]).set_index(["h_ref", "ticker", "type"]).sort_index()
    if not mine.index.equals(ref.index):
        return {"pass": False, "reason": "row sets differ"}
    diff = (mine[FIELDS] - ref[FIELDS]).abs()
    both_nan = mine[FIELDS].isna() & ref[FIELDS].isna()
    max_diff = float(diff.where(~both_nan, 0).max().max())
    macro = mine.xs(("equal_ticker_macro", "combined"), level=("ticker", "type"))
    ref_macro = ref.xs(("equal_ticker_macro", "combined"), level=("ticker", "type"))
    return {"rows": len(mine), "max_abs_diff": max_diff, "pass": bool(max_diff <= TOL),
            "macro_combined_f1": {int(h): {"recomputed": float(macro.loc[h, "f1"]), "table": float(ref_macro.loc[h, "f1"])}
                                  for h in macro.index}}


def c3(prices, cal, cfg):
    h_ref = L.CFG["consistency"]["c3_h_ref"]
    cfg10 = {**cfg, "h_ref_grid": [h_ref]}
    stored = pd.read_csv(L.TABLES / L.CFG["null_samples_file"])
    stored = stored[(stored.source == "B1") & (stored.h_ref == h_ref) & (stored.ticker == "equal_ticker_macro")
                    & (stored.type == "combined")].set_index("repeat").f1
    vals, worst = [], 0.0
    for rep in range(L.CFG["consistency"]["c3_repeats"]):
        paths = {t: L.empirical_null_path(prices[t], "B1", L.null_seed(L.CFG["nulls"]["seed"], "B1", rep, t))[0] for t in prices}
        s = quiet(evaluate_selected, paths, cal, cfg10, source="B1", repeat=rep)["summary"]
        f1 = float(s[(s.ticker == "equal_ticker_macro") & (s.type == "combined")].f1.iloc[0])
        vals.append(f1)
        worst = max(worst, abs(f1 - float(stored.loc[rep])))
        print(f"C3 repeat {rep + 1}/100 f1={f1:.6f}", flush=True)
    table = pd.read_csv(L.TABLES / L.CFG["table52_file"])
    t52 = float(table[(table.h_ref == h_ref) & (table.ticker == "equal_ticker_macro") & (table.type == "combined")].b1_f1_median.iloc[0])
    med = float(np.median(vals))
    return {"h_ref": h_ref, "R": len(vals), "median_recomputed": med, "median_table52": t52,
            "max_abs_diff_per_repeat": worst, "pass": bool(abs(med - t52) <= TOL and worst <= TOL)}


def c4(prices, cal):
    gen = pd.read_csv(L.TABLES / L.CFG["null_generation_file"])
    seed0 = L.CFG["nulls"]["seed"]
    bad, n = [], 0
    for source in ("B1", "B2"):
        g = gen[gen.source == source].set_index(["repeat", "ticker"])
        for rep in range(L.CFG["nulls"][source]):
            for t in prices:
                seed = L.null_seed(seed0, source, rep, t)
                a, audit = L.empirical_null_path(prices[t], source, seed)
                b, _ = L.empirical_null_path(prices[t], source, seed)
                row = g.loc[(rep, t)]
                ok = (a.to_numpy().tobytes() == b.to_numpy().tobytes() and int(row.seed) == seed
                      and int(row.n_prices) == audit["n_prices"]
                      and abs(audit["return_mean_full_series"] - row.return_mean_full_series) <= TOL * abs(row.return_mean_full_series)
                      and abs(audit["realized_surrogate_return_mean"] - row.realized_surrogate_return_mean) <= TOL * max(abs(row.realized_surrogate_return_mean), 1e-12))
                n += 1
                if not ok:
                    bad.append([source, rep, t])
    g = gen[gen.source == "B0"].set_index(["repeat", "ticker", "fold_id"])
    for rep in range(L.CFG["nulls"]["B0"]):
        for fold in cal.to_dict("records"):
            t = fold["ticker"]
            seed = L.null_seed(seed0, "B0", rep, t, fold["fold_id"])
            a, audit = L.generate_random_walk(prices[t], fold, seed=seed, ddof=1)
            b, _ = L.generate_random_walk(prices[t], fold, seed=seed, ddof=1)
            row = g.loc[(rep, t, fold["fold_id"])]
            ok = (a.to_numpy().tobytes() == b.to_numpy().tobytes() and int(row.seed) == seed
                  and int(row.n_prices) == audit["n_prices"]
                  and abs(audit["train_return_mean"] - row.train_return_mean) <= TOL * abs(row.train_return_mean)
                  and abs(audit["train_return_std"] - row.train_return_std) <= TOL * abs(row.train_return_std))
            n += 1
            if not ok:
                bad.append(["B0", rep, t, fold["fold_id"]])
    return {"n_checked": n, "n_bad": len(bad), "bad": bad[:20], "pass": not bad}


def main():
    t0 = time.time()
    cfg, _ = load_config(L.RESTORED / L.CFG["part2_diag_config"])
    prices = L.load_all_prices()
    cal = L.fold_calendar()
    rep = {}
    steps = [("C1", lambda: c1(prices, cal, cfg)), ("C4", lambda: c4(prices, cal)),
             ("C2", lambda: c2(prices, cal, cfg)), ("C3", lambda: c3(prices, cal, cfg))]
    for name, fn in steps:
        t = time.time()
        rep[name] = fn()
        rep[name]["seconds"] = round(time.time() - t, 1)
        ok = rep[name].get("identical", rep[name].get("pass"))
        print(name, json.dumps(rep[name], default=str)[:600], flush=True)
        json.dump(rep, open(L.OUT / "logs/consistency.json", "w"), indent=1, default=str)
        if not ok:
            print(f"STOP: {name} failed", flush=True)
            sys.exit(2)
    rep["all_pass"] = True
    rep["total_seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(L.OUT / "logs/consistency.json", "w"), indent=1, default=str)
    print("ALL CONSISTENCY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
