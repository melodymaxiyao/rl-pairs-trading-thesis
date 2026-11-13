"""gateb_v1 runner.  usage: run_gateb.py [--smoke]

Full mode: resumable; one journal record per (h_ref, cost, init) setting. After all settings, it
assembles tables and plots, hashes inputs again, and deletes checkpoint bodies (receipts are kept).
Smoke mode: SPY, primary setting only, outputs under smoke/.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gateb_lib as G

OUT = Path(__file__).resolve().parent
PREREG = {"preregistration.md": "96f95b0b35373e3c6a452bebf014061435d1fd23d05b046c25c9e456ddf011a0",
          "gateb_config.json": "b0773a80226124ccd3f2a383f25cd3cc5e962dfa3d11828aa5ccc68a9a026348"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
RULE_C, BH_C, HOLD_C = "#2a78d6", "#52514e", "#cde2fb"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_snapshot():
    df = subprocess.run(["df", "-h", "/Users/maxiyao"], capture_output=True, text=True).stdout.strip()
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {}
    for ln in vm.splitlines()[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[k.strip()] = int(v)
    ps = 16384
    avail = (pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"] + pages["Pages purgeable"]) * ps / 2**30
    st = os.statvfs("/Users/maxiyao")
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "df": df, "disk_avail_GiB": round(st.f_bavail * st.f_frsize / 2**30, 2),
            "free_GiB": round(pages["Pages free"] * ps / 2**30, 2), "free_inactive_speculative_purgeable_GiB": round(avail, 2)}


def input_files(cfg):
    files = {str(G.RESTORED / cfg[k]) for k in ("selected_params_file", "selected_params_v1_file", "detector_events_file",
                                                "fold_calendar_file", "detector_config_file")}
    files |= {str(G.RESTORED / "data/raw" / f"{t.lower()}_daily_adjusted_ohlcv.csv") for t in cfg["tickers"]}
    files |= {m.__file__ for m in list(sys.modules.values())
              if getattr(m, "__file__", None) and m.__file__.startswith(str(G.RESTORED))}
    files.add(str(G.RESTORED / ".venv/lib/python3.13/site-packages/py.py"))  # pytest shim used by the tests
    return sorted(files)


def hash_files(files):
    out = {}
    for p in files:
        st = os.lstat(p)
        if st.st_flags & 0x40000000:
            sys.exit(f"STOP: dataless input {p}")
        out[p] = {"sha256": sha256(p), "size": st.st_size}
    return out


class Journal:
    def __init__(self, path):
        self.path = path

    def write(self, rec):
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def done(self):
        if not self.path.exists():
            return {}
        return {r["setting"]: r for r in map(json.loads, self.path.read_text().splitlines())
                if r.get("event") == "setting_done"}


# ---------------------------------------------------------------- one setting
def run_setting(setting, cfg, det_cfg, tabs, prices, tickers):
    cal = tabs["calendar"]
    per, results = {}, {}
    for t in tickers:
        logp, ohlcv = prices[t]
        years = []
        for fold in cal[cal.ticker == t].sort_values("test_start").to_dict("records"):
            h, d, k = G.selected_params(tabs["selected"], fold["fold_id"], setting["h_ref"])
            years.append({"fold": fold, "events": G.fold_events(logp, fold, h, d, k, det_cfg), "params": (h, d, k)})
        target, info = G.build_targets(ohlcv, years, setting["init"])
        rule, bh = G.backtest(ohlcv, target, setting["cost"])
        results[t] = {"rule": rule, "bh": bh, "info": info, "years": years, "target": target}
        per[t] = (rule.daily_returns, bh.daily_returns)
    bc = cfg["bootstrap"]
    boot = G.joint_block_bootstrap(per, block_length=bc["block_length"], n_repetitions=bc["n_repetitions"], seed=bc["seed"])
    rows = []
    for t in tickers:
        r = results[t]
        dr, db = G.describe(r["rule"], r["info"]["n_test_days"]), G.describe(r["bh"], r["info"]["n_test_days"])
        dsh = dr["sharpe"] - db["sharpe"]
        lo, hi = G.percentile_ci(boot["per_ticker"][t], bc["ci"])
        row = {"ticker": t, **{k: setting[k] for k in ("h_ref", "cost", "init")}, "setting": setting["name"]}
        row.update({f"rule_{k}": v for k, v in dr.items() if k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")})
        row.update({f"bh_{k}": v for k, v in db.items() if k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")})
        row.update({"delta_sharpe": dsh, "delta_sharpe_ci_lo": lo, "delta_sharpe_ci_hi": hi,
                    "delta_sharpe_p": G.two_sided_p(boot["per_ticker"][t]),
                    "round_trips_per_year": dr["round_trips_per_year"], "holding_day_share": dr["holding_day_share"],
                    "n_confirmed_events": r["info"]["n_events"], "avg_holding_days": dr["avg_holding_days"],
                    "n_test_days": r["info"]["n_test_days"],
                    "first_test_day": str(r["rule"].daily.index[0].date()), "last_test_day": str(r["rule"].daily.index[-1].date())})
        rows.append(row)
    df = pd.DataFrame(rows)
    ew = {"ticker": "EW" if len(tickers) > 1 else f"EW ({tickers[0]} only)", **{k: setting[k] for k in ("h_ref", "cost", "init")},
          "setting": setting["name"]}
    for c in df.columns:
        if c not in ew and pd.api.types.is_numeric_dtype(df[c]) and c not in ("delta_sharpe_ci_lo", "delta_sharpe_ci_hi", "delta_sharpe_p"):
            ew[c] = float(df[c].mean())
    lo, hi = G.percentile_ci(boot["ew"], bc["ci"])
    ew.update({"delta_sharpe_ci_lo": lo, "delta_sharpe_ci_hi": hi, "delta_sharpe_p": G.two_sided_p(boot["ew"]),
               "n_union_days": boot["n_union"]})
    summary = pd.concat([df, pd.DataFrame([ew])], ignore_index=True)
    return summary, results, boot


def primary_extras(results, tickers, cfg, ohlcvs):
    by_year, fwd = [], []
    for t in tickers:
        r = results[t]
        y = G.by_year(r["rule"], r["bh"]).assign(ticker=t)
        by_year.append(y)
        test_ev = []
        for yr in r["years"]:
            f = yr["fold"]
            e = yr["events"]
            test_ev.append(e[(e.date >= pd.Timestamp(f["test_start"])) & (e.date <= pd.Timestamp(f["test_end"]))])
        fr = G.forward_returns(ohlcvs[t], pd.concat(test_ev, ignore_index=True), cfg["forward_diag"]["horizons"]).assign(ticker=t)
        fwd.append(fr)
    by_year = pd.concat(by_year, ignore_index=True)
    wide = by_year.pivot_table(index=["ticker", "year"], columns="strategy", values=["annual_return", "max_drawdown_in_year"])
    wide.columns = [f"{s}_{m}" for m, s in wide.columns]
    wide = wide.reset_index()
    wide["return_diff_rule_minus_bh"] = wide["rule_annual_return"] - wide["bh_annual_return"]
    wide["highlight_year"] = wide.year.isin(cfg["highlight_years"])
    fwd = pd.concat(fwd, ignore_index=True)
    fd = cfg["forward_diag"]
    rows = []
    for H in fd["horizons"]:
        for scope in tickers + (["pooled"] if len(tickers) > 1 else []):
            sub = fwd[(fwd.H == H) & ((fwd.ticker == scope) if scope != "pooled" else True)]
            years = sorted(set(sub.year))
            draws = G.year_block_bootstrap_d(sub, years, n_repetitions=fd["n_repetitions"], seed=fd["seed"])
            ok = draws[np.isfinite(draws)]
            lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
            rows.append({"scope": scope, "H": H, "D_H": G.d_stat(sub),
                         "mean_after_trough": float(sub.loc[sub.type == "trough", "fwd_log_return"].mean()),
                         "mean_after_peak": float(sub.loc[sub.type == "peak", "fwd_log_return"].mean()),
                         "n_trough": int(sub.loc[sub.type == "trough", "fwd_log_return"].notna().sum()),
                         "n_peak": int(sub.loc[sub.type == "peak", "fwd_log_return"].notna().sum()),
                         "n_dropped_incomplete_window": int(sub.fwd_log_return.isna().sum()),
                         "ci_lo": float(lo), "ci_hi": float(hi), "n_valid_boot": int(len(ok)),
                         "n_years": len(years), "ci_method": "bootstrap over whole test years, 5000 reps",
                         "note": "forward windows of nearby events overlap; descriptive only"})
    return wide, pd.DataFrame(rows)


def primary_table(summary, boot, cfg, tickers):
    s = summary.set_index("ticker")
    rows = []
    ew_key = [i for i in s.index if str(i).startswith("EW")][0]
    e = s.loc[ew_key]
    rows.append({"test": "primary", "ticker": ew_key, "statistic": "EW mean ΔSharpe", "delta_sharpe": e.delta_sharpe,
                 "ci_lo": e.delta_sharpe_ci_lo, "ci_hi": e.delta_sharpe_ci_hi, "p": e.delta_sharpe_p,
                 "criterion": "CI lower bound > 0", "passes": bool(e.delta_sharpe_ci_lo > 0)})
    h = G.holm({t: float(s.loc[t, "delta_sharpe_p"]) for t in tickers}, cfg["holm"]["alpha"]).set_index("ticker")
    for t in tickers:
        rows.append({"test": "secondary", "ticker": t, "statistic": "ΔSharpe", "delta_sharpe": s.loc[t, "delta_sharpe"],
                     "ci_lo": s.loc[t, "delta_sharpe_ci_lo"], "ci_hi": s.loc[t, "delta_sharpe_ci_hi"], "p": s.loc[t, "delta_sharpe_p"],
                     "holm_rank": int(h.loc[t, "holm_rank"]), "holm_threshold": h.loc[t, "holm_threshold"],
                     "holm_adjusted_p": h.loc[t, "holm_adjusted_p"], "holm_reject": bool(h.loc[t, "holm_reject"]),
                     "criterion": "Holm rejects (family=7, alpha=0.05) and ΔSharpe > 0",
                     "passes": bool(h.loc[t, "holm_reject"] and s.loc[t, "delta_sharpe"] > 0),
                     "rule_max_drawdown": s.loc[t, "rule_max_drawdown"], "bh_max_drawdown": s.loc[t, "bh_max_drawdown"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_equity(results, tickers, path):
    n = len(tickers)
    cols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(6.2 * cols, 2.6 * nrows), squeeze=False)
    for ax, t in zip(axes.ravel(), tickers):
        r = results[t]
        rule, bh = r["rule"], r["bh"]
        w = rule.daily["w_new"]
        on = w.to_numpy() == 1.0
        idx = w.index
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        ends = np.flatnonzero(on & ~np.r_[on[1:], False])
        for s0, e0 in zip(starts, ends):
            ax.axvspan(idx[s0], idx[e0], color=HOLD_C, lw=0, zorder=0)
        ax.plot(bh.equity_curve.index, bh.equity_curve.values, color=BH_C, lw=1.2, label="buy-and-hold")
        ax.plot(rule.equity_curve.index, rule.equity_curve.values, color=RULE_C, lw=1.6, label="rule")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))
        ax.set_ylabel(t, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=RULE_C, lw=1.6), plt.Line2D([], [], color=BH_C, lw=1.2),
               plt.Rectangle((0, 0), 1, 1, color=HOLD_C)]
    fig.legend(handles, ["rule", "buy-and-hold", "rule holds the asset"], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if n > 1 else 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_forest(summary, path):
    s = summary.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.axvline(0, color=INK2, lw=1)
    for i, r in s.iterrows():
        ew = str(r.ticker).startswith("EW")
        ax.plot([r.delta_sharpe_ci_lo, r.delta_sharpe_ci_hi], [i, i], color=RULE_C, lw=2.4 if ew else 1.8, solid_capstyle="round")
        ax.plot(r.delta_sharpe, i, marker="D" if ew else "o", ms=9 if ew else 8, color=RULE_C, mec="white", mew=2)
    ax.set_yticks(y, s.ticker, color=INK)
    ax.set_xlabel("ΔSharpe (rule − buy-and-hold), 95% block-bootstrap CI", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_drawdown(results, tickers, path):
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.5 * n + 0.6), sharex=True, squeeze=False)
    for ax, t in zip(axes.ravel(), tickers):
        for res, c, lw in ((results[t]["bh"], BH_C, 1.0), (results[t]["rule"], RULE_C, 1.4)):
            eq = res.equity_curve
            dd = eq / eq.cummax() - 1.0
            ax.plot(dd.index, dd.values, color=c, lw=lw)
        ax.set_ylabel(t, color=INK, fontsize=9)
        ax.set_ylim(-1, 0.02)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.legend([plt.Line2D([], [], color=RULE_C, lw=1.4), plt.Line2D([], [], color=BH_C, lw=1.0)],
               ["rule", "buy-and-hold"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for f, h in PREREG.items():
        if sha256(OUT / f) != h:
            sys.exit(f"STOP: {f} changed after preregistration")
    rep = json.load(open(OUT / "logs/event_consistency.json"))
    if not (rep["binding"]["identical"] and rep["all_summary"]["n_mismatch"] == 0):
        sys.exit("STOP: event consistency check has not passed")
    cfg = G.load_config(OUT)
    det_cfg = G.load_detector_config(cfg)
    tabs = G.load_tables(cfg)
    base = OUT / "smoke" if args.smoke else OUT
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    ck = base / "checkpoints"
    ck.mkdir(exist_ok=True)
    tickers = ["SPY"] if args.smoke else cfg["tickers"]
    settings = [s for s in cfg["settings"] if s["name"] == "primary"] if args.smoke else cfg["settings"]
    journal = Journal(base / "journal.jsonl")
    if journal.path.exists() and any(json.loads(l).get("event") == "run_complete"
                                     for l in journal.path.read_text().splitlines()):
        sys.exit("Run already complete and sealed (checkpoint bodies deleted); nothing to do.")
    env0 = env_snapshot()
    if env0["disk_avail_GiB"] < 2.0:
        journal.write({"event": "paused_low_disk", "env": env0})
        sys.exit("PAUSE: disk available below 2 GiB")

    prices = {t: G.load_ticker_prices(t, det_cfg) for t in tickers}
    files = input_files(cfg)
    before = base / "input_hashes_before.json"
    if not before.exists():
        json.dump(hash_files(files), open(before, "w"), indent=1)
        journal.write({"event": "start", "n_inputs": len(files), "env": env0})

    done = journal.done()
    t_start = time.time()
    for s in settings:
        cp = ck / f"{s['name']}.json.gz"
        if s["name"] in done and cp.exists() and sha256(cp) == done[s["name"]]["checkpoint_sha256"]:
            print(f"{s['name']}: already done (journal)", flush=True)
            continue
        env = env_snapshot()
        if env["disk_avail_GiB"] < 2.0:
            journal.write({"event": "paused_low_disk", "setting": s["name"], "env": env})
            sys.exit("PAUSE: disk available below 2 GiB (resumable)")
        t0 = time.time()
        summary, results, boot = run_setting(s, cfg, det_cfg, tabs, prices, tickers)
        payload = {"summary": summary.to_dict("records")}
        if s["name"] == "primary":
            by_year, fwd = primary_extras(results, tickers, cfg, {t: prices[t][1] for t in tickers})
            payload["primary"] = primary_table(summary, boot, cfg, tickers).to_dict("records")
            payload["by_year"] = by_year.to_dict("records")
            payload["forward"] = fwd.to_dict("records")
            plot_equity(results, tickers, base / "plots/gateb_equity.png")
            plot_drawdown(results, tickers, base / "plots/gateb_drawdown.png")
            plot_forest(summary, base / "plots/gateb_delta_sharpe_forest.png")
        with gzip.open(cp, "wt") as f:
            json.dump(payload, f, default=str)
        journal.write({"event": "setting_done", "setting": s["name"], "h_ref": s["h_ref"], "cost": s["cost"],
                       "init": s["init"], "checkpoint_sha256": sha256(cp), "checkpoint_bytes": cp.stat().st_size,
                       "seconds": round(time.time() - t0, 1), "env": env})
        print(f"{s['name']}: done in {time.time() - t0:.1f}s", flush=True)

    # ---- assemble
    payloads = {s["name"]: json.load(gzip.open(ck / f"{s['name']}.json.gz", "rt")) for s in settings}
    summary = pd.concat([pd.DataFrame(p["summary"]) for p in payloads.values()], ignore_index=True)
    summary.to_csv(base / "tables/gateb_summary.csv.gz", index=False)
    prim = payloads["primary"]
    pd.DataFrame(prim["primary"]).to_csv(base / "tables/gateb_primary.csv.gz", index=False)
    pd.DataFrame(prim["by_year"]).to_csv(base / "tables/gateb_by_year.csv.gz", index=False)
    pd.DataFrame(prim["forward"]).to_csv(base / "tables/forward_return_diag.csv.gz", index=False)

    after = hash_files(files)
    json.dump(after, open(base / "input_hashes_after.json", "w"), indent=1)
    b = json.load(open(before))
    diffs = sorted(p for p in set(b) | set(after) if b.get(p) != after.get(p))
    receipts = {}
    if not diffs:
        for s in settings:
            cp = ck / f"{s['name']}.json.gz"
            receipts[s["name"]] = {"sha256": sha256(cp), "bytes": cp.stat().st_size}
            cp.unlink()  # sealed: keep only the receipt
        json.dump(receipts, open(ck / "RECEIPTS.json", "w"), indent=1)
    env1 = env_snapshot()
    journal.write({"event": "run_complete", "hash_differences": diffs, "env": env1,
                   "seconds_this_invocation": round(time.time() - t_start, 1)})
    json.dump({"inputs": len(files), "hash_differences": diffs, "env_start": env0, "env_end": env1,
               "checkpoint_receipts": receipts}, open(base / "run_summary.json", "w"), indent=1)
    print(json.dumps({"inputs": len(files), "hash_differences": len(diffs), "env_end": env1}, indent=1))
    if diffs:
        sys.exit("STOP: input hash differences")


if __name__ == "__main__":
    main()
