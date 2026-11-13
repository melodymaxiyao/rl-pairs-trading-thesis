"""llt_turning_point_v1 runner.  usage: run_llt.py [--smoke] [--workers N|auto]

Unit of work = (source, repeat, ticker); folds run in calendar order inside a unit (warm-start chain).
Each finished fold is appended (fsynced) to the unit's checkpoint and journaled, so Ctrl-C resumes at the
next fold. When every unit is done, the selection/stitching, tables and plots are produced.
Smoke: SPY, h_ref = 10, real + B1 repeat 0; outputs under smoke/.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, multiprocessing as mp, os, resource, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import llt_lib as L

OUT = Path(__file__).resolve().parent
PREREG = {"preregistration.md": "74d2dcaceb429eabf71ddd280254ac72fe2b1291599a747207e7610db18b6f11",
          "llt_config.json": "3872b3987158e081f667701b92186bf47a4f6917891cedb0aa9d503f842d6a9a"}
NULL_ORDER = ("B1", "B0", "B2")
_LOCK = None
_PRICES = None
_CAL = None
_P2 = None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


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
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "df": df,
            "disk_avail_GiB": round(st.f_bavail * st.f_frsize / 2**30, 2),
            "free_GiB": round(pages["Pages free"] * ps / 2**30, 2),
            "free_inactive_speculative_purgeable_GiB": round(avail, 2)}


def journal(base, rec):
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    line = json.dumps(rec, default=str) + "\n"
    if _LOCK is not None:
        _LOCK.acquire()
    try:
        with open(base / "journal.jsonl", "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    finally:
        if _LOCK is not None:
            _LOCK.release()


def wait_for_disk(base, where):
    while shutil.disk_usage("/Users/maxiyao").free / 2**30 < L.CFG["disk_pause_GiB"]:
        journal(base, {"event": "paused_low_disk", "where": where, "env": env_snapshot()})
        time.sleep(60)


# ---------------------------------------------------------------- units
def unit_paths(base, source, rep, ticker):
    d = base / "checkpoints" / source / f"{rep}"
    return d / f"{ticker}.jsonl", d / f"{ticker}.json.gz"


def _init_worker(lock):
    global _LOCK, _PRICES, _CAL, _P2
    _LOCK = lock
    _PRICES = L.load_all_prices()
    _CAL = L.fold_calendar()
    _P2 = L.part2_config()


def run_unit(args):
    base, source, rep, ticker, h_ref_grid = args
    base = Path(base)
    t0 = time.time()
    part, done = unit_paths(base, source, rep, ticker)
    if done.exists():
        return {"unit": [source, rep, ticker], "skipped": True}
    part.parent.mkdir(parents=True, exist_ok=True)
    folds = _CAL[_CAL.ticker == ticker].sort_values("fold_id").to_dict("records")
    recs = []
    if part.exists():
        for ln in part.read_text().splitlines():
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                break                      # a torn last line from an interrupted write is redone
        part.write_text("".join(json.dumps(r) + "\n" for r in recs))
    finished = {r["fold_id"] for r in recs}
    seed0 = L.CFG["nulls"]["seed"]
    observed = _PRICES[ticker]
    if source in ("B1", "B2"):
        path_all, _ = L.empirical_null_path(observed, source, L.null_seed(seed0, source, rep, ticker))
    elif source == "real":
        path_all = observed
    warm = (recs[-1]["fit"]["param"], tuple(recs[-1]["fit"]["ratios"])) if recs else None
    for fold in folds:
        if fold["fold_id"] in finished:
            continue
        wait_for_disk(base, [source, rep, ticker, fold["fold_id"]])
        tf = time.time()
        if source == "B0":
            path, _ = L.generate_random_walk(observed, fold, seed=L.null_seed(seed0, "B0", rep, ticker, fold["fold_id"]),
                                             ddof=_P2["std_ddof"])
        else:
            path = path_all
        rec = L.evaluate_ticker_fold(path, fold, _P2, h_ref_grid, warm=warm)
        rec["fit"]["ratios"] = list(rec["fit"]["ratios"])
        rec["bandwidth"] = L.bandwidth_row(rec["fit"])
        rec.update(source=source, repeat=rep, seconds=round(time.time() - tf, 3))
        line = json.dumps(rec, default=float) + "\n"
        with open(part, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        journal(base, {"event": "fold_done", "source": source, "repeat": rep, "ticker": ticker,
                       "fold_id": fold["fold_id"], "record_sha256": sha256_bytes(line.encode()),
                       "seconds": rec["seconds"]})
        recs.append(json.loads(line))
        warm = (rec["fit"]["param"], tuple(rec["fit"]["ratios"]))
    blob = gzip.compress(json.dumps(recs).encode(), mtime=0)
    tmp = done.with_suffix(".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, done)
    part.unlink()
    journal(base, {"event": "unit_done", "source": source, "repeat": rep, "ticker": ticker,
                   "n_folds": len(recs), "sha256": sha256_bytes(blob), "bytes": len(blob),
                   "seconds": round(time.time() - t0, 2)})
    return {"unit": [source, rep, ticker], "seconds": time.time() - t0,
            "maxrss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def load_unit(base, source, rep, ticker):
    return json.loads(gzip.decompress(unit_paths(base, source, rep, ticker)[1].read_bytes()))


# ---------------------------------------------------------------- aggregation
def params_rows(units, source, rep):
    rows = []
    for t, recs in units.items():
        for r in recs:
            f, b = r["fit"], r["bandwidth"]
            rows.append({"source": source, "repeat": rep, "ticker": t, "fold_id": r["fold_id"],
                         "q_eta": f["q_eta"], "q_zeta": f["q_zeta"], "sigma_eps2": f["sigma_eps2"],
                         "sigma_eta2": f["sigma_eta2"], "sigma_zeta2": f["sigma_zeta2"],
                         "zeta_over_eta": f["zeta_over_eta"], "parameterisation": f["param"],
                         "ratio_1": f["ratios"][0], "ratio_2": f["ratios"][1],
                         "on_boundary": f["on_boundary"], "on_lower_bound": f["on_lower"], "on_upper_bound": f["on_upper"],
                         "loglik": f["loglik"], "loglik_E": f["loglik_E"], "loglik_H": f["loglik_H"],
                         "best_start": f["best_start"], "optimizer_converged": f["converged"], "n_train": f["n_train"],
                         "mean_lag": b.get("mean_lag"), "d90": b.get("d90"), "equiv_h": b.get("equiv_h"),
                         "equiv_h_flag": b.get("equiv_h_flag"), "fit_seconds": r["seconds"]})
    return rows


def ref_spacing(units, h_ref_grid):
    out = {}
    for h_ref in h_ref_grid:
        tot = {}
        for t, recs in units.items():
            s = sum(r["ref_spacing"][str(h_ref)][0] for r in recs)
            n = sum(r["ref_spacing"][str(h_ref)][1] for r in recs)
            tot[t] = (s, n)
        out[h_ref] = tot
    return out


def accuracy_table(summary, spacing, h_ref_grid, tickers):
    p2 = pd.read_csv(L.TABLES / L.CFG["table51_file"])
    m = summary.merge(p2, on=["h_ref", "ticker", "type"], suffixes=("", "_part2"), how="left", validate="one_to_one")
    m = m[m.h_ref.isin(h_ref_grid)]
    mean_gap = []
    for row in m.itertuples():
        tot = spacing[row.h_ref]
        keys = list(tot) if row.ticker == "equal_ticker_macro" else [row.ticker]
        s, n = sum(tot[k][0] for k in keys), sum(tot[k][1] for k in keys)
        mean_gap.append(s / n if n else np.nan)
    m["mean_ref_spacing_days"] = mean_gap
    m["delay_share"] = m.delay_median / m.mean_ref_spacing_days
    m["delay_share_part2"] = m.delay_median_part2 / m.mean_ref_spacing_days
    m["delay_iqr"] = m.delay_q75 - m.delay_q25
    m["delay_iqr_part2"] = m.delay_q75_part2 - m.delay_q25_part2
    m["f1_minus_part2"] = m.f1 - m.f1_part2
    order = {t: i for i, t in enumerate(["equal_ticker_macro", *tickers])}
    m["_o"] = m.ticker.map(order)
    m["_k"] = m.type.map({"combined": 0, "peak": 1, "trough": 2})
    cols = ["h_ref", "ticker", "type", "precision", "recall", "f1", "delay_median", "delay_q25", "delay_q75", "delay_iqr",
            "delay_share", "n_ref", "n_det", "n_match", "n_test_years", "precision_part2", "recall_part2", "f1_part2",
            "delay_median_part2", "delay_q25_part2", "delay_q75_part2", "delay_iqr_part2", "delay_share_part2",
            "f1_minus_part2", "mean_ref_spacing_days"]
    return m.sort_values(["h_ref", "_o", "_k"])[cols].reset_index(drop=True)


def null_table(real_summary, null_summaries, h_ref_grid, tickers):
    rows = []
    for h_ref in h_ref_grid:
        for ticker in ["equal_ticker_macro", *tickers]:
            sel = (real_summary.h_ref == h_ref) & (real_summary.ticker == ticker) & (real_summary.type == "combined")
            if not sel.any():
                continue
            real = float(real_summary[sel].f1.iloc[0])
            for source in NULL_ORDER:
                if source not in null_summaries:
                    continue
                s = null_summaries[source]
                v = s[(s.h_ref == h_ref) & (s.ticker == ticker) & (s.type == "combined")].f1.to_numpy(float)
                fin = v[np.isfinite(v)]
                q = np.quantile(fin, [0.05, 0.5, 0.95]) if len(fin) else [np.nan] * 3
                row = {"h_ref": h_ref, "scope": "macro" if ticker == "equal_ticker_macro" else "ticker",
                       "ticker": ticker, "benchmark": source, "R": len(v), "n_valid": len(fin), "real_f1": real,
                       "null_f1_q05": q[0], "null_f1_median": q[1], "null_f1_q95": q[2],
                       "real_midrank_position": L.midrank_percentile(v, real),
                       "real_minus_null_median": real - q[1]}
                if source == "B1":
                    row["p_one_sided_real_gt_B1"] = (1 + int(np.sum(fin >= real))) / (1 + len(v))
                rows.append(row)
    t = pd.DataFrame(rows)
    t["holm_adjusted_p"] = np.nan
    t["holm_reject"] = pd.Series([pd.NA] * len(t), dtype="boolean")
    mask = (t.benchmark == "B1") & (t.scope == "macro")
    if mask.any():
        adj, rej = L.holm(t.loc[mask, "p_one_sided_real_gt_B1"].to_numpy(), L.CFG["holm_alpha"])
        t.loc[mask, "holm_adjusted_p"] = adj
        t.loc[mask, "holm_reject"] = rej
    t["note"] = np.where(mask, "preregistered one-sided test (Holm, family = 3 h_ref)",
                         np.where(t.scope == "ticker", "descriptive", "descriptive benchmark"))
    return t


def calm_table(per_fold, h_ref_grid):
    vol = pd.read_csv(L.TABLES / L.CFG["vol_file"])
    vol = vol[(vol.source == "real") & (vol.type == "combined")]
    rows = []
    for h_ref in h_ref_grid:
        v = vol[vol.h_ref == h_ref][["ticker", "fold_id", "vol_ratio", "n_det", "n_ref", "n_match"]]
        llt = per_fold[per_fold.h_ref == h_ref].merge(v[["ticker", "fold_id", "vol_ratio"]], on=["ticker", "fold_id"],
                                                      how="left", validate="one_to_one")
        for det, frame in (("LLT", llt), ("Part II", v)):
            frame = frame[np.isfinite(frame.vol_ratio)]
            for group, g in (("calm (<0.8)", frame[frame.vol_ratio < L.CFG["calm_threshold"]]),
                             ("normal (>=0.8)", frame[frame.vol_ratio >= L.CFG["calm_threshold"]])):
                nd, nr, nm = int(g.n_det.sum()), int(g.n_ref.sum()), int(g.n_match.sum())
                prec = nm / nd if nd else np.nan
                rec = nm / nr if nr else np.nan
                f1 = 2 * prec * rec / (prec + rec) if nd and nr and (prec + rec) > 0 else np.nan
                rows.append({"h_ref": h_ref, "detector": det, "group": group, "n_asset_years": len(g),
                             "n_det": nd, "n_ref": nr, "n_match": nm, "precision": prec, "recall": rec, "f1": f1})
    t = pd.DataFrame(rows)
    drops = []
    for (h_ref, det), g in t.groupby(["h_ref", "detector"], sort=False):
        g = g.set_index("group")
        drops.append({"h_ref": h_ref, "detector": det, "group": "recall drop = normal − calm",
                      "recall": g.loc["normal (>=0.8)", "recall"] - g.loc["calm (<0.8)", "recall"],
                      "f1": g.loc["normal (>=0.8)", "f1"] - g.loc["calm (<0.8)", "f1"]})
    return pd.concat([t, pd.DataFrame(drops)], ignore_index=True)


def predictions(acc, nulls, calm, params_real, h_ref_grid):
    rows = []
    macro = acc[(acc.ticker == "equal_ticker_macro") & (acc.type == "combined")].set_index("h_ref")
    d = {h: float(macro.loc[h, "f1"] - macro.loc[h, "f1_part2"]) for h in h_ref_grid}
    ok = [abs(x) <= L.CFG["p1_tolerance"] for x in d.values()]
    rows.append({"prediction": "P1", "verdict": "holds" if all(ok) else "does not hold",
                 "support": "; ".join(f"h_ref={h}: LLT {macro.loc[h, 'f1']:.4f} vs Part II {macro.loc[h, 'f1_part2']:.4f} "
                                      f"(diff {d[h]:+.4f}, |diff|<=0.05: {o})" for (h, o) in zip(h_ref_grid, ok)),
                 "n_satisfied": int(sum(ok)), "n_total": len(ok)})
    b1 = nulls[(nulls.benchmark == "B1") & (nulls.scope == "macro")].set_index("h_ref")
    ok2 = [(b1.loc[h, "real_f1"] <= b1.loc[h, "null_f1_q95"]) and not bool(b1.loc[h, "holm_reject"]) for h in h_ref_grid]
    rows.append({"prediction": "P2", "verdict": "holds" if all(ok2) else "does not hold",
                 "support": "; ".join(f"h_ref={h}: real {b1.loc[h, 'real_f1']:.4f}, B1 q95 {b1.loc[h, 'null_f1_q95']:.4f}, "
                                      f"p {b1.loc[h, 'p_one_sided_real_gt_B1']:.4f}, Holm p {b1.loc[h, 'holm_adjusted_p']:.4f}, "
                                      f"reject {bool(b1.loc[h, 'holm_reject'])}" for h in h_ref_grid),
                 "n_satisfied": int(sum(ok2)), "n_total": len(ok2)})
    eh = params_real.equiv_h.to_numpy(float)
    fin = eh[np.isfinite(eh)]
    q = np.quantile(fin, [0.25, 0.5, 0.75]) if len(fin) else [np.nan] * 3
    flags = params_real.equiv_h_flag.value_counts().to_dict()
    rows.append({"prediction": "P3", "verdict": "no verdict (descriptive)",
                 "support": f"equivalent h over {len(eh)} ticker-folds: quartiles {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; flags {flags}; "
                            + "; ".join(f"F1 diff vs Part II at h_ref={h}: {d[h]:+.4f}" for h in h_ref_grid),
                 "n_satisfied": np.nan, "n_total": np.nan})
    drop = calm[calm.group.str.startswith("recall drop")].set_index(["h_ref", "detector"])
    ok4 = [drop.loc[(h, "LLT"), "recall"] < drop.loc[(h, "Part II"), "recall"] for h in h_ref_grid]
    rows.append({"prediction": "P4", "verdict": "holds" if all(ok4) else "does not hold",
                 "support": "; ".join(f"h_ref={h}: recall drop LLT {drop.loc[(h, 'LLT'), 'recall']:+.4f} vs Part II "
                                      f"{drop.loc[(h, 'Part II'), 'recall']:+.4f}" for h in h_ref_grid),
                 "n_satisfied": int(sum(ok4)), "n_total": len(ok4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, TEAL, GREY = "#2a78d6", "#d9822b", "#1f8a70", "#8a8985"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7)


def plot_spy(base, units_real, selected, prices, h_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p2 = L.part2_config()
    cal = L.fold_calendar()
    years = L.CFG["plot_years"]
    fig, axes = plt.subplots(2, len(years), figsize=(6.4 * len(years), 6.2), squeeze=False,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    p = prices[L.CFG["plot_ticker"]]
    recs = {r["fold_id"]: r for r in units_real[L.CFG["plot_ticker"]]}
    for col, year in enumerate(years):
        fid = f"test_{year}"
        fold = cal[(cal.ticker == L.CFG["plot_ticker"]) & (cal.fold_id == fid)].iloc[0].to_dict()
        fit = {**recs[fid]["fit"], "ratios": tuple(recs[fid]["fit"]["ratios"])}
        s = selected[(selected.fold_id == fid) & (selected.h_ref == h_ref)].iloc[0]
        c, d = float(s.c), int(s.d)
        start = p.index.searchsorted(pd.Timestamp(fold["train_start"]))
        seg = p.loc[fold["train_start"]:fold["test_end"]]
        pi, _ = L.filter_pi(seg.to_numpy(), fit)
        ev = L.detect_pi_events(pi, c, d)
        ev[:, 0] += start
        refs, bounds = L.reference_frame(p, fold, p2, [h_ref])
        b, e = bounds[(h_ref, "test")]
        ev = ev[(ev[:, 0] >= b) & (ev[:, 0] <= e)]
        rf = refs[(h_ref, "test")]
        yr = np.flatnonzero(p.index.year == year)
        top, bot = axes[0, col], axes[1, col]
        top.plot(p.index[yr], p.iloc[yr], color=INK2, lw=1.0)
        for kind, marker, color in ((-1, "v", ORANGE), (1, "^", TEAL)):
            rp = rf[rf[:, 1] == kind, 0]
            dp = ev[ev[:, 1] == kind, 0]
            top.scatter(p.index[rp], p.iloc[rp], marker=marker, s=70, facecolors="white", edgecolors=color, lw=1.4, zorder=3)
            top.scatter(p.index[dp], p.iloc[dp], marker=marker, s=34, color=color, edgecolors="white", lw=0.8, zorder=4)
        top.set_ylabel(f"SPY {year}: log price", color=INK, fontsize=9)
        top.text(0.01, 0.97, f"frozen TRAIN fit; selected c = {c:g}, d = {d}; h_ref = {h_ref}",
                 transform=top.transAxes, va="top", fontsize=8, color=INK2)
        pi_full = np.full(len(p), np.nan)
        pi_full[start:start + len(seg)] = pi
        bot.plot(p.index[yr], pi_full[yr], color=BLUE, lw=1.3)
        bot.axhline(c, color=TEAL, lw=1, ls="--")
        bot.axhline(1 - c, color=ORANGE, lw=1, ls="--")
        bot.axhline(0.5, color=GREY, lw=0.8)
        bot.set_ylim(-0.03, 1.03)
        bot.set_ylabel("π_t = P(S_t > 0 | data ≤ t)", color=INK, fontsize=9)
        for ax in (top, bot):
            _style(ax)
    handles = [plt.Line2D([], [], color=INK2, lw=1.0),
               plt.Line2D([], [], marker="^", ls="", markerfacecolor="white", markeredgecolor=TEAL, markersize=8),
               plt.Line2D([], [], marker="^", ls="", color=TEAL, markersize=6),
               plt.Line2D([], [], marker="v", ls="", markerfacecolor="white", markeredgecolor=ORANGE, markersize=8),
               plt.Line2D([], [], marker="v", ls="", color=ORANGE, markersize=6),
               plt.Line2D([], [], color=BLUE, lw=1.3), plt.Line2D([], [], color=TEAL, lw=1, ls="--"),
               plt.Line2D([], [], color=ORANGE, lw=1, ls="--")]
    labels = ["log adjusted close", "reference trough", "LLT confirmed trough", "reference peak", "LLT confirmed peak",
              "π_t", "c", "1 − c"]
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(base / "plots/llt_spy_2020_2022.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_vs_null(base, nulls, h_ref_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = [s for s in NULL_ORDER if s in nulls]
    fig, axes = plt.subplots(1, len(h_ref_grid), figsize=(4.2 * len(h_ref_grid), 3.8), squeeze=False, sharey=False)
    rng = np.random.default_rng(0)
    for ax, h_ref in zip(axes[0], h_ref_grid):
        real = None
        for i, s in enumerate(srcs):
            v = nulls[s]
            v = v[(v.h_ref == h_ref) & (v.ticker == "equal_ticker_macro") & (v.type == "combined")].f1.to_numpy(float)
            x = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=10, color=GREY, alpha=0.55, lw=0)
            q = np.quantile(v, [0.05, 0.5, 0.95])
            ax.plot([i, i], [q[0], q[2]], color=INK, lw=1.6)
            ax.plot([i - 0.25, i + 0.25], [q[1], q[1]], color=INK, lw=2)
        real = nulls["_real"]
        r = float(real[(real.h_ref == h_ref) & (real.ticker == "equal_ticker_macro") & (real.type == "combined")].f1.iloc[0])
        ax.axhline(r, color=BLUE, lw=1.8)
        ax.set_xticks(range(len(srcs)), srcs)
        ax.set_xlim(-0.6, len(srcs) - 0.4)
        ax.set_title(f"h_ref = {h_ref}", fontsize=9, color=INK)
        ax.set_ylabel("equal-ticker test F1" if h_ref == h_ref_grid[0] else "", color=INK, fontsize=9)
        _style(ax)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.8), plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=4),
               plt.Line2D([], [], color=INK, lw=1.6)]
    fig.legend(handles, ["real LLT F1", "one null repetition", "null median and 5–95% range"],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(base / "plots/llt_vs_null.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_bandwidth(base, params_real, tickers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(tickers) + 1.4))
    for i, t in enumerate(tickers):
        v = params_real[params_real.ticker == t]
        fin = v[np.isfinite(v.equiv_h)]
        ax.scatter(fin.equiv_h, np.full(len(fin), i), s=28, color=BLUE, edgecolors="white", lw=0.8, zorder=3)
        flags = v.equiv_h_flag.value_counts()
        notes = [f"{int(flags[k])} {label}" for k, label in (("above_range", "> h 1000"), ("below_range", "< h 1"),
                                                             ("riccati_not_converged", "not converged")) if k in flags]
        if notes:
            ax.text(1.01, i, "; ".join(notes), transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="center")
    for h in L.CFG["h_ref_grid"]:
        ax.axvline(h, color=ORANGE, lw=1, ls="--")
        ax.text(h * 1.04, len(tickers) - 0.45, f"h_ref {h}", color=ORANGE, fontsize=7, ha="left")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(range(len(tickers)), tickers, color=INK)
    ax.set_ylim(-0.6, len(tickers) - 0.1)
    ax.set_xlabel("equivalent h of the MLE slope filter (days; Part II causal kernel with the same mean |weight| lag)",
                  color=INK, fontsize=8)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.7)
    fig.tight_layout()
    fig.savefig(base / "plots/llt_effective_bandwidth.png", dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main
def workers_auto(peak_mb):
    snap = env_snapshot()
    by_mem = math.floor(0.7 * snap["free_inactive_speculative_purgeable_GiB"] * 1024 / peak_mb)
    return max(1, min(os.cpu_count() - 2, by_mem)), snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", default="auto")
    args = ap.parse_args()
    for f, h in PREREG.items():
        if hashlib.sha256((OUT / f).read_bytes()).hexdigest() != h:
            sys.exit(f"STOP: {f} changed after preregistration")
    if not json.load(open(OUT / "logs/consistency.json")).get("all_pass"):
        sys.exit("STOP: consistency checks have not passed")
    base = OUT / "smoke" if args.smoke else OUT
    for sub in ("tables", "plots", "checkpoints", "logs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    jl = base / "journal.jsonl"
    if jl.exists() and any(json.loads(x).get("event") == "run_complete" for x in jl.read_text().splitlines()):
        print("run already complete; nothing to do")
        return
    if args.smoke:
        tickers, h_ref_grid = L.CFG["smoke"]["tickers"], L.CFG["smoke"]["h_ref_grid"]
        plan = {s: reps for s, reps in L.CFG["smoke"]["sources"].items()}
    else:
        tickers, h_ref_grid = L.CFG["tickers"], L.CFG["h_ref_grid"]
        plan = {"real": [-1], "B1": list(range(L.CFG["nulls"]["B1"])), "B0": list(range(L.CFG["nulls"]["B0"])),
                "B2": list(range(L.CFG["nulls"]["B2"]))}
    peak_file = OUT / "logs/smoke_peak.json"
    if args.workers == "auto":
        peak = json.load(open(peak_file))["per_worker_peak_MB"] if peak_file.exists() else 400.0
        workers, snap = workers_auto(peak)
    else:
        workers, snap = int(args.workers), env_snapshot()
    journal(base, {"event": "start", "workers": workers, "tickers": tickers, "h_ref_grid": h_ref_grid,
                   "plan": {s: len(r) for s, r in plan.items()}, "env": snap})
    print(f"workers={workers}; plan={ {s: len(r) for s, r in plan.items()} }", flush=True)
    units = [(str(base), s, r, t, h_ref_grid) for s in ("real", *NULL_ORDER) if s in plan for r in plan[s] for t in tickers]
    timing = {}
    lock = mp.get_context("spawn").Lock()
    t_all = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(lock,)) as pool:
        for part in ("real", "nulls"):
            todo = [u for u in units if (u[1] == "real") == (part == "real")]
            if not todo:
                continue
            t0 = time.time()
            n_done, peak_rss = 0, 0.0
            for res in pool.imap_unordered(run_unit, todo):
                n_done += 1
                peak_rss = max(peak_rss, res.get("maxrss_MB", 0.0))
                if n_done % max(1, len(todo) // 20) == 0 or n_done == len(todo):
                    print(f"{part}: {n_done}/{len(todo)} units, {time.time() - t0:.0f}s", flush=True)
            timing[part] = {"wall_seconds": round(time.time() - t0, 1), "units": len(todo), "max_worker_rss_MB": round(peak_rss, 1)}
            journal(base, {"event": "part_done", "part": part, **timing[part]})
    global _LOCK
    _LOCK = None
    if args.smoke:
        peak = max(v["max_worker_rss_MB"] for v in timing.values()) if timing else 0.0
        fold_secs = [json.loads(x)["seconds"] for x in jl.read_text().splitlines() if '"fold_done"' in x]
        json.dump({"per_worker_peak_MB": peak, "n_fold_fits": len(fold_secs),
                   "mean_fold_seconds": float(np.mean(fold_secs)) if fold_secs else None,
                   "timing": timing}, open(peak_file, "w"), indent=1)
    prices = L.load_all_prices(tickers)
    p2 = L.part2_config()
    loaded = {}
    for s in plan:
        for r in plan[s]:
            loaded[(s, r)] = {t: load_unit(base, s, r, t) for t in tickers}
    real_units = loaded[("real", -1)]
    real_summary, real_sel, real_pf = L.aggregate_source(real_units, tickers, h_ref_grid, p2)
    params_real = pd.DataFrame(params_rows(real_units, "real", -1))
    acc = accuracy_table(real_summary, ref_spacing(real_units, h_ref_grid), h_ref_grid, tickers)
    null_sum, null_params = {}, []
    for (s, r), u in loaded.items():
        if s == "real":
            continue
        summ, _, _ = L.aggregate_source(u, tickers, h_ref_grid, p2)
        null_sum.setdefault(s, []).append(summ.assign(source=s, repeat=r))
        null_params.extend(params_rows(u, s, r))
    null_sum = {s: pd.concat(v, ignore_index=True) for s, v in null_sum.items()}
    nulls = null_table(real_summary, null_sum, h_ref_grid, tickers)
    tables = {"llt_params_by_fold": params_real, "llt_selected_by_fold": real_sel, "llt_accuracy": acc,
              "llt_null_comparison": nulls}
    if null_params:
        tables["llt_params_nulls"] = pd.DataFrame(null_params)
        tables["llt_null_samples"] = pd.concat(null_sum.values(), ignore_index=True)
        allp = pd.concat([params_real, tables["llt_params_nulls"]])
        rows = []
        for s, g in allp.groupby("source", sort=False):
            for col, f in (("log10_q_zeta", np.log10(g.q_zeta)), ("log10_zeta_over_eta", np.log10(g.zeta_over_eta)),
                           ("equiv_h", g.equiv_h)):
                v = np.asarray(f, float)
                v = v[np.isfinite(v)]
                q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(v) else [np.nan] * 5
                rows.append({"source": s, "quantity": col, "n_fits": len(g), "n_finite": len(v),
                             **dict(zip(["q05", "q25", "median", "q75", "q95"], q)),
                             "share_on_boundary": float(g.on_boundary.mean()),
                             "share_H_parameterisation": float((g.parameterisation == "H").mean())})
        tables["llt_smoothing_real_vs_null"] = pd.DataFrame(rows)
    if not args.smoke:
        calm = calm_table(real_pf, h_ref_grid)
        tables["llt_calm_years"] = calm
        tables["predictions"] = predictions(acc, nulls, calm, params_real, h_ref_grid)
    for name, frame in tables.items():
        with gzip.open(base / "tables" / f"{name}.csv.gz", "wt", compresslevel=6) as f:
            frame.to_csv(f, index=False, float_format="%.10g")
    plot_spy(base, real_units, real_sel, prices, L.CFG["plot_h_ref"] if L.CFG["plot_h_ref"] in h_ref_grid else h_ref_grid[0]) \
        if not args.smoke else None
    if null_sum:
        plot_vs_null(base, {**null_sum, "_real": real_summary}, h_ref_grid)
    plot_bandwidth(base, params_real, tickers)
    after = L.hash_inputs()
    before = json.load(open(OUT / "input_hashes_before.json"))
    diffs = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if not args.smoke:
        json.dump(after, open(OUT / "input_hashes_after.json", "w"), indent=1, sort_keys=True)
    summary = {"timing": timing, "total_wall_seconds": round(time.time() - t_all, 1), "workers": workers,
               "inputs": len(after), "hash_differences": diffs, "env_end": env_snapshot()}
    json.dump(summary, open(base / "run_summary.json", "w"), indent=1)
    journal(base, {"event": "run_complete", **summary})
    print(json.dumps(summary, indent=1), flush=True)
    if diffs:
        sys.exit(f"STOP: input hashes changed: {diffs}")


if __name__ == "__main__":
    main()
