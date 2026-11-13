"""policy_stability_v1 main run (definitions: preregistration.md, SHA-256 in preregistration.sha256)."""
from __future__ import annotations

import gzip, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ps_lib as L

OUT = Path(__file__).resolve().parent
FYP_RESULTS = Path("/Users/maxiyao/Desktop/FYP/experiments/single_stock/results")
CFG_DIR = Path("/Users/maxiyao/single_stock_runs/inputs_restored_v1/experiments/single_stock/configs")
PREREG_SHA = "6364fb888c6ab8be84bdd14b55e0255fd285ccc74b40956ae79bdf42dffcd4c7"
SF_DATALESS = 0x40000000
TICKERS = ["SPY", "XLU", "KO", "IWM", "NVDA", "TSLA", "GOOGL"]
MODELS = ["bandit_3state", "tabular_q_6state", "tabular_q_12state"]
HORIZONS = [5, 10]
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]
SEEDS = list(range(10))
CKPT = OUT / "checkpoints"
JOURNAL = OUT / "journal.jsonl"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exp2_path(t, m, f, h, s):
    return FYP_RESULTS / "walkforward" / t.lower() / m / f / f"horizon_{h}" / "cost_0.001" / f"seed_{s}" / "q_table.json"


def exp1_path(t, m, h, s):
    return FYP_RESULTS / t.lower() / m / f"horizon_{h}" / "cost_0.001" / f"seed_{s}" / "q_table.json"


def all_inputs():
    files = [CFG_DIR / "single_stock_walkforward.json", CFG_DIR / "single_stock.json"]
    for t in TICKERS:
        files += [exp2_path(t, m, f, h, s) for m in MODELS for f in FOLDS for h in HORIZONS for s in SEEDS]
        files += [exp1_path(t, m, h, s) for m in MODELS for h in HORIZONS for s in SEEDS]
    return files


def env_snapshot():
    import subprocess
    df = subprocess.run(["df", "-h", "/Users/maxiyao"], capture_output=True, text=True).stdout.strip()
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in
             (ln.split(":") for ln in vm.splitlines()[1:] if ":" in ln) if v.strip().rstrip(".").isdigit()}
    ps = 16384
    free = pages["Pages free"] * ps / 2**30
    avail = (pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"] + pages["Pages purgeable"]) * ps / 2**30
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "df": df, "free_GiB": round(free, 2),
            "free_inactive_speculative_purgeable_GiB": round(avail, 2)}


def hash_inputs(files):
    out = {}
    for p in files:
        st = os.lstat(p)
        if st.st_flags & SF_DATALESS:
            sys.exit(f"STOP: dataless input {p}")
        out[str(p)] = {"sha256": sha256(p), "size": st.st_size}
    return out


def journal(rec):
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_tickers():
    if not JOURNAL.exists():
        return {}
    out = {}
    for ln in JOURNAL.read_text().splitlines():
        r = json.loads(ln)
        if r.get("event") == "ticker_done":
            out[r["ticker"]] = r["checkpoint_sha256"]
    return out


def read_cell(paths):
    """Return labels, seeds_long per state, and greedy mismatches vs stored greedy_policy."""
    labels, counts, mismatches = None, None, 0
    for p in paths:
        st = os.lstat(p)
        if st.st_flags & SF_DATALESS:
            sys.exit(f"STOP: dataless input {p}")
        d = json.load(open(p))
        lab = [d["state_labels"][str(i)] for i in range(len(d["q_table"]))]
        if labels is None:
            labels, counts = lab, [0] * len(lab)
        elif lab != labels:
            sys.exit(f"STOP: state labels differ across seeds at {p}")
        for i, q in enumerate(d["q_table"]):
            if len(q) != 2:
                sys.exit(f"STOP: expected 2 actions at {p}")
            g = L.greedy_action(q)
            counts[i] += g
            if g != int(d["greedy_policy"][str(i)]):
                mismatches += 1
    for lab in labels:
        L.z_bucket(lab)  # stop on any unmappable label
    return labels, counts, mismatches


def process_ticker(t):
    rec = {"ticker": t, "exp2": {}, "exp1": {}, "greedy_mismatch_vs_stored": 0}
    for m in MODELS:
        for h in HORIZONS:
            for f in FOLDS:
                labels, counts, mm = read_cell([exp2_path(t, m, f, h, s) for s in SEEDS])
                rec["exp2"][f"{m}|{h}|{f}"] = {"labels": labels, "seeds_long": counts}
                rec["greedy_mismatch_vs_stored"] += mm
            labels, counts, mm = read_cell([exp1_path(t, m, h, s) for s in SEEDS])
            rec["exp1"][f"{m}|{h}"] = {"labels": labels, "seeds_long": counts}
            rec["greedy_mismatch_vs_stored"] += mm
    path = CKPT / f"{t}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)
    journal({"event": "ticker_done", "ticker": t, "checkpoint_sha256": sha256(path)})
    return rec


def main():
    assert sha256(OUT / "preregistration.md") == PREREG_SHA, "preregistration changed"
    CKPT.mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    env_start = env_snapshot()
    files = all_inputs()
    assert len(files) == 2 + 7 * (240 + 60)
    before_path = OUT / "input_hashes_before.json.gz"
    if not before_path.exists():
        with gzip.open(before_path, "wt") as f:
            json.dump(hash_inputs(files), f)
        journal({"event": "hashes_before", "n": len(files), "env": env_start})

    done = done_tickers()
    recs = {}
    for t in TICKERS:
        cp = CKPT / f"{t}.json.gz"
        if t in done and cp.exists() and sha256(cp) == done[t]:
            recs[t] = json.load(gzip.open(cp, "rt"))
        else:
            recs[t] = process_ticker(t)
        print(f"{t}: done", flush=True)

    # ---- modal actions table
    rows = []
    for t in TICKERS:
        for key, v in recs[t]["exp2"].items():
            m, h, f = key.split("|")
            for s, (lab, k) in enumerate(zip(v["labels"], v["seeds_long"])):
                ma = L.modal_action(k)
                rows.append({"ticker": t, "model": m, "horizon": int(h), "fold": f, "state": s, "state_label": lab,
                             "seeds_long": k, "modal_action": ma, "is_tie": ma == "tie"})
    modal_df = pd.DataFrame(rows)
    modal_df.to_csv(OUT / "tables/modal_actions_by_fold.csv.gz", index=False)

    # ---- stability summary
    srows = []
    for t in TICKERS:
        for m in MODELS:
            for h in HORIZONS:
                modal = [[L.modal_action(k) for k in recs[t]["exp2"][f"{m}|{h}|{f}"]["seeds_long"]] for f in FOLDS]
                shares = [L.fold_shares(mf) for mf in modal]
                exp1 = [L.modal_action(k) for k in recs[t]["exp1"][f"{m}|{h}"]["seeds_long"]]
                assert recs[t]["exp1"][f"{m}|{h}"]["labels"] == recs[t]["exp2"][f"{m}|{h}|fold_01"]["labels"]
                row = {"ticker": t, "model": m, "horizon": h, "n_states": len(modal[0]),
                       "all_folds_agree_share": L.all_folds_agree_share(modal),
                       "mean_pairwise_agreement": L.mean_pairwise_agreement(modal),
                       "chance_all_agree_share": L.chance_all_agree(shares),
                       "chance_pairwise": L.chance_pairwise(shares),
                       "n_ties": sum(mf.count("tie") for mf in modal),
                       "exp1_n_ties": exp1.count("tie")}
                for f, mf in zip(FOLDS, modal):
                    row[f"exp1_vs_{f}_agreement"] = L.exp1_fold_agreement(exp1, mf)
                srows.append(row)
    stab = pd.DataFrame(srows)
    stab.to_csv(OUT / "tables/stability_summary.csv.gz", index=False)

    # ---- bandit patterns
    brows = []
    for t in TICKERS:
        for h in HORIZONS:
            row = {"ticker": t, "horizon": h}
            labs = []
            for f in FOLDS:
                v = recs[t]["exp2"][f"bandit_3state|{h}|{f}"]
                by = {L.z_bucket(lab): k for lab, k in zip(v["labels"], v["seeds_long"])}
                assert set(by) == set(L.Z_BUCKETS)
                lab = L.pattern_label({z: L.modal_action(by[z]) for z in L.Z_BUCKETS})
                labs.append(lab)
                row[f"label_{f}"] = lab
                row[f"direction_index_{f}"] = L.direction_index(by["oversold"], by["overbought"])
                for z in L.Z_BUCKETS:
                    row[f"seeds_long_{z}_{f}"] = by[z]
            st = L.stable_label(labs)
            row["is_stable"] = st is not None
            row["stable_label"] = st if st else ""
            brows.append(row)
    bp = pd.DataFrame(brows)
    bp.to_csv(OUT / "tables/bandit_patterns.csv.gz", index=False)

    # ---- hypotheses
    h1_cells = bp[bp.ticker.isin(["XLU", "KO"])]
    h1_ok = h1_cells.apply(lambda r: bool(r.is_stable and r.stable_label == "MR exit"), axis=1)
    di_cols = [f"direction_index_{f}" for f in FOLDS]
    h2_cells = bp[bp.ticker.isin(["GOOGL", "TSLA", "NVDA"])]
    h2_ok = h2_cells.apply(lambda r: bool(all(r[c] <= 0 for c in di_cols)), axis=1)
    diff_all = stab.all_folds_agree_share - stab.chance_all_agree_share
    diff_pw = stab.mean_pairwise_agreement - stab.chance_pairwise
    n_above = int((diff_all > 0).sum())
    h3 = n_above >= 32 and float(diff_all.mean()) >= 0.10
    hyp = pd.DataFrame([
        {"hypothesis": "H1", "holds": bool(h1_ok.all()),
         "support": "; ".join(f"{r.ticker} h{r.horizon}: labels={[r[f'label_{f}'] for f in FOLDS]} stable={r.stable_label or 'none'}"
                              for _, r in h1_cells.iterrows()),
         "cells_satisfied": int(h1_ok.sum()), "cells_total": len(h1_ok)},
        {"hypothesis": "H2", "holds": bool(h2_ok.all()),
         "support": "; ".join(f"{r.ticker} h{r.horizon}: DI={[round(r[c], 2) for c in di_cols]}" for _, r in h2_cells.iterrows()),
         "cells_satisfied": int(h2_ok.sum()), "cells_total": len(h2_ok)},
        {"hypothesis": "H3", "holds": bool(h3),
         "support": (f"all-folds: observed>chance in {n_above}/42 cells (rule >=32); mean(obs-chance)={diff_all.mean():.4f} (rule >=0.10); "
                     f"pairwise (descriptive): observed>chance in {int((diff_pw > 0).sum())}/42, mean diff={diff_pw.mean():.4f}"),
         "cells_satisfied": n_above, "cells_total": 42,
         "mean_diff_all_folds": float(diff_all.mean()), "mean_diff_pairwise": float(diff_pw.mean())},
    ])
    hyp.to_csv(OUT / "tables/hypotheses.csv.gz", index=False)

    # ---- plots
    plot_policy(bp)
    plot_direction(bp)

    # ---- integrity after
    after = hash_inputs(files)
    with gzip.open(OUT / "input_hashes_after.json.gz", "wt") as f:
        json.dump(after, f)
    before = json.load(gzip.open(before_path, "rt"))
    diffs = [p for p in before if before[p] != after.get(p)] + [p for p in after if p not in before]
    mismatch_total = sum(recs[t]["greedy_mismatch_vs_stored"] for t in TICKERS)
    summary = {"n_inputs": len(files), "hash_differences": diffs, "greedy_mismatch_vs_stored_total": mismatch_total,
               "env_start": env_start, "env_end": env_snapshot()}
    json.dump(summary, open(OUT / "run_summary.json", "w"), indent=1)
    journal({"event": "run_complete", "hash_differences": len(diffs)})
    print(json.dumps({k: v for k, v in summary.items() if k != "hash_differences"}, indent=1))
    print("hash differences:", len(diffs))


ACTION_COLORS = {"cash": "#f5c7ae", "tie": "#dcdbd5", "long": "#86b6ef"}  # light orange / neutral / blue-250
TICKER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2 = "#0b0b0b", "#52514e"


def plot_policy(bp):
    order = ["cash", "tie", "long"]
    cmap = ListedColormap([ACTION_COLORS[a] for a in order])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))
    zs = ["OS", "N", "OB"]
    for ax, h in zip(axes, HORIZONS):
        sub = bp[bp.horizon == h].set_index("ticker").loc[TICKERS]
        grid = np.zeros((len(TICKERS), 12))
        txt = [[""] * 12 for _ in TICKERS]
        for i, t in enumerate(TICKERS):
            for j, f in enumerate(FOLDS):
                for k, z in enumerate(L.Z_BUCKETS):
                    n = int(sub.loc[t, f"seeds_long_{z}_{f}"])
                    grid[i, 3 * j + k] = order.index(L.modal_action(n))
                    txt[i][3 * j + k] = f"{n}/10"
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        for i in range(len(TICKERS)):
            for c in range(12):
                ax.text(c, i, txt[i][c], ha="center", va="center", fontsize=8, color=INK)
        for x in (2.5, 5.5, 8.5):
            ax.axvline(x, color="white", lw=4)
        ax.set_xticks(range(12), [f"F{j + 1}\n{z}" for j in range(4) for z in zs], fontsize=8, color=INK2)
        ax.set_yticks(range(len(TICKERS)), TICKERS, fontsize=9, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(f"Δ = {h}", fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a]) for a in ("long", "tie", "cash")]
    fig.legend(handles, ["modal long (≥6/10)", "tie (5/10)", "modal cash (≤4/10)"], loc="lower center",
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "plots/bandit_policy_by_fold.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_direction(bp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(1, 5)
    for ax, h in zip(axes, HORIZONS):
        sub = bp[bp.horizon == h].set_index("ticker")
        ax.axhline(0, color=INK2, lw=1)
        for t, c in zip(TICKERS, TICKER_COLORS):
            y = [sub.loc[t, f"direction_index_{f}"] for f in FOLDS]
            ax.plot(x, y, color=c, lw=2, marker="o", ms=6, label=t)
        ax.set_xticks(x, [f"fold {i}" for i in x], color=INK2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"Δ = {h}", color=INK)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("direction index  (oversold − overbought long share)", color=INK)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "plots/direction_index_by_fold.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
