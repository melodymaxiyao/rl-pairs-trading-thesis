"""Display-only test plots; plotting never changes parameter selection."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.turning_point_detection.new_stage1.slope_accuracy import greedy_match_events


def plot_accuracy(price_map, result_map, selected_map, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, skips = [], []
    for year in config["plot_test_years"]:
        fold_id = f"test_{year}"
        if fold_id not in result_map:
            skips.append({"test_year": year, "reason": "test fold absent from execution scope"})
            continue
        for h_ref in config["h_ref_grid"]:
            h, d, k = selected_map[(fold_id, h_ref)]
            for group in (1, 2):
                tickers = [t for t in config[f"tickers_group{group}"] if t in result_map[fold_id]]
                if not tickers:
                    skips.append({"test_year": year, "h_ref": h_ref, "group": group,
                                  "reason": "no evaluated ticker in group"})
                    continue
                fig, axes = plt.subplots(2*len(tickers), 1, figsize=(16, 4.6*len(tickers)), squeeze=False)
                for row, ticker in enumerate(tickers):
                    p = price_map[ticker]
                    result = result_map[fold_id][ticker]
                    begin, end = result["bounds"][(h_ref, "test")]
                    year_positions = np.flatnonzero(p.index.year == year)
                    refs = result["references"][(h_ref, "test")]
                    events = result["events"][(h, d, k)]
                    det = events[(events[:, 0] >= begin) & (events[:, 0] <= end)]
                    matched = greedy_match_events(det, refs, l_pre=config["match_l_pre"],
                                                  max_delay=int(config["match_m_multiplier"]*h_ref))
                    top, bottom = axes[2*row, 0], axes[2*row+1, 0]
                    top.plot(p.index[year_positions], p.iloc[year_positions], color="#475569", linewidth=.8, label="Log adjusted close")
                    for kind, name, marker, color in ((-1, "peak", "v", "#C2413B"), (1, "trough", "^", "#007F73")):
                        ref_positions = refs[refs[:, 1] == kind, 0]
                        det_positions = det[det[:, 1] == kind, 0]
                        top.scatter(p.index[ref_positions], p.iloc[ref_positions], marker=marker, s=60,
                                    facecolors="none", edgecolors=color, linewidths=1.1, label=f"Reference {name}", zorder=4)
                        top.scatter(p.index[det_positions], p.iloc[det_positions], marker=marker, s=22,
                                    color=color, label=f"Confirmed {name}", zorder=5)
                    for t, ref_t, _ in matched:
                        top.plot(p.index[[ref_t, t]], p.iloc[[ref_t, t]], color="#7C3AED", linewidth=.65, alpha=.7)
                    b, b_ref = result["slopes"][h], result["reference_slopes"][h_ref]
                    tau = k*result["sigma_b"][h]
                    bottom.plot(p.index[year_positions], b[year_positions], color="#007F73", linewidth=.9, label="Causal slope b_t")
                    bottom.plot(p.index[year_positions], b_ref[year_positions], color="#7C3AED", linewidth=.9, linestyle="--", label="Retrospective slope b_ref")
                    bottom.axhline(tau, color="#C2413B", linewidth=.8, linestyle=":", label="+/- tau (TRAIN only)")
                    bottom.axhline(-tau, color="#C2413B", linewidth=.8, linestyle=":")
                    bottom.axhline(0, color="#94A3B8", linewidth=.6)
                    top.set_title(f"{ticker} TEST {year} | frozen TRAIN params h={h}, d={d}, k={k:g}; h_ref={h_ref}")
                    top.set_ylabel("Log adjusted price")
                    bottom.set_ylabel("Log-price slope / trading day")
                    for axis in (top, bottom):
                        if end < year_positions[-1]:
                            axis.axvspan(p.index[end+1], p.index[year_positions[-1]], color="#E2E8F0", alpha=.6, label="Excluded reference tail")
                        axis.grid(axis="y", color="#E2E8F0", linewidth=.5)
                        axis.spines[["top", "right"]].set_visible(False)
                        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                        axis.legend(loc="best", fontsize=7, ncol=4)
                fig.suptitle("Accuracy only: hollow references, filled confirmations, purple matching lines", fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, .98))
                path = output_dir / f"slope_accuracy_test{year}_href{h_ref}_group{group}.png"
                fig.savefig(path, dpi=config["plot_dpi"], facecolor="white", bbox_inches="tight")
                plt.close(fig)
                outputs.append(path)
    return outputs, skips
