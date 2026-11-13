"""After-the-fact volatility tables/plots; never consumed by detector code."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_rows(price_map, calendar, evaluated, config, source, repeat):
    rows = []
    for choice in evaluated["selected"].itertuples(index=False):
        results = evaluated["results"][choice.fold_id]
        folds = calendar.loc[calendar.fold_id == choice.fold_id].set_index("ticker")
        for ticker, result in results.items():
            index = price_map[ticker].index
            fold = folds.loc[ticker]
            test_mask = (index >= pd.Timestamp(fold.test_start)) & (index <= pd.Timestamp(fold.test_end))
            slopes = result["slopes"][choice.h][test_mask]
            finite = slopes[np.isfinite(slopes)]
            sigma = result["sigma_b"][choice.h]
            sd_test = float(np.std(finite, ddof=config["std_ddof"])) if len(finite) > config["std_ddof"] else np.nan
            ratio = sd_test / sigma if sigma > 0 else np.nan
            effective = choice.k * sigma / sd_test if sd_test > 0 else np.nan
            begin, end = result["bounds"][(choice.h_ref, "test")]
            for kind in ("peak", "trough", "combined"):
                score = result["test_scores"][(choice.h_ref, choice.h, choice.d, choice.k, kind)]
                rows.append({"source": source, "repeat": repeat, "ticker": ticker, "fold_id": choice.fold_id,
                             "test_year": pd.Timestamp(fold.test_start).year, "h_ref": choice.h_ref,
                             "h": choice.h, "d": choice.d, "k": choice.k, "type": kind,
                             "sigma_b_train": sigma, "sd_test_b": sd_test, "vol_ratio": ratio,
                             "effective_k": effective, "f1": score["f1"], "precision": score["precision"],
                             "recall": score["recall"], "n_det": score["n_det"], "n_ref": score["n_ref"],
                             "n_match": score["n_match"], "test_slope_observations": len(finite),
                             "test_evaluation_sessions": end - begin + 1,
                             "diagnostic_note": "effective_k仅用于事后诊断，不进入检测器；sd_test取整个test年"})
    return rows


def plot_volatility(frame, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    styles = {ticker: markers[i % len(markers)] for i, ticker in enumerate(config["tickers"])}
    paths = []
    for (source, h_ref), group in frame.loc[frame.type == "combined"].groupby(["source", "h_ref"], sort=True):
        for metric in ("f1", "precision", "recall"):
            fig, ax = plt.subplots(figsize=(9, 6))
            plotted = 0
            for ticker, values in group.groupby("ticker", sort=True):
                valid = values.vol_ratio.gt(0) & np.isfinite(values.vol_ratio) & np.isfinite(values[metric])
                values = values.loc[valid]
                plotted += len(values)
                ax.scatter(np.log(values.vol_ratio), values[metric], marker=styles[ticker], s=45, alpha=.75, label=ticker)
            ax.axvline(0, color="grey", ls="--", lw=.8)
            ax.set_xlabel("log(sd_test(b) / sigma_b_train)")
            ax.set_ylabel(metric.upper() if metric == "f1" else metric.capitalize())
            suffix = "" if source == "real" else "_B1_repeat0"
            title = f"{source} · h_ref={h_ref} · selected TRAIN configuration"
            if source == "B1":
                title += f" · base seed={config['seed']}, repeat=0"
            ax.set_title(title)
            ax.set_ylim(-.02, 1.02); ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout()
            path = output / f"test_{metric}_vs_vol_ratio_href{h_ref}{suffix}.png"
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=config["plot_dpi"])
            plt.close(fig)
            paths.append({"file": str(path), "source": source, "h_ref": int(h_ref), "metric": metric,
                          "plotted_rows": plotted, "excluded_rows": len(group) - plotted})
    return paths
