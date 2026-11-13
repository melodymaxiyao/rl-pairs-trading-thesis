"""Binding event-consistency check (preregistration §2). Writes logs/event_consistency.json; exit 2 on mismatch."""
import json
import sys
from pathlib import Path

import pandas as pd

import gateb_lib as G

OUT = Path(__file__).resolve().parent


def main():
    cfg = G.load_config(OUT)
    det_cfg = G.load_detector_config(cfg)
    tabs = G.load_tables(cfg)
    stored = pd.read_csv(G.RESTORED / cfg["detector_events_file"])
    cal = tabs["calendar"]
    report = {"binding": None, "all": []}

    c = cfg["consistency_check"]
    logp, _ = G.load_ticker_prices(c["ticker"], det_cfg)
    fold = cal[(cal.ticker == c["ticker"]) & (cal.fold_id == c["fold_id"])].iloc[0].to_dict()
    h, d, k = G.selected_params(tabs["selected"], c["fold_id"], c["h_ref"])
    ev = G.fold_events(logp, fold, h, d, k, det_cfg)
    st = stored[(stored.ticker == c["ticker"]) & (stored.fold_id == c["fold_id"]) & (stored.h == h)
                & (stored.d == d) & (stored.k == k)]
    res = G.compare_with_detector_events(ev, st)
    seg_ok = bool((ev.segment.values == st.sort_values("date").segment.values).all()) if res["identical"] else False
    report["binding"] = {**c, "h": h, "d": d, "k": k, **res, "segments_identical": seg_ok,
                         "n_test_events": int((ev.segment == "test").sum())}
    print("BINDING:", json.dumps(report["binding"]))
    if not (res["identical"] and seg_ok):
        json.dump(report, open(OUT / "logs/event_consistency.json", "w"), indent=1, default=str)
        sys.exit(2)

    n_bad = 0
    for ticker in cfg["tickers"]:
        logp, _ = G.load_ticker_prices(ticker, det_cfg)
        for fold in cal[cal.ticker == ticker].to_dict("records"):
            for h_ref in sorted({s["h_ref"] for s in cfg["settings"]}):
                h, d, k = G.selected_params(tabs["selected"], fold["fold_id"], h_ref)
                ev = G.fold_events(logp, fold, h, d, k, det_cfg)
                st = stored[(stored.ticker == ticker) & (stored.fold_id == fold["fold_id"]) & (stored.h == h)
                            & (stored.d == d) & (stored.k == k)]
                r = G.compare_with_detector_events(ev, st)
                n_bad += not r["identical"]
                report["all"].append({"ticker": ticker, "fold_id": fold["fold_id"], "h_ref": h_ref,
                                      "h": h, "d": d, "k": k, **r})
    report["all_summary"] = {"n_comparisons": len(report["all"]), "n_mismatch": n_bad}
    json.dump(report, open(OUT / "logs/event_consistency.json", "w"), indent=1, default=str)
    print("ALL:", report["all_summary"])
    if n_bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
