"""LLT Kalman turning-point detector for llt_turning_point_v1 (see preregistration.md)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CFG = json.loads((OUT / "llt_config.json").read_text())
RESTORED = Path(CFG["restored_root"])
if str(RESTORED) not in sys.path:
    sys.path.insert(0, str(RESTORED))

import fyp_rl.data as fdata  # noqa: E402


def _blocked_download(*args, **kwargs):
    raise RuntimeError("download blocked in llt_turning_point_v1")


fdata.yf.download = _blocked_download

from scipy.optimize import minimize  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from experiments.turning_point_detection.new_stage1.run_slope_accuracy import (  # noqa: E402
    load_accuracy_config, load_prices, macro_score, add_stitched, aggregate_selected,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy import (  # noqa: E402
    score_events, generate_random_walk, stitch_scores,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy_diagnostics import (  # noqa: E402
    evaluate_fold_v2, null_seed, empirical_null_path, midrank_percentile,
)
from experiments.turning_point_detection.new_stage1.kernel_smoothing import local_linear_slope_weights  # noqa: E402

TABLES = RESTORED / CFG["tables_dir"]
KINDS = ("peak", "trough", "combined")
PART2_QUANTILES = [0.25, 0.5, 0.75]
LOG_LO, LOG_HI = (math.log(b) for b in CFG["ratio_bounds"])
LOG_EDGE = math.log(CFG["boundary_factor"])
LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------- inputs
def part2_config():
    return load_accuracy_config(RESTORED / CFG["part2_config"])


def load_all_prices(tickers=None):
    cfg = part2_config()
    return {t: load_prices(t, cfg) for t in (tickers or CFG["tickers"])}


def fold_calendar():
    cal = pd.read_csv(TABLES / CFG["fold_calendar_file"])
    return cal.sort_values(["fold_id", "ticker"]).reset_index(drop=True)


def input_files():
    """Every restored file this task reads or imports (imports resolved through sys.modules)."""
    import experiments.turning_point_detection.new_stage1.run_slope_accuracy_diagnostics  # noqa: F401
    files = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(RESTORED)}
    files |= {RESTORED / CFG["part2_config"], RESTORED / CFG["part2_diag_config"],
              RESTORED / ".venv/lib/python3.13/site-packages/py.py"}
    files |= {TABLES / CFG[k] for k in ("fold_calendar_file", "detector_events_file", "selected_params_file",
                                        "table51_file", "table52_file", "null_samples_file",
                                        "null_generation_file", "vol_file")}
    files |= {RESTORED / "data/raw" / f"{t.lower()}_daily_adjusted_ohlcv.csv" for t in CFG["tickers"]}
    return sorted(str(f.relative_to(RESTORED)) for f in files)


def hash_inputs():
    import hashlib
    return {f: hashlib.sha256((RESTORED / f).read_bytes()).hexdigest() for f in input_files()}


# ---------------------------------------------------------------- Kalman filter
def _variances(param, ratios):
    """Normalised (H, Q_eta, Q_zeta) with scale 1 for parameterisation 'E' or 'H'."""
    a, b = ratios
    return (1.0, a, b) if param == "E" else (a, 1.0, b)


def kalman(p, H, Qe, Qz, *, n_lik=None, steady_tol=None, want_states=False):
    """Exact-diffuse LLT filter with unit scale.

    Returns sum(v^2/F), sum(log F) and the number of prediction errors over positions
    2..n_lik-1; with want_states also the filtered slope and its normalised variance
    for every position (NaN at position 0).
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    n_lik = n if n_lik is None else n_lik
    tol = CFG["steady_state_tol"] if steady_tol is None else steady_tol
    a0, a1 = p[1], p[1] - p[0]
    P00, P01, P11 = H, H, 2 * H + Qe + Qz
    if want_states:
        S = np.full(n, np.nan)
        PS = np.full(n, np.nan)
        S[1], PS[1] = a1, P11
    ssq = slog = 0.0
    prev = None
    t = 2
    pl = p.tolist()
    while t < n:
        pp00 = P00 + 2 * P01 + P11 + Qe
        pp01 = P01 + P11
        pp11 = P11 + Qz
        if prev is not None:
            m = max(abs(pp00), abs(pp01), abs(pp11))
            if max(abs(pp00 - prev[0]), abs(pp01 - prev[1]), abs(pp11 - prev[2])) <= tol * m:
                break
        prev = (pp00, pp01, pp11)
        F = pp00 + H
        v = pl[t] - a0 - a1
        k0, k1 = pp00 / F, pp01 / F
        a0, a1 = a0 + a1 + k0 * v, a1 + k1 * v
        P00, P01, P11 = pp00 - pp00 * k0, pp01 - pp00 * k1, pp11 - pp01 * k1
        if t < n_lik:
            ssq += v * v / F
            slog += math.log(F)
        if want_states:
            S[t], PS[t] = a1, P11
        t += 1
    if t < n:
        # Steady state: fixed gain from here on, run as a 2nd-order IIR filter.
        F = pp00 + H
        k0, k1 = pp00 / F, pp01 / F
        P11s = pp11 - pp01 * k1
        # x_t = M x_{t-1} + K p_t with M = (I - K e1') T.
        m00, m01, m10, m11 = 1 - k0, 1 - k0, -k1, 1 - k1
        den = np.array([1.0, -(m00 + m11), m00 * m11 - m01 * m10])
        num_l = (k0, m01 * k1 - m11 * k0)
        num_s = (k1, m10 * k0 - m00 * k1)
        b_v = np.array([1.0, den[1] - num_l[0] - num_s[0], den[2] - num_l[1] - num_s[1]])
        b_s = np.array([num_s[0], num_s[1], 0.0])
        x = (a0, a1)
        Mx = (m00 * x[0] + m01 * x[1], m10 * x[0] + m11 * x[1])
        MMx = (m00 * Mx[0] + m01 * Mx[1], m10 * Mx[0] + m11 * Mx[1])
        rest = p[t:]
        y0, y1 = -(x[0] + x[1]), -(Mx[0] + Mx[1])
        v = lfilter(b_v, den, rest, zi=np.array([y0, y1 + den[1] * y0]))[0]
        stop = max(0, min(n, n_lik) - t)
        ssq += float(np.dot(v[:stop], v[:stop])) / F
        slog += stop * math.log(F)
        if want_states:
            y0, y1 = Mx[1], MMx[1]
            S[t:] = lfilter(b_s, den, rest, zi=np.array([y0, y1 + den[1] * y0]))[0]
            PS[t:] = P11s
    n_err = max(0, min(n, n_lik) - 2)
    out = {"ssq": ssq, "slog": slog, "n_err": n_err, "steady_from": t if t < n else None}
    if want_states:
        out.update(S=S, PS=PS)
    return out


def concentrated_loglik(p, param, ratios, *, n_lik=None, steady_tol=None):
    H, Qe, Qz = _variances(param, ratios)
    k = kalman(p, H, Qe, Qz, n_lik=n_lik, steady_tol=steady_tol)
    n = k["n_err"]
    s2 = k["ssq"] / n
    return -0.5 * n * (LOG2PI + 1 + math.log(s2)) - 0.5 * k["slog"], s2


def full_loglik(p, sig_eps2, sig_eta2, sig_zeta2, *, steady_tol=None):
    """Unconcentrated exact-diffuse log-likelihood at absolute variances (for tests)."""
    k = kalman(p, sig_eps2, sig_eta2, sig_zeta2, steady_tol=steady_tol)
    return -0.5 * (k["n_err"] * LOG2PI + k["slog"] + k["ssq"])


# ---------------------------------------------------------------- estimation
def _optimise(p, param, starts):
    best = None
    for i, s in enumerate(starts):
        x0 = np.clip(np.asarray(s, dtype=float), LOG_LO, LOG_HI)

        def f(theta):
            ll, _ = concentrated_loglik(p, param, np.exp(theta))
            return -ll if np.isfinite(ll) else 1e300

        r = minimize(f, x0, method=CFG["optimizer"]["method"], bounds=[(LOG_LO, LOG_HI)] * 2,
                     options={"maxiter": CFG["optimizer"]["maxiter"]})
        ll = -float(r.fun)
        if best is None or ll > best["loglik"]:
            best = {"theta": r.x.copy(), "loglik": ll, "start_index": i, "converged": bool(r.success)}
    return best


def _to_E(param, ratios):
    if param == "E":
        return ratios
    r_eps, qz = ratios
    return (1.0 / r_eps, qz / r_eps)


def _to_H(param, ratios):
    if param == "H":
        return ratios
    qe, qz = ratios
    return (1.0 / qe, qz / qe)


def fit_llt(p_train, warm=None):
    """MLE on TRAIN prices. warm = (param, ratios) of the previous fold or None."""
    ln10 = math.log(10)
    starts_E = [np.array(s) * ln10 for s in CFG["starts_E_log10"]]
    if warm is not None:
        starts_E.append(np.log(np.clip(_to_E(*warm), *CFG["ratio_bounds"])))
    E = _optimise(p_train, "E", starts_E)
    E_ratios = np.exp(E["theta"])
    result = {"param": "E", "fit": E, "loglik_E": E["loglik"], "loglik_H": np.nan}
    if np.any(E["theta"] >= LOG_HI - LOG_EDGE):
        starts_H = [np.array(s) * ln10 for s in CFG["starts_H_log10"]]
        if warm is not None:
            starts_H.append(np.log(np.clip(_to_H(*warm), *CFG["ratio_bounds"])))
        Hf = _optimise(p_train, "H", starts_H)
        result["loglik_H"] = Hf["loglik"]
        if Hf["loglik"] > E["loglik"]:
            result.update(param="H", fit=Hf)
    fit = result["fit"]
    ratios = tuple(float(x) for x in np.exp(fit["theta"]))
    _, s2 = concentrated_loglik(p_train, result["param"], ratios)
    H, Qe, Qz = _variances(result["param"], ratios)
    on_lo = bool(np.any(fit["theta"] <= LOG_LO + LOG_EDGE))
    on_hi = bool(np.any(fit["theta"] >= LOG_HI - LOG_EDGE))
    return {"param": result["param"], "ratios": ratios, "scale": s2,
            "sigma_eps2": H * s2, "sigma_eta2": Qe * s2, "sigma_zeta2": Qz * s2,
            "q_eta": Qe / H, "q_zeta": Qz / H, "zeta_over_eta": Qz / Qe,
            "loglik": fit["loglik"], "loglik_E": result["loglik_E"], "loglik_H": result["loglik_H"],
            "on_boundary": on_lo or on_hi, "on_lower": on_lo, "on_upper": on_hi,
            "best_start": fit["start_index"], "converged": fit["converged"], "n_train": len(p_train)}


def filter_pi(p_segment, fit):
    H, Qe, Qz = _variances(fit["param"], fit["ratios"])
    k = kalman(p_segment, H, Qe, Qz, want_states=True)
    sd = np.sqrt(k["PS"] * fit["scale"])
    return ndtr(k["S"] / sd), k["S"]


# ---------------------------------------------------------------- detection
def detect_pi_events(pi, c, d):
    """Confirmed turns from pi (defined from position 1). Returns [[position, sign]] like Part II."""
    pi = np.asarray(pi, dtype=float)
    n = len(pi)
    events = []
    if n < 2:
        return np.empty((0, 2), dtype=int)
    up = pi[1] > 0.5
    low = np.zeros(n, dtype=bool)
    high = np.zeros(n, dtype=bool)
    low[1:] = pi[1:] < 1 - c
    high[1:] = pi[1:] > c
    # run[t] = number of consecutive days ending at t satisfying the condition
    def runs(mask):
        r = np.zeros(n, dtype=int)
        cnt = 0
        for i, m in enumerate(mask):
            cnt = cnt + 1 if m else 0
            r[i] = cnt
        return r
    rl, rh = runs(low), runs(high)
    for t in range(2, n):
        if up and rl[t] >= d:
            events.append((t, -1))
            up = False
        elif not up and rh[t] >= d:
            events.append((t, 1))
            up = True
    return np.asarray(events, dtype=int).reshape(-1, 2)


# ---------------------------------------------------------------- one ticker x fold
def reference_frame(prices, fold, p2cfg, h_ref_grid):
    """References and bounds from new_stage1's own evaluate_fold_v2 (1-point Part II grid)."""
    cfg = {**p2cfg, "h_grid": [3], "d_grid": [1], "k_grid": [0], "h_ref_grid": list(h_ref_grid)}
    r = evaluate_fold_v2(prices, fold, cfg)
    return r["references"], r["bounds"]


def _pack(score):
    return [score["n_ref"], score["n_det"], score["n_match"], [int(x) for x in score["delays"]]]


def unpack(entry, fold_id=None):
    """Rebuild a per-ticker score dict exactly as score_events returns it (metrics via stitch_scores)."""
    raw = {"n_ref": entry[0], "n_det": entry[1], "n_match": entry[2], "delays": list(entry[3])}
    s = stitch_scores([raw], quantiles=PART2_QUANTILES)
    if fold_id is not None:
        s["fold_id"] = fold_id
    return s


def evaluate_ticker_fold(prices, fold, p2cfg, h_ref_grid, warm=None):
    index = prices.index
    start = index.searchsorted(pd.Timestamp(fold["train_start"]))
    stop = index.searchsorted(pd.Timestamp(fold["test_end"]), side="right")
    seg = prices.iloc[start:stop]
    train = seg.loc[seg.index <= pd.Timestamp(fold["train_end"])]
    fit = fit_llt(train.to_numpy(), warm=warm)
    pi, _ = filter_pi(seg.to_numpy(), fit)
    refs, bounds = reference_frame(prices, fold, p2cfg, h_ref_grid)
    scores = {}
    spacing = {}
    for c in CFG["c_grid"]:
        for d in CFG["d_grid"]:
            ev = detect_pi_events(pi, c, d)
            ev[:, 0] += start
            for h_ref in h_ref_grid:
                for segment in ("train", "test"):
                    b, e = bounds[(h_ref, segment)]
                    det = ev[(ev[:, 0] >= b) & (ev[:, 0] <= e)]
                    sc = score_events(det, refs[(h_ref, segment)], l_pre=p2cfg["match_l_pre"],
                                      max_delay=int(p2cfg["match_m_multiplier"] * h_ref),
                                      quantiles=p2cfg["delay_quantiles"])
                    kinds = ("combined",) if segment == "train" else KINDS
                    for kind in kinds:
                        scores[f"{h_ref}|{segment}|{c}|{d}|{kind}"] = _pack(sc[kind])
    for h_ref in h_ref_grid:
        pos = refs[(h_ref, "test")][:, 0]
        gaps = np.diff(np.sort(pos))
        spacing[str(h_ref)] = [int(gaps.sum()), int(len(gaps))]
    return {"fold_id": fold["fold_id"], "ticker": fold["ticker"], "fit": fit, "scores": scores,
            "ref_spacing": spacing}


# ---------------------------------------------------------------- selection and aggregation
def select_cd(train_scores_by_ticker, h_ref, p2cfg):
    """train_scores_by_ticker: {ticker: scores dict of one fold}. Returns ((c, d), chosen row)."""
    rows = []
    for c in CFG["c_grid"]:
        for d in CFG["d_grid"]:
            key = f"{h_ref}|train|{c}|{d}|combined"
            m = macro_score([unpack(s[key]) for s in train_scores_by_ticker.values()], p2cfg["delay_quantiles"])
            rows.append({"c": c, "d": d, "f1": m["f1"], "delay_median": m["delay_median"],
                         "precision": m["precision"], "recall": m["recall"]})
    frame = pd.DataFrame(rows)
    frame["f1_sort"] = frame.f1.fillna(0)
    frame["delay_sort"] = frame.delay_median.fillna(np.inf)
    best = frame.sort_values(["f1_sort", "delay_sort", "c", "d"], ascending=[False, True, True, True],
                             kind="stable").iloc[0]
    return (float(best.c), int(best.d)), best.drop(["f1_sort", "delay_sort"]).to_dict()


def aggregate_source(units, tickers, h_ref_grid, p2cfg):
    """units: {ticker: [fold records]} for one (source, repeat). Returns summary, selections, per-fold test rows."""
    by_fold = {}
    for t, recs in units.items():
        for r in recs:
            by_fold.setdefault(r["fold_id"], {})[t] = r
    stitched, selected, per_fold = {}, [], []
    for fold_id in sorted(by_fold):
        recs = by_fold[fold_id]
        for h_ref in h_ref_grid:
            (c, d), chosen = select_cd({t: r["scores"] for t, r in recs.items()}, h_ref, p2cfg)
            test = macro_score([unpack(r["scores"][f"{h_ref}|test|{c}|{d}|combined"]) for r in recs.values()],
                               p2cfg["delay_quantiles"])
            selected.append({"h_ref": h_ref, "fold_id": fold_id, "c": c, "d": d, "train_f1": chosen["f1"],
                             "train_precision": chosen["precision"], "train_recall": chosen["recall"],
                             "train_delay_median": chosen["delay_median"], "test_f1": test["f1"],
                             "test_delay_median": test["delay_median"], "n_tickers": len(recs),
                             "selection_tickers": ";".join(sorted(recs))})
            for t, r in recs.items():
                res = {"test_scores": {(h_ref, c, d, kind): unpack(r["scores"][f"{h_ref}|test|{c}|{d}|{kind}"])
                                       for kind in KINDS}}
                add_stitched(stitched, t, fold_id, res, h_ref, (c, d))
                s = res["test_scores"][(h_ref, c, d, "combined")]
                per_fold.append({"ticker": t, "fold_id": fold_id, "h_ref": h_ref, "c": c, "d": d,
                                 "n_det": s["n_det"], "n_ref": s["n_ref"], "n_match": s["n_match"]})
    cfg = {**p2cfg, "h_ref_grid": list(h_ref_grid), "tickers": list(tickers)}
    return aggregate_selected(stitched, cfg), pd.DataFrame(selected), pd.DataFrame(per_fold)


# ---------------------------------------------------------------- effective bandwidth
def steady_state_weights(fit):
    bw = CFG["bandwidth"]
    H, Qe, Qz = _variances(fit["param"], fit["ratios"])
    P00, P01, P11 = H, H, 2 * H + Qe + Qz            # filtered covariance after 2 observations
    prev = None
    for it in range(bw["riccati_max_iter"]):
        pp00, pp01, pp11 = P00 + 2 * P01 + P11 + Qe, P01 + P11, P11 + Qz
        if prev is not None:
            m = max(abs(pp00), abs(pp01), abs(pp11))
            if max(abs(pp00 - prev[0]), abs(pp01 - prev[1]), abs(pp11 - prev[2])) <= bw["riccati_tol"] * m:
                break
        prev = (pp00, pp01, pp11)
        F = pp00 + H
        k0, k1 = pp00 / F, pp01 / F
        P00, P01, P11 = pp00 - pp00 * k0, pp01 - pp00 * k1, pp11 - pp01 * k1
    else:
        return None, {"riccati_converged": False}
    F = pp00 + H
    k0, k1 = pp00 / F, pp01 / F
    m00, m01, m10, m11 = 1 - k0, 1 - k0, -k1, 1 - k1
    w, x0, x1 = [], k0, k1
    run = 0
    wmax = 0.0
    for j in range(bw["weight_max_j"] + 1):
        w.append(x1)
        wmax = max(wmax, abs(x1))
        run = run + 1 if abs(x1) < bw["weight_tail_rel"] * wmax else 0
        if run >= bw["weight_tail_run"]:
            break
        x0, x1 = m00 * x0 + m01 * x1, m10 * x0 + m11 * x1
    return np.asarray(w), {"riccati_converged": True, "riccati_iterations": it, "n_weights": len(w),
                           "weights_truncated": run < bw["weight_tail_run"]}


def lag_stats(w):
    a = np.abs(w)
    j = np.arange(len(w))
    cum = np.cumsum(a)
    return float(np.dot(j, a) / a.sum()), int(np.searchsorted(cum, 0.9 * a.sum()))


_LAG_GRID = None


def part2_lag_grid():
    global _LAG_GRID
    if _LAG_GRID is None:
        lo, hi = CFG["bandwidth"]["equiv_h_grid_m"]
        hs, lags = [], []
        for m in range(lo, hi + 1):
            h = m / 3
            x, l = local_linear_slope_weights(h, side="causal", window_multiplier=CFG["bandwidth"]["window_multiplier"])
            a = np.abs(l)
            hs.append(h)
            lags.append(float(np.dot(-x, a) / a.sum()))
        _LAG_GRID = (np.asarray(hs), np.asarray(lags))
    return _LAG_GRID


def equivalent_h(mean_lag):
    hs, lags = part2_lag_grid()
    if mean_lag < lags[0]:
        return np.nan, "below_range"
    if mean_lag > lags[-1]:
        return np.nan, "above_range"
    if np.all(np.diff(lags) > 0):
        return float(np.interp(mean_lag, lags, hs)), "interpolated"
    return float(hs[np.argmax(lags >= mean_lag)]), "first_grid_h_at_or_above"


def bandwidth_row(fit):
    w, info = steady_state_weights(fit)
    if w is None:
        return {"mean_lag": np.nan, "d90": np.nan, "equiv_h": np.nan, "equiv_h_flag": "riccati_not_converged", **info}
    mean_lag, d90 = lag_stats(w)
    eh, flag = equivalent_h(mean_lag)
    return {"mean_lag": mean_lag, "d90": d90, "equiv_h": eh, "equiv_h_flag": flag,
            "weight_sum": float(w.sum()), "weight_lag_moment": float(-np.dot(np.arange(len(w)), w)), **info}


def holm(pvals, alpha):
    order = np.argsort(pvals, kind="stable")
    m = len(pvals)
    adj = np.empty(m)
    reject = np.zeros(m, dtype=bool)
    running, stop = 0.0, False
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
        if not stop and pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            stop = True
    return adj, reject
