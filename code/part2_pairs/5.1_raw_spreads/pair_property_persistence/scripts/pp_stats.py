"""Weighted rank statistics, terciles, year-block bootstrap and ticker dispersion.

+inf is an ordinary largest value: equal infinities tie and share a midrank.
NaN (undefined) must be removed by the caller. The bootstrap unit is the
nominal test-year fold; one multiplicity matrix serves every statistic.
"""
import math
import numpy as np

def midranks(x, w=None):
    """Weighted midranks: cumulative weight before a tie group plus half its weight."""
    x = np.asarray(x, float)
    assert not np.isnan(x).any()
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    order = np.argsort(x, kind='mergesort')
    xs, ws = x[order], w[order]
    if len(x) == 0:
        return np.zeros(0)
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(xs)])
    group_weight = np.add.reduceat(ws, starts)
    before = np.r_[0., np.cumsum(group_weight)[:-1]]
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(before + group_weight / 2, sizes)
    return ranks

def wspearman(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    if len(x) < 2:
        return math.nan
    rx, ry = midranks(x, w), midranks(y, w)
    total = w.sum()
    rx = rx - (w @ rx) / total
    ry = ry - (w @ ry) / total
    vx, vy = w @ (rx * rx), w @ (ry * ry)
    if not (vx > 0 and vy > 0):
        return math.nan
    return float((w @ (rx * ry)) / math.sqrt(vx * vy))

def top_tercile(x):
    """Top-tercile flags by midrank percentile u = midrank/n >= 2/3 (ties share membership)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.zeros(0, bool)
    u = midranks(x) / len(x)
    return u >= 2 / 3 - 1e-12

def tercile_persistence(a, b):
    """P(B top | A top) - 1/3 and the tie-adjusted P(B top | A top) - P(B top)."""
    ta, tb = top_tercile(a), top_tercile(b)
    p = (ta & tb).sum() / ta.sum() if ta.sum() else math.nan
    return p - 1 / 3, p - tb.mean()

def sign_agreement(a, b):
    a, b = np.sign(np.asarray(a, float)), np.sign(np.asarray(b, float))
    keep = (a != 0) & (b != 0)
    return float(np.mean(a[keep] == b[keep])) - 0.5 if keep.any() else math.nan

def bootstrap_matrix(folds=22, draws=2000, seed=20261021):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, folds, (draws, folds))
    return np.stack([np.bincount(row, minlength=folds) for row in indices]).astype(float)

def interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, 0
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), int(len(values))

def ratio(M, numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (M @ numerator) / (M @ denominator)

def fold_weighted(M, values, weights):
    """Fold-weighted mean of per-fold values; undefined folds get zero weight."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    defined = np.isfinite(values) & (weights > 0)
    return ratio(M, np.where(defined, weights * np.where(defined, values, 0.), 0.), np.where(defined, weights, 0.))

def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    keep = ~np.isnan(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return math.nan
    order = np.argsort(values, kind='mergesort')
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2 - 1e-9 * cumulative[-1])])

def bootstrap_weighted_median(M, values, weights, folds, block=100):
    """Weighted medians under fold multiplicities, vectorised in blocks of draws."""
    values, weights, folds = np.asarray(values, float), np.asarray(weights, float), np.asarray(folds, int)
    keep = ~np.isnan(values) & (weights > 0)
    values, weights, folds = values[keep], weights[keep], folds[keep]
    out = np.full(len(M), math.nan)
    if not len(values):
        return out
    order = np.argsort(values, kind='mergesort')
    values, weights, folds = values[order], weights[order], folds[order]
    for start in range(0, len(M), block):
        part = M[start:start + block][:, folds] * weights
        cumulative = np.cumsum(part, axis=1)
        half = cumulative[:, -1:] / 2 - 1e-9 * cumulative[:, -1:]
        index = (cumulative < half).sum(axis=1)
        valid = cumulative[:, -1] > 0
        out[start:start + block] = np.where(valid, values[np.minimum(index, len(values) - 1)], math.nan)
    return out

def difference(a, b):
    with np.errstate(invalid='ignore'):
        return np.asarray(a, float) - np.asarray(b, float)

# ---- ticker effects --------------------------------------------------------

def ticker_counts(leg_a, leg_b, passed):
    """Per-ticker eligible partners n and passes k for one fold."""
    leg_a, leg_b, passed = np.asarray(leg_a), np.asarray(leg_b), np.asarray(passed, bool)
    tickers, inverse = np.unique(np.r_[leg_a, leg_b], return_inverse=True)
    ia, ib = inverse[:len(leg_a)], inverse[len(leg_a):]
    n = np.bincount(ia, minlength=len(tickers)) + np.bincount(ib, minlength=len(tickers))
    k = np.bincount(ia, weights=passed, minlength=len(tickers)) + np.bincount(ib, weights=passed, minlength=len(tickers))
    return tickers, ia, ib, n.astype(float), k

def dispersion(n, k, passes, pairs):
    """Pearson X^2 against a common fold rate, number of tickers, variance numerator/denominator."""
    p = passes / pairs if pairs else math.nan
    if not (0 < p < 1):
        return dict(X2=math.nan, tickers=len(n), rate=p, var_num=math.nan, var_den=math.nan)
    expected = n * p
    x2 = float(np.sum((k - expected) ** 2 / (expected * (1 - p))))
    rates = k / n
    return dict(X2=x2, tickers=len(n), rate=p, var_num=float(np.sum((rates - p) ** 2)),
                var_den=float(np.sum(p * (1 - p) / n)))

def top_share(tickers, ia, ib, k, passed, fraction=0.05):
    """Share of passes involving at least one of the top ceil(5%) tickers by degree."""
    passed = np.asarray(passed, bool)
    total = int(passed.sum())
    m = int(math.ceil(fraction * len(tickers)))
    order = np.lexsort((tickers, -k))
    top = np.zeros(len(tickers), bool)
    top[order[:m]] = True
    involved = int(np.sum(passed & (top[ia] | top[ib])))
    return involved, total, top

def homogeneous_reference(ia, ib, n_tickers, pairs, rate, simulations, seed_sequence):
    """Pairs pass independently at the fold rate on the real pair graph."""
    rng = np.random.default_rng(seed_sequence)
    n = np.bincount(ia, minlength=n_tickers) + np.bincount(ib, minlength=n_tickers)
    rows = []
    tickers = np.array([f'{i:06d}' for i in range(n_tickers)])
    for _ in range(simulations):
        passed = rng.random(pairs) < rate
        k = np.bincount(ia, weights=passed, minlength=n_tickers) + np.bincount(ib, weights=passed, minlength=n_tickers)
        d = dispersion(n.astype(float), k, int(passed.sum()), pairs)
        involved, total, _ = top_share(tickers, ia, ib, k, passed)
        rows.append((d['X2'], d['tickers'], involved, total))
    return np.asarray(rows, float)
