"""Plain-language README results and the five-line summary, generated from the tables."""
from ppio import *
import math
import numpy as np
import pandas as pd
from pp_checkpoints import ready, receipt
from pp_properties import PROPERTIES, GROUP, REVERSION_GROUPS

MARK = '\n\n## Computed train-only results\n'
# For each reversion-type property, the sign of (real - null) that means "more mean-reverting".
MORE_REVERTING = dict(ar1_phi=-1, half_life=-1, VR_5=-1, VR_10=-1, zero_crossings_per_year=1, KM_med_2_0=-1, KM_med_1_5=-1,
                      reverted_share_2_0=1, reverted_share_1_5=1)

def pct(x, digits=2):
    return 'n/a' if x is None or not np.isfinite(x) else f'{100 * x:.{digits}f}%'

def num(x, digits=3):
    if x is None: return 'n/a'
    if isinstance(x, float) and math.isinf(x): return '∞' if x > 0 else '−∞'
    return 'n/a' if not np.isfinite(x) else f'{x:.{digits}f}'

def ci(lo, hi, f=num):
    return f'[{f(lo)}, {f(hi)}]'

def verdict(lo, hi, more='higher', less='lower', same='not distinguishable from'):
    if np.isfinite(lo) and lo > 0: return more
    if np.isfinite(hi) and hi < 0: return less
    return same

def tables():
    load = lambda name: pd.read_csv(TABLES / name)
    return dict(L=load('labels.csv'), G=load('G_vs_null.csv'), V=load('property_levels_vs_null.csv'),
                TS=load('ticker_effects_summary.csv'), top=load('top_degree_tickers.csv'), C=load('sample_counts.csv'),
                P=load('property_persistence.csv'))

def facts(t):
    L, G, TS = t['L'], t['G'], t['TS']
    g = G[(G.metric == 'G_pass') & (G.era == 'all') & (G.sector_relation == 'all')].iloc[0]
    f = G[(G.metric == 'FAST63') & (G.era == 'all') & (G.sector_relation == 'all')].iloc[0]
    persistent = L[L.persistent.astype(str) == 'True'].property.tolist()
    informative = L[L.reversion_informative.astype(str) == 'True'].property.tolist()
    fast = TS[(TS.indicator == 'FAST63') & (TS.test_year.astype(str) == 'pooled')].iloc[0]
    gt = TS[(TS.indicator == 'G_pass') & (TS.test_year.astype(str) == 'pooled')].iloc[0]
    return dict(G_real=g.real_rate, G_null=g.null_rate, G_diff=g.real_minus_null, G_lo=g.lo, G_hi=g.hi,
                FAST_real=f.real_rate, FAST_null=f.null_rate, FAST_lo=f.lo, FAST_hi=f.hi,
                persistent=persistent, informative=informative, ticker_label=fast.label, D=fast.dispersion_D, D_lo=fast.D_lo,
                D_hi=fast.D_hi, share=fast.top5_share, share_lo=fast.top5_share_lo, share_hi=fast.top5_share_hi,
                sim_D_p99=fast.sim_D_p99, sim_share=fast.sim_top5_share_mean, G_ticker_label=gt.label, G_D=gt.dispersion_D,
                G_share=gt.top5_share, km_persistent='KM_med_2_0' in persistent,
                gross_persistent='median_reverted_gross_2_0' in persistent)

def next_step_sentence(x):
    usable = [p for p in x['persistent'] if GROUP[p] in ('scale', 'comovement', 'reversion')]
    speed = ('speed (KM_med at 2.0) is persistent' if x['km_persistent'] else 'speed (KM_med at 2.0) is not persistent')
    if usable:
        lead = f"Persistent train properties ({', '.join(usable)}) are the defensible candidates to define the pool or Q-table groups"
    else:
        lead = 'No scale, co-movement or reversion property clears the persistence bar, so none can yet define the pool or Q-table groups'
    tail = '' if x['km_persistent'] else '; the frozen speed × size groups should not be used as they are'
    return f'{lead}; {speed}{tail} (suggestion only; the next prompt decides).'

def summary_lines(x):
    return [
        f"1. G pass rate: real {pct(x['G_real'])} vs random-walk null {pct(x['G_null'])} (difference {pct(x['G_diff'])}, 95% year-block {ci(x['G_lo'], x['G_hi'], pct)}).",
        '2. Persistent (pooled ρ lower bound ≥ 0.20): ' + (', '.join(x['persistent']) if x['persistent'] else 'none') + '.',
        '3. Reversion-informative (real − null ρ lower bound > 0): ' + (', '.join(x['informative']) if x['informative'] else 'none') + '.',
        f"4. FAST ticker effect: {x['ticker_label'].replace('_', ' ')} (dispersion D {num(x['D'], 2)}, 95% {ci(x['D_lo'], x['D_hi'], lambda v: num(v, 2))}; "
        f"top-5% tickers in {pct(x['share'], 1)} of FAST passes vs {pct(x['sim_share'], 1)} under a common rate).",
        '5. ' + next_step_sentence(x)]

def results_text(t, x):
    L, G, V, TS, top, C, P = t['L'], t['G'], t['V'], t['TS'], t['top'], t['C'], t['P']
    out = [MARK, f'Generated {utc()[:10]} from the sealed tables in `outputs/tables/`. Medians are weighted lower medians; '
           'intervals are 95% year-block percentile intervals over 2,000 fold resamples (seed 20261021).\n']
    out.append(f'Sample: {int(C.eligible.sum()):,} eligible pair-folds; {int(C.random_sample.sum()):,} random-sample rows '
               f'(up to 10,000 per fold, relation-stratified) plus {int(C.pool_members_not_drawn.sum()):,} capped-pool members not drawn at random. '
               'Every sampled inner (window A) statistic and Y−1 KM median reproduced the universe inventory exactly, and all 44,000 '
               'regenerated null pairs reproduced their stored statistics exactly (see `audit/reproduction_*.json`).\n')
    # 1. G
    out.append('### 1. Is G better than chance?\n')
    out.append(f"Overall, G passes {pct(x['G_real'])} of real eligible pair-folds and {pct(x['G_null'])} of fold-matched random-walk pairs: "
               f"real is {verdict(x['G_lo'], x['G_hi'])} the null (difference {pct(x['G_diff'])}, {ci(x['G_lo'], x['G_hi'], pct)}). "
               f"For reference, FAST63 passes {pct(x['FAST_real'])} real vs {pct(x['FAST_null'])} null ({ci(x['FAST_lo'], x['FAST_hi'], pct)}).\n")
    rows = ['| Condition | Real | Null | Real − null [95%] | Verdict |', '|---|---:|---:|---:|---|']
    for metric, label in [('cond_entries', '≥ 6 events'), ('cond_reverted_share', 'reverted share ≥ 0.80'),
                          ('cond_price_share', 'median price share ≥ 0.70'), ('cond_net_positive', 'mean net of 10 bps > 0'),
                          ('G_pass', 'all four (G)')]:
        r = G[(G.metric == metric) & (G.era == 'all') & (G.sector_relation == 'all')].iloc[0]
        rows.append(f'| {label} | {pct(r.real_rate)} | {pct(r.null_rate)} | {pct(r.real_minus_null)} {ci(r.lo, r.hi, pct)} | real {verdict(r.lo, r.hi)} null |')
    out.append('\n'.join(rows) + '\n')
    rows = ['| Era | Relation | Real G | Null G | Real − null [95%] |', '|---|---|---:|---:|---:|']
    for r in G[G.metric == 'G_pass'].itertuples():
        rows.append(f'| {r.era} | {r.sector_relation} | {pct(r.real_rate)} | {pct(r.null_rate)} | {pct(r.real_minus_null)} {ci(r.lo, r.hi, pct)} |')
    out.append('<details><summary>G by era and relation</summary>\n\n' + '\n'.join(rows) + '\n\n</details>\n')
    # 2. persistence
    out.append('### 2. Which properties persist, and which carry reversion information beyond chance?\n')
    out.append('Persistent (pooled within-fold Spearman ρ(A,B) lower bound ≥ 0.20): ' + (', '.join(f'`{p}`' for p in x['persistent']) or 'none') + '.  \n'
               'Reversion-informative (real ρ − null ρ lower bound > 0, reversion/event properties only): ' +
               (', '.join(f'`{p}`' for p in x['informative']) or 'none') + '.  \n'
               'Scale properties persist mechanically under the null (constant per-leg volatility), so for them the null ρ is only a reference.\n')
    rows = ['| Property | Group | Real ρ [95%] | Null ρ | Real − null [95%] | Top-tercile excess (tie-adj.) | Persistent | Rev.-informative | ≠ random walk (A) |',
            '|---|---|---:|---:|---:|---:|---|---|---|']
    pooled = P[P.stratum_type == 'pooled'].set_index('property')
    for r in L.itertuples():
        p = pooled.loc[r.property]
        rows.append(f'| `{r.property}` | {r.group} | {num(r.rho, 2)} {ci(r.rho_lo, r.rho_hi, lambda v: num(v, 2))} | {num(p.null_rho, 2)} | '
                    f'{num(r.rho_minus_null, 2)} {ci(r.rho_minus_null_lo, r.rho_minus_null_hi, lambda v: num(v, 2))} | '
                    f'{num(p.tercile_excess, 2)} ({num(p.tercile_tie_adjusted, 2)}) | {r.persistent} | {r.reversion_informative} | {r.different_from_random_walk} |')
    out.append('\n'.join(rows) + '\n')
    ds = P[(P.property == 'drift_sign') & (P.stratum_type == 'pooled')].iloc[0]
    out.append(f'Drift sign agreement minus 0.5: real {num(ds.sign_agreement_excess, 3)} {ci(ds.sign_agreement_excess_lo, ds.sign_agreement_excess_hi)}, '
               f'null {num(ds.null_sign_agreement_excess, 3)}. Relation, era and pool strata are in `property_persistence.csv`; '
               'pool members were selected on all three train years, so their A→B persistence is descriptive only.\n')
    # 3. levels
    out.append('### 3. Are real spreads more mean-reverting than random walks on average?\n')
    rows = ['| Property | Window | Real median | Null median | Real − null [95%] | Reading |', '|---|---|---:|---:|---:|---|']
    more = []
    for r in V.itertuples():
        if r.property not in MORE_REVERTING: continue
        direction = MORE_REVERTING[r.property]
        if r.excludes_zero:
            reading = 'more mean-reverting' if np.sign(r.lo if r.lo > 0 else r.hi) == direction else 'less mean-reverting'
        else:
            reading = 'no difference shown'
        if r.window == 'A': more.append((r.property, reading))
        rows.append(f'| `{r.property}` | {r.window} | {num(r.real_median)} | {num(r.null_median)} | {num(r.real_minus_null)} {ci(r.lo, r.hi)} | {reading} |')
    count = sum(reading == 'more mean-reverting' for _, reading in more)
    out.append(f'On window A, {count} of {len(more)} reversion measures show real spreads as more mean-reverting than random walks '
               'at the level of the typical pair, whether or not that is persistent per pair: ' +
               ', '.join(f'`{p}` {reading}' for p, reading in more) + '.\n')
    out.append('\n'.join(rows) + '\n')
    out.append('All 21 properties on both windows are in `property_levels_vs_null.csv`.\n')
    # 4. tickers
    out.append('### 4. Is FAST mainly a single-ticker effect?\n')
    f63 = TS[(TS.indicator == 'FAST63') & (TS.test_year.astype(str) != 'pooled')]
    out.append(f"Label: **{x['ticker_label'].replace('_', ' ')}**. Ticker pass counts are overdispersed relative to a common fold rate by "
               f"D = {num(x['D'], 2)} {ci(x['D_lo'], x['D_hi'], lambda v: num(v, 2))} (homogeneous-rate simulation 99th percentile "
               f"{num(x['sim_D_p99'], 2)}). The top 5% of tickers by degree take part in {pct(x['share'], 1)} "
               f"{ci(x['share_lo'], x['share_hi'], lambda v: pct(v, 1))} of FAST passes, against {pct(x['sim_share'], 1)} if every pair "
               f"passed at its fold rate. The largest single-fold degree is {int(f63.max_degree.max())}. For G: D = {num(x['G_D'], 2)}, "
               f"top-5% share {pct(x['G_share'], 1)}, label {x['G_ticker_label'].replace('_', ' ')}.\n")
    t30 = top[(top.indicator == 'FAST63') & (top['rank'] > 0)]
    ref = top[(top.indicator == 'FAST63') & (top['rank'] == 0)].iloc[0]
    sectors = t30.sector.value_counts()
    out.append(f"The top 30 high-degree FAST tickers have median train daily volatility {num(t30.median_train_vol.median(), 4)} vs "
               f"{num(ref.median_train_vol, 4)} for all tickers, median largest absolute daily return {pct(t30.max_abs_return.median(), 1)} vs "
               f"{pct(ref.max_abs_return, 1)}, median {num(t30.distinct_days_abs_return_gt_10pct.median(), 0)} vs "
               f"{num(ref.distinct_days_abs_return_gt_10pct, 0)} days with |return| > 10%, and median detrended-price AR(1) φ "
               f"{num(t30.median_ar1_phi_detrended.median(), 4)} vs {num(ref.median_ar1_phi_detrended, 4)}. Sectors: "
               + ', '.join(f'{s} {n}' for s, n in sectors.items()) + '. Full list: `top_degree_tickers.csv`.\n')
    # 5. implications
    out.append('### 5. Implications for the next step (suggestions only; the next prompt decides)\n')
    out.append('- ' + next_step_sentence(x) + '\n')
    if x['G_lo'] <= 0:
        out.append('- G does not pass real pairs more often than random walks, so G membership by itself is not evidence of a tradeable relationship.\n')
    else:
        out.append('- G passes real pairs more often than random walks; it remains a train-only screen, and its persistence is not tested here.\n')
    if not x['gross_persistent']:
        out.append('- Median reverted gross (the "size" axis of the frozen groups) is not persistent either; size groups would also be re-drawn each year by noise.\n')
    if x['ticker_label'] != 'not_ticker_driven':
        out.append('- Because FAST status is partly carried by individual tickers, any pool should cap or model ticker degree, and properties should be checked per leg.\n')
    out.append('\n### Limitations\n')
    out.append('- Window B is a single year, so event-based properties there are noisy; the null comparison absorbs part of this, not all.\n'
               '- The null uses independent geometric random walks with constant volatility; it has no volatility clustering or common factors.\n'
               '- Current GICS labels are applied to past dates; sector relations are not point-in-time.\n'
               '- Survivorship: stored prices omit delisted names, most in early eras.\n'
               '- Every year has been used before; this is a development diagnostic with no held-out part.\n'
               '- Capped pools were selected on all three train years, so pool-level A→B persistence includes selection on B.\n'
               '- Fold windows overlap in calendar years, so folds are not independent; the year-block bootstrap treats folds as exchangeable.\n')
    return ''.join(out)

def report():
    marker = HERE / 'audit/report_ready.json'
    if ready(marker): return read(marker)
    t = tables(); x = facts(t)
    from verify_property_persistence import registration_path
    registration = read(registration_path())
    readme = HERE / 'README.md'
    data = readme.read_bytes()
    top = data[:registration['README_top_bytes']]
    assert hashlib.sha256(top).hexdigest() == registration['README_top_sha256'], 'README top changed'
    text_ = results_text(t, x)
    if data == top:
        with readme.open('ab') as stream:
            stream.write(text_.encode())
    else:
        assert data[len(top):].startswith(MARK.encode()), 'Unexpected README content after the frozen top'
    lines = summary_lines(x)
    write(HERE / 'summary.json', dict(lines=lines, facts={k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in x.items()}))
    receipt(marker, [HERE / 'summary.json'], README_results_appended=True)
    return read(marker)
