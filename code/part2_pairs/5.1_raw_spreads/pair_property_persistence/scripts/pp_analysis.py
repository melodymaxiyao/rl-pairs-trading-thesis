"""Tables and labels from completed fold checkpoints. Reads no prices."""
from ppio import *
import gzip, io, math
import numpy as np
import pandas as pd
from pp_checkpoints import ready, receipt, load_rows, load_object
from pp_properties import PROPERTIES, NAMES, GROUP, REVERSION_GROUPS
from pp_sample import RELATIONS, POOLS
import pp_stats as S

ERAS = ['E1', 'E2', 'E3', 'E4', 'E5']
METRICS = ['G_pass', 'cond_entries', 'cond_reverted_share', 'cond_price_share', 'cond_net_positive', 'FAST63']
MIN_FOLD_ROWS = 20
SIMULATIONS = 200

def csv_new(path, frame):
    path = inside(path)
    pending = path.with_name(path.name + '.pending_' + str(time.time_ns()))
    with pending.open('xb') as binary:
        if path.suffix == '.gz':
            with gzip.GzipFile(filename='', fileobj=binary, mode='wb', mtime=0) as gz:
                with io.TextIOWrapper(gz, newline='') as stream:
                    frame.to_csv(stream, index=False)
        else:
            # Repair checkpoint_dir_v1: detach without a context manager (closing a detached wrapper raises).
            stream = io.TextIOWrapper(binary, newline='')
            frame.to_csv(stream, index=False)
            stream.flush(); stream.detach()
    if path.exists():
        assert sha(path) == sha(pending), ('Existing table differs', path)
    else:
        os.link(pending, path)
    pending.unlink()
    record('CSV_generated', CSV_rows_written=len(frame), CSV_files_written=1)
    return path

def chunk_rows(kind, year):
    rows = []
    for marker in sorted((HERE / 'runtime/checkpoints').glob(f'{kind}_{year}_[0-9][0-9][0-9][0-9][0-9].json')):
        rows.extend(load_rows(marker))
    return rows

def load_data():
    real, null, eligible, tickers = [], [], [], []
    for year in YEARS:
        done = read(HERE / f'runtime/final_fold_{year}.json')
        part = chunk_rows('real', year); assert len(part) == done['sampled_pair_folds']
        real.extend(part)
        part = chunk_rows('null', year); assert len(part) == 2000
        null.extend(part)
        frame = load_object(f'eligible_{year}'); frame['test_year'] = year; eligible.append(frame)
        for ticker, values in load_rows(HERE / f'runtime/checkpoints/ticker_stats_{year}.json').items():
            tickers.append(dict(test_year=year, ticker=ticker, **values))
    R = pd.DataFrame(real); N = pd.DataFrame(null); E = pd.concat(eligible, ignore_index=True); T = pd.DataFrame(tickers)
    for frame in (R, N, E, T):
        frame['fi'] = frame.test_year.astype(int) - YEARS[0]
    for frame in (R, N):
        for name in NAMES:
            for w in 'AB':
                frame[f'{name}_{w}'] = frame[f'{name}_{w}'].astype(float)
    R['w'] = R.ipw.astype(float).where(R.in_random.astype(bool), 0.)
    N['w'] = 1.
    return R, N, E, T

# ---- persistence ---------------------------------------------------------------

def fold_arrays(frame, name, weight, group):
    """Per-fold Spearman, tercile and sign aggregates for one property."""
    out = {k: np.zeros(len(YEARS)) for k in ['W', 'n', 'n11', 'nA', 'nB', 'tot', 'agree', 'agree_den']}
    out['rho'] = np.full(len(YEARS), math.nan)
    a_all, b_all = frame[name + '_A'].to_numpy(float), frame[name + '_B'].to_numpy(float)
    keep_all = ~np.isnan(a_all) & ~np.isnan(b_all)
    folds = frame.fi.to_numpy(int); w_all = frame[weight].to_numpy(float)
    groups = frame[group].to_numpy() if group else np.zeros(len(frame))
    for f in range(len(YEARS)):
        mask = (folds == f) & keep_all & (w_all > 0)
        if not mask.any(): continue
        a, b, w, g = a_all[mask], b_all[mask], w_all[mask], groups[mask]
        out['n'][f] = mask.sum()
        if mask.sum() >= MIN_FOLD_ROWS:
            out['rho'][f] = S.wspearman(a, b, w); out['W'][f] = w.sum()
        for key in np.unique(g):
            sel = g == key
            ta, tb = S.top_tercile(a[sel]), S.top_tercile(b[sel])
            ws = w[sel]
            out['n11'][f] += ws[ta & tb].sum(); out['nA'][f] += ws[ta].sum()
            out['nB'][f] += ws[tb].sum(); out['tot'][f] += ws.sum()
        sa, sb = np.sign(a), np.sign(b); both = (sa != 0) & (sb != 0)
        out['agree'][f] = w[both & (sa == sb)].sum(); out['agree_den'][f] = w[both].sum()
    raw_keep = keep_all & (w_all > 0)
    out['raw'] = S.wspearman(a_all[raw_keep], b_all[raw_keep], w_all[raw_keep]) if raw_keep.sum() >= 2 else math.nan
    return out

def per_fold(values, numerator, denominator):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(values[denominator] > 0, values[numerator] / np.where(values[denominator] > 0, values[denominator], 1), math.nan)

def summarise(MM, real, null, null_rows, weight_fold):
    """Point (first row of MM is ones) and bootstrap draws for real, like-for-like null and differences."""
    W = np.where(np.isfinite(real['rho']), real['W'], 0.)
    rho = S.fold_weighted(MM, real['rho'], W)
    rho_null = S.fold_weighted(MM, null['rho'], np.where(np.isfinite(null['rho']), W, 0.))
    p_real = S.ratio(MM, real['n11'], real['nA'])
    terc = p_real - 1 / 3
    terc_adj = p_real - S.ratio(MM, real['nB'], real['tot'])
    null_p = per_fold(null, 'n11', 'nA'); null_b = per_fold(null, 'nB', 'tot')
    nw = np.where(np.isfinite(null_p), weight_fold, 0.)
    null_terc = S.fold_weighted(MM, null_p, nw) - 1 / 3
    null_terc_adj = S.fold_weighted(MM, null_p - null_b, nw)
    sign = S.ratio(MM, real['agree'], real['agree_den']) - .5
    null_sign_rate = per_fold(null, 'agree', 'agree_den')
    null_sign = S.fold_weighted(MM, null_sign_rate, np.where(np.isfinite(null_sign_rate), weight_fold, 0.)) - .5
    values = dict(rho=rho, null_rho=rho_null, rho_minus_null=rho - rho_null, tercile_excess=terc, tercile_tie_adjusted=terc_adj,
                  null_tercile_excess=null_terc, null_tercile_tie_adjusted=null_terc_adj,
                  tercile_minus_null=terc - null_terc, sign_agreement_excess=sign, null_sign_agreement_excess=null_sign,
                  sign_minus_null=sign - null_sign)
    row = {}
    for key, draws in values.items():
        row[key] = float(draws[0])
        lo, hi, _ = S.interval(draws[1:])
        row[key + '_lo'], row[key + '_hi'] = lo, hi
    row.update(n_rows=int(real['n'].sum()), n_folds=int(np.isfinite(real['rho']).sum()), rho_raw_pooled=real['raw'],
               null_rows=int(null_rows))
    return row

def persistence(R, N, MM):
    rows = []
    random = R[R.in_random.astype(bool)]
    for group, name in PROPERTIES:
        null = fold_arrays(N, name, 'w', None)
        strata = [('pooled', 'all', random, 'w', 'sector_relation')]
        strata += [('relation', r, random[random.sector_relation == r], 'w', 'sector_relation') for r in RELATIONS]
        strata += [('era', e, random[random.era == e], 'w', 'sector_relation') for e in ERAS]
        for pool in POOLS:
            members = R[R[pool].astype(bool)].assign(unit=1., pool=pool)
            strata.append(('pool', pool, members, 'unit', 'pool'))
        for kind, label, frame, weight, tgroup in strata:
            real = fold_arrays(frame, name, weight, tgroup)
            fold_weight = real['tot']
            row = dict(property=name, group=group, stratum_type=kind, stratum=label, **summarise(MM, real, null, len(N), fold_weight))
            if name != 'drift_sign':
                for key in list(row):
                    if key.startswith(('sign_', 'null_sign_')): row[key] = math.nan
            rows.append(row)
        record('persistence_property_completed', persistence_strata=len(strata))
    return pd.DataFrame(rows)

def levels(R, N, M):
    random = R[R.in_random.astype(bool)]
    fold_w = random.groupby('fi').w.sum().reindex(range(len(YEARS)), fill_value=0.).to_numpy()
    null_w = fold_w[N.fi.to_numpy(int)] / 2000.
    rows = []
    for group, name in PROPERTIES:
        for window in 'AB':
            column = f'{name}_{window}'
            real_point = S.weighted_median(random[column], random.w)
            null_point = S.weighted_median(N[column], null_w)
            real_draws = S.bootstrap_weighted_median(M, random[column], random.w, random.fi)
            null_draws = S.bootstrap_weighted_median(M, N[column], null_w, N.fi)
            diff_point = float(S.difference(real_point, null_point))
            lo, hi, finite = S.interval(S.difference(real_draws, null_draws))
            rows.append(dict(property=name, group=group, window=window, real_median=real_point, null_median=null_point,
                             real_minus_null=diff_point, lo=lo, hi=hi, finite_draws=finite,
                             real_rows=int(random[column].notna().sum()), null_rows=int(N[column].notna().sum()),
                             excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))))
    return pd.DataFrame(rows)

def labels(P, L):
    rows = []
    for group, name in PROPERTIES:
        p = P[(P.property == name) & (P.stratum_type == 'pooled')].iloc[0]
        level = L[(L.property == name) & (L.window == 'A')].iloc[0]
        level_b = L[(L.property == name) & (L.window == 'B')].iloc[0]
        informative = (bool(np.isfinite(p.rho_minus_null_lo) and p.rho_minus_null_lo > 0)
                       if group in REVERSION_GROUPS else 'not_applicable')
        rows.append(dict(property=name, group=group,
                         persistent=bool(np.isfinite(p.rho_lo) and p.rho_lo >= 0.20), rho=p.rho, rho_lo=p.rho_lo, rho_hi=p.rho_hi,
                         reversion_informative=informative, rho_minus_null=p.rho_minus_null,
                         rho_minus_null_lo=p.rho_minus_null_lo, rho_minus_null_hi=p.rho_minus_null_hi,
                         different_from_random_walk=bool(level.excludes_zero), level_diff_A=level.real_minus_null,
                         level_lo_A=level.lo, level_hi_A=level.hi, level_diff_B=level_b.real_minus_null,
                         level_lo_B=level_b.lo, level_hi_B=level_b.hi, B_excludes_zero=bool(level_b.excludes_zero),
                         null_persistence_mechanical=group == 'scale'))
    return pd.DataFrame(rows)

# ---- G versus chance ------------------------------------------------------------

def G_vs_null(E, N, M):
    MM = np.vstack([np.ones(len(YEARS)), M])
    null_rate = {m: N.groupby('fi')[m].mean().reindex(range(len(YEARS))).to_numpy(float) for m in METRICS}
    assert (N.groupby('fi').size() == 2000).all()
    rows = []
    for metric in METRICS:
        for era_ in ['all'] + ERAS:
            for relation in ['all'] + RELATIONS:
                part = E
                if era_ != 'all': part = part[part.era == era_]
                if relation != 'all': part = part[part.sector_relation == relation]
                n = part.groupby('fi').size().reindex(range(len(YEARS)), fill_value=0).to_numpy(float)
                k = part.groupby('fi')[metric].sum().reindex(range(len(YEARS)), fill_value=0).to_numpy(float)
                real = S.ratio(MM, k, n)
                null = S.ratio(MM, n * null_rate[metric], n)
                lo, hi, _ = S.interval(real[1:] - null[1:])
                rlo, rhi, _ = S.interval(real[1:]); nlo, nhi, _ = S.interval(null[1:])
                rows.append(dict(metric=metric, era=era_, sector_relation=relation, eligible_pair_folds=int(n.sum()),
                                 real_passes=int(k.sum()), real_rate=float(real[0]), real_lo=rlo, real_hi=rhi,
                                 null_rate=float(null[0]), null_lo=nlo, null_hi=nhi, real_minus_null=float(real[0] - null[0]),
                                 lo=lo, hi=hi, null_pairs=int(2000 * (n > 0).sum()),
                                 excludes_zero=bool(np.isfinite(lo) and (lo > 0 or hi < 0))))
    return pd.DataFrame(rows)

# ---- ticker effects -----------------------------------------------------------

def ticker_effects(E, T, M):
    per_row, summary, sims = [], [], {}
    sector = {}
    for leg, sec in [('leg_A', 'sector_A'), ('leg_B', 'sector_B')]:
        for (y, t, s) in E[['test_year', leg, sec]].drop_duplicates().itertuples(index=False):
            sector[(y, t)] = s
    fold_values = {ind: [] for ind in ['FAST63', 'G_pass']}
    for indicator in ['FAST63', 'G_pass']:
        sim_rows = []
        for f, year in enumerate(YEARS):
            part = E[E.test_year == year]
            passed = part[indicator].to_numpy(bool)
            tickers, ia, ib, n, k = S.ticker_counts(part.leg_A.to_numpy(), part.leg_B.to_numpy(), passed)
            d = S.dispersion(n, k, int(passed.sum()), len(part))
            involved, total, top = S.top_share(tickers, ia, ib, k, passed)
            sim = S.homogeneous_reference(ia, ib, len(tickers), len(part), d['rate'], SIMULATIONS,
                                          np.random.SeedSequence([SEED, year, 7 if indicator == 'FAST63' else 8]))
            sim_rows.append(sim)
            fold_values[indicator].append(dict(X2=d['X2'], T=d['tickers'], var_num=d['var_num'], var_den=d['var_den'],
                                               involved=involved, total=total))
            summary.append(dict(indicator=indicator, test_year=year, era=era(year), pairs=len(part), passes=int(passed.sum()),
                                rate=d['rate'], tickers=d['tickers'], dispersion_D=d['X2'] / d['tickers'],
                                variance_ratio=d['var_num'] / d['var_den'], top5_tickers=int(top.sum()),
                                top5_share=involved / total if total else math.nan, max_degree=int(k.max()),
                                sim_D_mean=float(np.mean(sim[:, 0] / sim[:, 1])), sim_D_p99=float(np.quantile(sim[:, 0] / sim[:, 1], .99)),
                                sim_top5_share_mean=float(np.mean(sim[:, 2] / sim[:, 3])),
                                sim_top5_share_p99=float(np.quantile(sim[:, 2] / sim[:, 3], .99))))
            p = d['rate']
            for i, ticker in enumerate(tickers):
                key = (year, ticker)
                per_row.append(dict(indicator=indicator, test_year=year, era=era(year), ticker=ticker, sector=sector[key],
                                    eligible_partners=int(n[i]), passes=int(k[i]), pass_rate=k[i] / n[i], fold_rate=p,
                                    standardized_residual=(k[i] - n[i] * p) / math.sqrt(n[i] * p * (1 - p)) if 0 < p < 1 else math.nan,
                                    top5=bool(top[i])))
            record('ticker_effects_fold_completed', homogeneous_simulations=SIMULATIONS)
        sims[indicator] = np.stack(sim_rows)  # folds x simulations x 4
    MM = np.vstack([np.ones(len(YEARS)), M])
    pooled = {}
    for indicator in ['FAST63', 'G_pass']:
        v = pd.DataFrame(fold_values[indicator])
        D = S.ratio(MM, v.X2, v['T']); VR = S.ratio(MM, v.var_num, v.var_den); share = S.ratio(MM, v.involved, v.total)
        sim = sims[indicator]
        sim_D = sim[:, :, 0].sum(axis=0) / sim[:, :, 1].sum(axis=0)
        sim_share = sim[:, :, 2].sum(axis=0) / sim[:, :, 3].sum(axis=0)
        Dlo, Dhi, _ = S.interval(D[1:]); Vlo, Vhi, _ = S.interval(VR[1:]); slo, shi, _ = S.interval(share[1:])
        sim_p99 = float(np.quantile(sim_D, .99))
        if Dlo >= 2 and share[0] >= 0.5: label = 'mainly_single_ticker'
        elif Dlo > sim_p99: label = 'partly_ticker_driven'
        else: label = 'not_ticker_driven'
        pooled[indicator] = dict(indicator=indicator, test_year='pooled', era='all', pairs=int(E.shape[0]), passes=int(E[indicator].sum()),
                                 rate=float(E[indicator].mean()), tickers=int(v['T'].sum()), dispersion_D=float(D[0]), D_lo=Dlo, D_hi=Dhi,
                                 variance_ratio=float(VR[0]), variance_ratio_lo=Vlo, variance_ratio_hi=Vhi,
                                 top5_share=float(share[0]), top5_share_lo=slo, top5_share_hi=shi,
                                 max_degree=int(max(s['max_degree'] for s in summary if s['indicator'] == indicator)),
                                 sim_D_mean=float(sim_D.mean()), sim_D_p99=sim_p99, sim_top5_share_mean=float(sim_share.mean()),
                                 sim_top5_share_p99=float(np.quantile(sim_share, .99)), label=label)
    summary = pd.DataFrame(summary + list(pooled.values()))
    rows = pd.DataFrame(per_row)
    wide = rows[rows.indicator == 'FAST63'].drop(columns='indicator').rename(columns={
        'passes': 'FAST_passes', 'pass_rate': 'FAST_rate', 'fold_rate': 'FAST_fold_rate', 'standardized_residual': 'FAST_z', 'top5': 'FAST_top5'})
    g = rows[rows.indicator == 'G_pass'][['test_year', 'ticker', 'passes', 'pass_rate', 'fold_rate', 'standardized_residual', 'top5']].rename(columns={
        'passes': 'G_passes', 'pass_rate': 'G_rate', 'fold_rate': 'G_fold_rate', 'standardized_residual': 'G_z', 'top5': 'G_top5'})
    wide = wide.merge(g, on=['test_year', 'ticker'], how='left', validate='one_to_one')
    st = T.drop(columns=['jump_dates', 'fi'])
    wide = wide.merge(st, on=['test_year', 'ticker'], how='left', validate='one_to_one')
    assert wide.train_vol.notna().all()
    top = []
    jumps = {(r.test_year, r.ticker): r.jump_dates for r in T.itertuples(index=False)}
    for indicator, column in [('FAST63', 'FAST_passes'), ('G_pass', 'G_passes')]:
        agg = []
        for ticker, part in wide.groupby('ticker'):
            latest = part.sort_values('test_year').iloc[-1]
            days = set()
            for y in part.test_year: days.update(jumps[(y, ticker)])
            agg.append(dict(indicator=indicator, ticker=ticker, sector=latest.sector, pooled_degree=int(part[column].sum()),
                            pooled_partners=int(part.eligible_partners.sum()), pass_rate=part[column].sum() / part.eligible_partners.sum(),
                            folds=len(part), max_fold_degree=int(part[column].max()), median_train_vol=float(part.train_vol.median()),
                            max_abs_return=float(part.max_abs_return.max()), distinct_days_abs_return_gt_10pct=len(days),
                            median_ar1_phi_detrended=float(part.ar1_phi_detrended.median())))
        agg = pd.DataFrame(agg).sort_values(['pooled_degree', 'ticker'], ascending=[False, True])
        reference = dict(indicator=indicator, ticker='ALL_TICKERS_MEDIAN', sector='', pooled_degree=float(agg.pooled_degree.median()),
                         pooled_partners=float(agg.pooled_partners.median()), pass_rate=float(agg.pass_rate.median()),
                         folds=float(agg.folds.median()), max_fold_degree=float(agg.max_fold_degree.median()),
                         median_train_vol=float(agg.median_train_vol.median()), max_abs_return=float(agg.max_abs_return.median()),
                         distinct_days_abs_return_gt_10pct=float(agg.distinct_days_abs_return_gt_10pct.median()),
                         median_ar1_phi_detrended=float(agg.median_ar1_phi_detrended.median()))
        head = agg.head(30).assign(rank=range(1, 31))
        top.append(pd.concat([head, pd.DataFrame([dict(reference, rank=0)])], ignore_index=True))
    return wide.sort_values(['test_year', 'ticker']), summary, pd.concat(top, ignore_index=True)

def sample_counts():
    rows = []
    for year in YEARS:
        rows.extend(read(HERE / f'runtime/sample_summary_{year}.json'))
    return pd.DataFrame(rows)

def analyse():
    marker = HERE / 'audit/analysis_ready.json'
    if ready(marker): return read(marker)
    R, N, E, T = load_data()
    M = S.bootstrap_matrix(len(YEARS), 2000, SEED)
    MM = np.vstack([np.ones(len(YEARS)), M])
    tables = {}
    tables['sample_counts.csv'] = sample_counts()
    keep = ['test_year', 'era', 'pair', 'leg_A', 'leg_B', 'sector_A', 'sector_B', 'sector_relation', 'in_random', 'ipw'] + POOLS + \
           ['FAST63', 'G_pass'] + [f'{n}_{w}' for n in NAMES for w in 'AB']
    tables['property_values_sample.csv.gz'] = R[keep]
    tables['property_values_null.csv.gz'] = N[['test_year', 'era', 'simulation', 'FAST63', 'G_pass', 'cond_entries', 'cond_reverted_share',
                                               'cond_price_share', 'cond_net_positive'] + [f'{n}_{w}' for n in NAMES for w in 'AB']]
    P = persistence(R, N, MM); tables['property_persistence.csv'] = P
    L = levels(R, N, M); tables['property_levels_vs_null.csv'] = L
    tables['labels.csv'] = labels(P, L)
    tables['G_vs_null.csv'] = G_vs_null(E, N, M)
    wide, summary, top = ticker_effects(E, T, M)
    tables['ticker_effects.csv'] = wide; tables['ticker_effects_summary.csv'] = summary; tables['top_degree_tickers.csv'] = top
    files = [csv_new(TABLES / name, frame) for name, frame in tables.items()]
    receipt(marker, files, tables=len(files), bootstrap_draws=2000, seed=SEED)
    return read(marker)
