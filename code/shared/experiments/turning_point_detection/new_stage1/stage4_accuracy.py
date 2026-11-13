"""Additive fixed/adaptive accuracy evaluation, using the existing v2 contracts.

Only TRAIN combined scores are needed by the unchanged selector. TEST scores
are computed after selection. Bilateral slopes are cached across folds of the
same path; this avoids redundant work without changing the mathematical rule.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from experiments.turning_point_detection.new_stage1.kernel_smoothing import (
    causal_local_linear_slope, bilateral_local_linear_slope,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy import (
    detect_hysteresis_events, reference_events, greedy_match_events, stitch_scores, score_events,
)
from experiments.turning_point_detection.new_stage1.run_slope_accuracy import (
    select_fold, macro_score, scalar_score, add_stitched, aggregate_selected,
)
from experiments.turning_point_detection.new_stage1.slope_accuracy_diagnostics import midrank_percentile
from experiments.turning_point_detection.new_stage1.slope_accuracy_vol_diagnostics import volatility_rows


def return_volatility(prices, *, train_mask, L, ddof):
    if int(L) != L or L <= ddof or ddof < 0:
        raise ValueError('Integral L > ddof >= 0 required')
    returns = prices.diff()
    return {'train': float(returns.loc[train_mask].std(ddof=ddof)),
            'rolling': returns.rolling(int(L), min_periods=int(L)).std(ddof=ddof)}


def detect_adaptive_events(slopes, *, tau, d):
    """Each of d comparisons uses its own day's threshold; NaNs keep state."""
    slopes, tau = np.asarray(slopes, float), np.asarray(tau, float)
    if slopes.shape != tau.shape or slopes.ndim != 1 or int(d) != d or d < 1:
        raise ValueError('Equal-length vectors and positive integral d required')
    if np.any(tau[np.isfinite(tau)] < 0):
        raise ValueError('Thresholds must be nonnegative')
    signs = np.zeros(len(slopes), int)
    valid = np.isfinite(slopes) & np.isfinite(tau)
    signs[valid & (slopes > tau)] = 1
    signs[valid & (slopes < -tau)] = -1
    if len(signs) < d:
        return np.empty((0, 2), int)
    windows = np.lib.stride_tricks.sliding_window_view(signs, int(d))
    positions = np.flatnonzero((windows[:, 0] != 0) & np.all(windows == windows[:, :1], axis=1)) + int(d)-1
    if not len(positions):
        return np.empty((0, 2), int)
    candidate = signs[positions]
    changes = np.r_[False, candidate[1:] != candidate[:-1]]
    return np.column_stack((positions[changes], candidate[changes])).astype(int)


def reference_cache(prices, config):
    slopes = {h: bilateral_local_linear_slope(prices, bandwidth_h=h,
              radius_multiplier=config['reference_radius_multiplier']).to_numpy()
              for h in config['h_ref_grid']}
    return {'slopes': slopes, 'events': {h: reference_events(b) for h, b in slopes.items()}}


def combined_score(detected, reference, config, h_ref):
    matches = greedy_match_events(detected, reference, l_pre=config['match_l_pre'],
                                 max_delay=int(config['match_m_multiplier'] * h_ref))
    return stitch_scores([{'n_ref': len(reference), 'n_det': len(detected), 'n_match': len(matches),
                          'delays': (matches[:, 0]-matches[:, 1]).tolist()}],
                         quantiles=config['delay_quantiles'])


def evaluate_variant_fold(prices, fold, config, *, variant, L, cached_reference=None):
    if variant not in ('fixed', 'adaptive'):
        raise ValueError(variant)
    index = prices.index
    start = int(index.searchsorted(pd.Timestamp(fold['train_start'])))
    stop = int(index.searchsorted(pd.Timestamp(fold['test_end']), side='right'))
    local = prices.iloc[start:stop]
    train = local.index <= pd.Timestamp(fold['train_end'])
    vol = return_volatility(local, train_mask=train, L=L, ddof=config['std_ddof'])
    if not np.isfinite(vol['train']):
        raise ValueError('Insufficient TRAIN returns')
    result = dict(slopes={}, sigma_b={}, events={}, references={}, reference_slopes={},
                  train_scores={}, test_scores={}, bounds={}, tau={}, sigma_r_train=vol['train'])
    full_vol = np.full(len(prices), np.nan)
    full_vol[start:stop] = vol['rolling'].to_numpy()
    result['sigma_r_rolling'] = full_vol
    if vol['train'] > 0:
        ratio = vol['rolling'].to_numpy() / vol['train']
    else:
        # Zero/zero has no volatility adjustment. A positive rolling SD with
        # zero calibrated SD is undefined and cannot confirm a turn.
        ratio = np.where(vol['rolling'].to_numpy() == 0, 1., np.nan)
    for h in config['h_grid']:
        b = causal_local_linear_slope(local, bandwidth_h=h, window_multiplier=config['window_multiplier'])
        sigma = float(b.loc[train].std(ddof=config['std_ddof']))
        if not np.isfinite(sigma):
            raise ValueError('Insufficient TRAIN slopes')
        full = np.full(len(prices), np.nan); full[start:stop] = b.to_numpy()
        result['slopes'][h], result['sigma_b'][h] = full, sigma
        for k in config['k_grid']:
            threshold = k*sigma*ratio if variant == 'adaptive' else np.full(len(local), k*sigma)
            full_tau = np.full(len(prices), np.nan); full_tau[start:stop] = threshold
            for d in config['d_grid']:
                events = (detect_adaptive_events(b.to_numpy(), tau=threshold, d=d) if variant == 'adaptive'
                          else detect_hysteresis_events(b.to_numpy(), tau=k*sigma, d=d))
                events[:, 0] += start
                result['events'][(h, d, k)] = events
                result['tau'][(h, d, k)] = full_tau
    cache = cached_reference if cached_reference is not None else reference_cache(prices, config)
    for h_ref in config['h_ref_grid']:
        radius = int(config['reference_radius_multiplier'] * h_ref)
        result['reference_slopes'][h_ref] = cache['slopes'][h_ref]
        for segment in ('train', 'test'):
            begin = int(index.searchsorted(pd.Timestamp(fold[f'{segment}_start'])))
            last = int(index.searchsorted(pd.Timestamp(fold[f'{segment}_end']), side='right')-1)
            end = last-radius if segment == 'train' else min(last, len(prices)-1-radius)
            if end < begin:
                raise ValueError('Empty evaluation segment')
            result['bounds'][(h_ref, segment)] = (begin, end)
            refs = cache['events'][h_ref]
            refs = refs[(refs[:, 0] >= begin) & (refs[:, 0] <= end)]
            result['references'][(h_ref, segment)] = refs
            if segment == 'train':
                for params, events in result['events'].items():
                    det = events[(events[:, 0] >= begin) & (events[:, 0] <= end)]
                    result['train_scores'][(h_ref, *params, 'combined')] = combined_score(det, refs, config, h_ref)
    return result


def evaluate_variant(price_map, calendar, config, *, variant, L, source='real', repeat=-1,
                     collect_train=False):
    stitched, choices, vol_rows, train_rows = {}, [], [], []
    cache = {ticker: reference_cache(p, config) for ticker, p in price_map.items() if isinstance(ticker, str)}
    for fold_id, group in calendar.groupby('fold_id', sort=True):
        folds = {row['ticker']: row for row in group.to_dict('records')}
        current = {ticker: price_map.get((fold_id, ticker), price_map.get(ticker)) for ticker in folds}
        results = {ticker: evaluate_variant_fold(p, folds[ticker], config, variant=variant, L=L,
                   cached_reference=cache.get(ticker)) for ticker, p in current.items()}
        local_choices = []
        for h_ref in config['h_ref_grid']:
            params, chosen = select_fold(results, config, h_ref)
            for ticker, result in results.items():
                begin, end = result['bounds'][(h_ref, 'test')]
                events = result['events'][params]
                det = events[(events[:, 0] >= begin) & (events[:, 0] <= end)]
                scores = score_events(det, result['references'][(h_ref, 'test')],
                       l_pre=config['match_l_pre'], max_delay=int(config['match_m_multiplier']*h_ref),
                       quantiles=config['delay_quantiles'])
                for kind, score in scores.items():
                    result['test_scores'][(h_ref, *params, kind)] = score
                add_stitched(stitched, ticker, fold_id, result, h_ref, params)
            test = macro_score([r['test_scores'][(h_ref, *params, 'combined')] for r in results.values()], config['delay_quantiles'])
            row = dict(source=source, repeat=repeat, variant=variant, L=L, h_ref=h_ref, fold_id=fold_id,
                       h=params[0], d=params[1], k=params[2], train_f1=chosen['f1'], test_f1=test['f1'],
                       train_minus_test_f1=chosen['f1']-test['f1'], train_delay_median=chosen['delay_median'],
                       test_delay_median=test['delay_median'], n_tickers=len(results), selection_tickers=';'.join(results))
            choices.append(row); local_choices.append(row)
            if collect_train:
                for h, d, k in itertools.product(config['h_grid'], config['d_grid'], config['k_grid']):
                    score = macro_score([r['train_scores'][(h_ref, h, d, k, 'combined')] for r in results.values()], config['delay_quantiles'])
                    train_rows.append(dict(variant=variant, L=L, fold_id=fold_id, h_ref=h_ref, h=h, d=d, k=k,
                                           **scalar_score(score), n_tickers=len(results), usage_note='TRAIN selection only'))
        rows = volatility_rows(current, group, {'selected': pd.DataFrame(local_choices), 'results': {fold_id: results}}, config, source, repeat)
        for row in rows:
            row.update(variant=variant, L=L, sigma_r_train=results[row['ticker']]['sigma_r_train'])
            if variant == 'adaptive':
                row['diagnostic_note'] += '; adaptive使用逐日收益波动缩放，effective_k仅为固定阈值的对照描述'
        vol_rows.extend(rows)
    summary = aggregate_selected(stitched, config).assign(source=source, repeat=repeat, variant=variant, L=L)
    return {'summary': summary, 'selected': pd.DataFrame(choices), 'vol': pd.DataFrame(vol_rows),
            'train_grid': pd.DataFrame(train_rows)}


def empirical_two_sided_p(null, real):
    values = np.asarray(null, float)
    if not np.isfinite(real) or not len(values) or not np.isfinite(values).all():
        return np.nan
    denominator = len(values)+1
    return float(min(1., 2*min((1+np.sum(values <= real))/denominator,
                              (1+np.sum(values >= real))/denominator)))


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm requires finite probabilities for the complete family')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    output = np.empty(len(p)); output[order] = adjusted
    return output


def null_comparison(samples, real, *, expected_repetitions):
    """F1 hypotheses only. Separate macro families for B1 and B2; B0 descriptive."""
    rows = []
    for source, R in expected_repetitions.items():
        for actual in real.to_dict('records'):
            group = samples.loc[(samples.source == source) & (samples.h_ref == actual['h_ref'])
                         & (samples.ticker == actual['ticker']) & (samples.type == actual['type'])]
            if len(group) != R or group.repeat.nunique() != R:
                raise AssertionError(f'Missing/duplicated null repetitions: {source}, {actual}')
            row = dict(source=source, h_ref=actual['h_ref'], ticker=actual['ticker'], type=actual['type'], R=R,
                       real_f1=actual['f1'], real_delay_median=actual['delay_median'], p_raw=np.nan, p_holm=np.nan,
                       family_size=0, test_role='descriptive' if source == 'B0' else ('primary' if source == 'B1' else 'secondary'))
            for field, actual_field in (('f1', 'f1'), ('delay', 'delay_median')):
                values = group[actual_field].dropna().to_numpy()
                q = np.quantile(values, [.05, .5, .95]) if len(values) else [np.nan]*3
                for suffix, value in zip(('q05', 'median', 'q95'), q):
                    row[f'null_{field}_{suffix}'] = value
                row[f'n_valid_{field}'] = len(values)
                row[f'real_{field}_null_percentile'] = midrank_percentile(values, actual[actual_field])
            row['real_f1_minus_null_median'] = actual['f1']-row['null_f1_median']
            if actual['ticker'] == 'equal_ticker_macro' and source in ('B1', 'B2'):
                row['p_raw'] = empirical_two_sided_p(group.f1, actual['f1'])
            rows.append(row)
    frame = pd.DataFrame(rows)
    for source in ('B1', 'B2'):
        mask = frame.source.eq(source) & frame.ticker.eq('equal_ticker_macro')
        if mask.any():
            frame.loc[mask, 'family_size'] = int(mask.sum())
            frame.loc[mask, 'p_holm'] = holm_adjust(frame.loc[mask, 'p_raw'])
    return frame


def vol_relationship_statistics(frame, *, low, high):
    """Equal ticker-fold points, combined only; difference is LOW minus CENTRAL."""
    rows = []
    for h_ref, group in frame.loc[frame.type == 'combined'].groupby('h_ref', sort=True):
        for metric in ('recall', 'f1'):
            valid = group.vol_ratio.gt(0) & np.isfinite(group.vol_ratio) & np.isfinite(group[metric])
            x, y = np.log(group.loc[valid, 'vol_ratio']), group.loc[valid, metric]
            value = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and x.std() > 0 and y.std() > 0 else np.nan
            rows.append(dict(h_ref=h_ref, statistic=f'corr_{metric}_log_vol_ratio', value=value,
                             n_points=len(x), n_low=0, n_central=0))
        valid = np.isfinite(group.vol_ratio) & np.isfinite(group.f1)
        a = group.loc[valid & group.vol_ratio.lt(low), 'f1']
        b = group.loc[valid & group.vol_ratio.ge(low) & group.vol_ratio.le(high), 'f1']
        rows.append(dict(h_ref=h_ref, statistic='f1_mean_low_minus_central',
                         value=float(a.mean()-b.mean()) if len(a) and len(b) else np.nan,
                         n_points=int(valid.sum()), n_low=len(a), n_central=len(b)))
    return pd.DataFrame(rows)


def vol_relationship_comparison(samples):
    rows = []
    for actual in samples.loc[samples.source.eq('real')].to_dict('records'):
        group = samples.loc[samples.source.eq('B1') & samples.h_ref.eq(actual['h_ref']) & samples.statistic.eq(actual['statistic'])]
        values = group.value.dropna().to_numpy()
        q = np.quantile(values, [.05, .5, .95]) if len(values) else [np.nan]*3
        rows.append(dict(h_ref=actual['h_ref'], statistic=actual['statistic'], real_value=actual['value'],
                         b1_q05=q[0], b1_median=q[1], b1_q95=q[2], real_b1_percentile=midrank_percentile(values, actual['value']),
                         R=group.repeat.nunique(), n_valid=len(values), real_n_points=actual['n_points'],
                         real_n_low=actual['n_low'], real_n_central=actual['n_central']))
    return pd.DataFrame(rows)


def variant_comparison(fixed, adaptive):
    keys = ['h_ref', 'ticker', 'type']
    fields = ['precision', 'recall', 'f1', 'delay_median', 'n_det']
    frame = fixed[keys+fields].merge(adaptive[keys+fields], on=keys, validate='one_to_one', suffixes=('_fixed', '_adaptive'))
    if len(frame) != len(fixed) or len(frame) != len(adaptive):
        raise AssertionError('Variant scopes differ')
    for field in fields:
        frame[f'{field}_adaptive_minus_fixed'] = frame[f'{field}_adaptive']-frame[f'{field}_fixed']
    return frame
