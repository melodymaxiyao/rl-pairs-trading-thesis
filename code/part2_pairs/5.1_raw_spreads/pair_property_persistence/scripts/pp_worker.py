"""Per-fold train-only context, sampled real pair-folds and regenerated null pairs."""
from ppio import *
import math
import numpy as np
import pandas as pd
from pp_checkpoints import ready, save_rows, load_rows, save_object, load_object
from pp_reader import read_train
from pp_properties import window_properties, ticker_statistics, INNER_FIELDS, NAMES
from pp_imports import universe

G_CONDITIONS = ['condition_entries', 'condition_reverted_share', 'condition_price_share', 'condition_net_positive']
NULL_COMPARE = {'KM_med': ('full', 'KM_med'), 'n_events': ('full', 'n_events'), 'n_reverted': ('full', 'n_reverted'),
                'median_reverted_gross_bps': ('full', 'median_reverted_gross_bps'), 'VR_5': ('full', 'VR_5'),
                'FAST63': ('full', 'FAST63'), 'G_pass': ('full', 'G_pass'), 'G_n_events': ('full', 'G_n_events'),
                'G_reverted_share': ('full', 'G_reverted_share'),
                'G_median_price_share_reverted': ('full', 'G_median_price_share_reverted'),
                'G_gross_mean_bps': ('full', 'G_gross_mean_bps'), 'inner_KM_med': ('A', 'KM_med'),
                'inner_n_reverted': ('A', 'n_reverted'), 'inner_FAST63': ('A', 'FAST63'),
                'validation_KM_med': ('B', 'KM_med')}

def price_path(ticker):
    return note(FRESH / 'data/raw/prices' / (ticker.lower() + '_yahoo.csv'))

def build_context(year, tickers, source=price_path, universe_volatilities=True):
    """Train-only logs for every eligible ticker of fold Y and single-stock statistics."""
    h = set_year(year)
    cal = h.comp.CAL.copy()
    assert cal.max().year < year
    logs, stats = {}, {}
    train = (cal.year >= year - 3) & (cal.year < year)
    for ticker in sorted(tickers):
        path = source(ticker)
        raw, rows = read_train(path, year)
        with np.errstate(divide='ignore', invalid='ignore'):
            logs[ticker] = np.log(raw['Adj Close'].reindex(cal).to_numpy())
        record('train_only_input', test_year=year, price_files_read=1, train_numeric_rows=rows,
               max_numeric_year=int(raw.index.year.max()) if len(raw) else None, test_rows_parsed=0,
               source=str(Path(path).name))
        stats[ticker] = ticker_statistics(h, logs[ticker][train], cal[train])
    if universe_volatilities:
        stored = read(note(UNIV / f'runtime/null_volatilities_{year}.json'))
        assert stored['eligible_tickers'] == sorted(tickers), ('Eligible ticker set differs from universe null volatilities', year)
        mine = [stats[t]['train_vol'] for t in stored['eligible_tickers']]
        assert mine == stored['volatilities'], ('Train volatilities differ from universe', year)
        volatilities = stored['volatilities']
    else:
        volatilities = [stats[t]['train_vol'] for t in sorted(tickers)]
    return dict(year=year, cal=cal, logs=logs, ticker_stats=stats, volatilities=volatilities)

@lru_cache(maxsize=2)
def context(year):
    value = load_object(f'context_{year}')
    assert value['year'] == year and value['cal'].max().year < year
    record('fold_context_load', test_year=year, max_numeric_year=int(value['cal'].max().year), test_rows_parsed=0)
    return value

def pair_windows(h, U, cal, log_a, log_b, year, pair):
    es = U.make_episodes(h, cal, log_a, log_b, year, pair, range(year - 3, year))
    assert [e['year'] for e in es] == [year - 3, year - 2, year - 1]
    a, stats_a = window_properties(h, U, es[:2], True)
    b, stats_b = window_properties(h, U, es[-1:], False)
    record('pair_windows_computed', window_property_sets=2, episode_statistics_calls=2, test_rows_used=0)
    return es, a, stats_a, b, stats_b

def compute_real(ctx, rows):
    h, U, year = set_year(ctx['year']), universe(), ctx['year']
    out, mismatches = [], []
    for row in rows:
        _, a, sa, b, sb = pair_windows(h, U, ctx['cal'], ctx['logs'][row['leg_A']], ctx['logs'][row['leg_B']], year, row['pair'])
        # Repair inf_text_compare_v1: checkpoint loading restores the cell text 'inf' to float inf,
        # so both sides go through the same universe cell formatter before the exact comparison.
        for field, key in INNER_FIELDS.items():
            if text(sa[key]) != text(row['repro'][field]):
                mismatches.append((row['pair'], field, row['repro'][field], text(sa[key])))
        if text(sb['KM_med']) != text(row['repro']['validation_KM_med']):
            mismatches.append((row['pair'], 'validation_KM_med', row['repro']['validation_KM_med'], text(sb['KM_med'])))
        entry = {k: v for k, v in row.items() if k != 'repro'}
        entry.update({n + '_A': a[n] for n in NAMES}); entry.update({n + '_B': b[n] for n in NAMES})
        out.append(entry)
    assert not mismatches, ('Reproduction of inventory inner/validation statistics failed', year, mismatches[:20], len(mismatches))
    return out

def G_decision(h, es):
    events = h.ho.gate_events(es, 3)
    if events.empty:
        # Same empty-frame convention as the universe episode_statistics.
        events = pd.DataFrame(columns=['window', 'anchor', 'construction', 'pair', 'fold', 'ideal_pnl_bps', 'reverted', 'price_share'])
        events['reverted'] = events.reverted.astype(bool)
    decision = h.ho.decide(events, [(es[0]['pair'], es[0]['fold'])]).iloc[0].to_dict()
    record('original_G_decide', original_G_calls=1, test_rows_used=0)
    return decision

def compute_null(ctx, indices, stored):
    h, U, year = set_year(ctx['year']), universe(), ctx['year']
    out, mismatches = [], []
    for index in indices:
        la, lb, sa_, sb_ = U.simulate_logs(len(ctx['cal']), year, index, ctx['volatilities'])
        pair = f'NULL_{index:04d}'
        es, a, stats_a, b, stats_b = pair_windows(h, U, ctx['cal'], la, lb, year, pair)
        full, _, _ = U.episode_statistics(h, es)
        decision = G_decision(h, es)
        assert bool(decision['passes']) == bool(full['G_pass'])
        sources = dict(full=full, A=stats_a, B=stats_b)
        reference = stored[index]
        assert int(reference['simulation']) == index
        for field, (where, key) in NULL_COMPARE.items():
            if text(sources[where][key]) != text(reference[field]):
                mismatches.append((index, field, reference[field], text(sources[where][key])))
        for field, value in [('sigma_A', sa_), ('sigma_B', sb_)]:
            if text(value) != text(reference[field]):
                mismatches.append((index, field, reference[field], text(value)))
        entry = dict(test_year=year, era=era(year), simulation=index, FAST63=bool(full['FAST63']), G_pass=bool(decision['passes']),
                     cond_entries=bool(decision['condition_entries']), cond_reverted_share=bool(decision['condition_reverted_share']),
                     cond_price_share=bool(decision['condition_price_share']), cond_net_positive=bool(decision['condition_net_positive']),
                     G_n_events=decision['n_events'], G_reverted_share=decision['reverted_share'],
                     G_median_price_share_reverted=decision['median_price_share_reverted'], G_gross_mean_bps=decision['gross_mean_bps'])
        entry.update({n + '_A': a[n] for n in NAMES}); entry.update({n + '_B': b[n] for n in NAMES})
        out.append(entry)
        record('null_pair_regenerated', null_pairs=1, test_rows_used=0)
    assert not mismatches, ('Regenerated null pairs do not reproduce stored universe statistics; STOP', year, mismatches[:20], len(mismatches))
    return out

def real_chunk(year, number, rows):
    os.environ['PP_SCOPE'] = 'production_real'
    guard()
    marker = HERE / f'runtime/checkpoints/real_{year}_{number:05d}.json'
    if ready(marker): return str(marker)
    values = compute_real(context(year), rows)
    record('real_chunk_completed', sampled_pair_folds=len(values), reproduced_pair_folds=len(values))
    save_rows(f'real_{year}_{number:05d}', values, test_year=year, chunk=number, kind='real', reproduced=len(values))
    return str(marker)

def null_chunk(year, number, indices):
    os.environ['PP_SCOPE'] = 'production_null'
    guard()
    marker = HERE / f'runtime/checkpoints/null_{year}_{number:05d}.json'
    if ready(marker): return str(marker)
    stored = {int(r['simulation']): r for r in load_rows(HERE / f'runtime/checkpoints/null_stored_{year}.json')}
    values = compute_null(context(year), indices, stored)
    record('null_chunk_completed', null_pairs_completed=len(values), null_pairs_reproduced=len(values))
    save_rows(f'null_{year}_{number:05d}', values, test_year=year, chunk=number, kind='null', reproduced=len(values))
    return str(marker)
