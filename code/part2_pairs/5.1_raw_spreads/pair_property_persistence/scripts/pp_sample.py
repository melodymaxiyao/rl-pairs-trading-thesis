"""Stream the universe inventory once; keep eligible rows; draw the stratified sample.

The inventory is sorted by (test_year, leg_A, leg_B) (asserted by its verifier
and again here), so each fold is complete when a later year appears. Values
used for exact reproduction are kept as the inventory's own cell text.
"""
from ppio import *
import math
import numpy as np
from pp_checkpoints import ready, save_rows, save_object, receipt
from pp_properties import INNER_FIELDS

RELATIONS = ['cross_sector', 'same_sector', 'same_subindustry']
POOLS = ['FAST_cap', 'FAST_X_cap', 'FAST_W_cap']
G_FIELDS = ['G_n_events', 'G_reverted_share', 'G_median_price_share_reverted', 'G_gross_mean_bps', 'G_mean_net_bps']
REPRO = list(INNER_FIELDS) + ['validation_KM_med']
COLUMNS = (['test_year', 'era', 'pair', 'leg_A', 'leg_B', 'sector_A', 'sector_B', 'sector_relation', 'eligible',
            'FAST63', 'G_pass'] + POOLS + G_FIELDS + REPRO)
NULL_FIELDS = ['test_year', 'simulation', 'KM_med', 'n_events', 'n_reverted', 'median_reverted_gross_bps', 'VR_5', 'FAST63',
               'G_pass', 'G_n_events', 'G_reverted_share', 'G_median_price_share_reverted', 'G_gross_mean_bps',
               'inner_KM_med', 'inner_n_reverted', 'inner_FAST63', 'validation_KM_med', 'sigma_A', 'sigma_B']
CAP = 10000

def truth(value):
    assert value in ('True', 'False'), value
    return value == 'True'

def number(value):
    return math.nan if value == '' else float(value)

@lru_cache(None)
def gate_rule():
    return read(note(GATE / 'preregistration.json'))['gate']['primary']

def G_conditions(n_events, reverted_share, price_share, gross_mean):
    rule = gate_rule()
    return dict(cond_entries=bool(n_events >= rule['min_entries']),
                cond_reverted_share=bool(reverted_share >= rule['min_reverted_share']),
                cond_price_share=bool(price_share >= rule['min_median_price_share']),
                cond_net_positive=bool((gross_mean - rule['net_cost_bps']) > rule['mean_net_pnl_strictly_gt']))

def allocate(counts, cap=CAP):
    """Proportional largest-remainder allocation of min(cap, N) over relations."""
    total = sum(counts[r] for r in RELATIONS)
    n = min(cap, total)
    if total == 0:
        return {r: 0 for r in RELATIONS}
    quota = {r: n * counts[r] / total for r in RELATIONS}
    base = {r: int(math.floor(quota[r])) for r in RELATIONS}
    left = n - sum(base.values())
    order = sorted(RELATIONS, key=lambda r: (-(quota[r] - base[r]), RELATIONS.index(r)))
    for r in order[:left]:
        base[r] += 1
    assert sum(base.values()) == n and all(base[r] <= counts[r] for r in RELATIONS)
    return base

def sample_fold(rows, year, seed=SEED):
    """rows: list of eligible dicts for one fold. Returns sampled rows (copies) with flags."""
    by_relation = {r: sorted((x for x in rows if x['sector_relation'] == r), key=lambda x: x['pair']) for r in RELATIONS}
    counts = {r: len(by_relation[r]) for r in RELATIONS}
    quota = allocate(counts)
    chosen = {}
    for index, relation in enumerate(RELATIONS):
        part = by_relation[relation]
        if not quota[relation]:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, year, index]))
        for i in sorted(rng.choice(len(part), quota[relation], replace=False).tolist()):
            row = dict(part[i], in_random=True, ipw=counts[relation] / quota[relation])
            chosen[row['pair']] = row
    for row in rows:
        if any(row[p] for p in POOLS) and row['pair'] not in chosen:
            chosen[row['pair']] = dict(row, in_random=False, ipw=None)
    sampled = sorted(chosen.values(), key=lambda x: x['pair'])
    summary = []
    for relation in RELATIONS:
        part = [x for x in sampled if x['sector_relation'] == relation]
        summary.append(dict(test_year=year, era=era(year), sector_relation=relation, eligible=counts[relation],
                            random_sample=quota[relation], ipw=counts[relation] / quota[relation] if quota[relation] else None,
                            **{p + '_members': sum(x[p] for x in rows if x['sector_relation'] == relation) for p in POOLS},
                            pool_members_not_drawn=sum(not x['in_random'] for x in part), sampled_rows=len(part)))
    return sampled, summary

def finish_fold(year, frames):
    import pandas as pd
    frame = pd.concat(frames, ignore_index=True)
    assert (frame.test_year == str(year)).all()
    keys = list(zip(frame.leg_A, frame.leg_B))
    assert all(a < b for a, b in keys) and keys == sorted(keys) and len(set(keys)) == len(keys)
    rows = []
    for record_ in frame.to_dict('records'):
        row = dict(test_year=year, era=record_['era'], pair=record_['pair'], leg_A=record_['leg_A'], leg_B=record_['leg_B'],
                   sector_A=record_['sector_A'], sector_B=record_['sector_B'], sector_relation=record_['sector_relation'],
                   FAST63=truth(record_['FAST63']), G_pass=truth(record_['G_pass']),
                   **{p: truth(record_[p]) for p in POOLS}, **{f: number(record_[f]) for f in G_FIELDS})
        assert row['era'] == era(year) and row['sector_relation'] in RELATIONS
        row.update(G_conditions(row['G_n_events'], row['G_reverted_share'], row['G_median_price_share_reverted'], row['G_gross_mean_bps']))
        assert all(row[c] for c in ['cond_entries', 'cond_reverted_share', 'cond_price_share', 'cond_net_positive']) == row['G_pass'], \
            ('G conditions do not reproduce inventory G_pass', year, row['pair'])
        row['repro'] = {f: record_[f] for f in REPRO}
        rows.append(row)
    sampled, summary = sample_fold(rows, year)
    eligible = pd.DataFrame([{k: v for k, v in r.items() if k != 'repro'} for r in rows])
    save_object(f'eligible_{year}', eligible, test_year=year, eligible_pair_folds=len(rows))
    save_rows(f'sample_{year}', sampled, test_year=year, sampled_rows=len(sampled), random_rows=sum(r['in_random'] for r in sampled))
    write(HERE / f'runtime/sample_summary_{year}.json', summary)
    record('fold_extracted', test_year=year, eligible_pair_folds=len(rows), sampled_pair_folds=len(sampled), test_rows_used=0)
    print('Extracted fold', year, 'eligible', len(rows), 'sampled', len(sampled), flush=True)

def extract():
    import pandas as pd
    marker = HERE / 'audit/extract_ready.json'
    if ready(marker): return read(marker)
    path = note(UNIV / 'outputs/tables/pair_fold_inventory.csv.gz')
    pending, last, done, total = {}, None, [], 0
    for chunk in pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False, chunksize=200000):
        total += len(chunk)
        years = chunk.test_year.astype(int)
        assert years.is_monotonic_increasing and (last is None or years.iloc[0] >= last)
        for year, part in chunk[chunk.eligible == 'True'].groupby(years[chunk.eligible == 'True']):
            pending.setdefault(int(year), []).append(part)
        last = int(years.iloc[-1])
        for year in sorted(y for y in pending if y < last):
            if not (HERE / f'runtime/checkpoints/sample_{year}.json').exists() or not (HERE / f'runtime/sample_summary_{year}.json').exists():
                finish_fold(year, pending[year])
            done.append(year); del pending[year]
    for year in sorted(pending):
        if not (HERE / f'runtime/checkpoints/sample_{year}.json').exists() or not (HERE / f'runtime/sample_summary_{year}.json').exists():
            finish_fold(year, pending[year])
        done.append(year)
    assert done == YEARS, done
    stored = pd.read_csv(note(UNIV / 'outputs/tables/null_simulation.csv.gz'), usecols=NULL_FIELDS, dtype=str, keep_default_na=False)
    for year in YEARS:
        part = stored[stored.test_year == str(year)].sort_values('simulation', key=lambda s: s.astype(int))
        assert part.simulation.astype(int).tolist() == list(range(2000)), ('Stored null pairs incomplete', year)
        save_rows(f'null_stored_{year}', part.to_dict('records'), test_year=year)
    files = [HERE / f'runtime/sample_summary_{y}.json' for y in YEARS]
    receipt(marker, files, inventory_rows=total, folds=len(done), null_stored=44000)
    return read(marker)
