"""PIT stock candidates, alphabetical unordered pairs and historical overlaps."""
from universe_io import *
import csv
import itertools

def csv_rows(path):
    with note(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def members_at(membership, year):
    month = f'{year - 1}-12-31'
    assert month in membership, ('Missing last pre-year month-end', year, month)
    return membership[month]

@lru_cache(None)
def original_candidates():
    with aliases(sector_io=facade('sector_io', HERE=SECTOR)):
        return load(SECTOR / 'scripts/candidates.py', '_fast_original_candidate_helpers')

@lru_cache(None)
def metadata():
    classification = {r['ticker']: r for r in csv_rows(FRESH / 'classification.csv')}
    mapping = {r['ticker']: r for r in csv_rows(FRESH / 'ticker_map.csv')}
    membership = {}
    for row in csv_rows(FRESH / 'membership_monthly.csv'):
        membership.setdefault(row['month_end'], []).append(row['ticker'])
    return classification, mapping, membership

def unordered(a, b):
    return tuple(sorted((a, b)))

@lru_cache(None)
def historical_overlap():
    expansion = read(note(SECTOR / 'outputs/tables/pair_fold_inventory.json'))
    all_exp, g_exp = set(), set()
    for row in expansion:
        key = row['test_year'], unordered(row['anchor'], row['candidate'])
        if row['EXP_ALL']:
            all_exp.add(key)
        if row['EXP_G']:
            g_exp.add(key)
    selected = read(note(NOISE / 'outputs/tables/selected_sources.json'))
    old_all = set()
    old68 = set()
    # The input's fixed 68-pair definitions are the same unchanged v4 pair table
    # used by learner_noise; selected_sources supplies fold-specific old ALL.
    for row in csv_rows(UNI / 'outputs/tables/pairs.csv'):
        old68.add(unordered(row['anchor'], row['candidate']))
    assert len(old68) == 68
    for row in selected:
        a, b = row['pair'].split('_')
        old_all.add((row['test_year'], unordered(a, b)))
    return all_exp, g_exp, old68, old_all

def stocks_for_year(year):
    classification, mapping, membership = metadata()
    helpers = original_candidates()
    included, missing = [], []
    for ticker in sorted(members_at(membership, year)):
        assert ticker in mapping, ('Missing ticker mapping', ticker)
        c = classification.get(ticker, {})
        symbol = mapping[ticker]['yahoo_symbol']
        kind = helpers.kind(c)
        stored = (FRESH / 'data/raw/prices' / (symbol.lower() + '_yahoo.csv')).is_file()
        row = dict(ticker=ticker, symbol=symbol, sector=c.get('sector', ''),
                   subindustry=c.get('industry', ''), security=c.get('security', ''),
                   classification_group=kind, stored_price=stored,
                   membership_month=f'{year - 1}-12-31', test_year=year, era=era(year))
        if kind == 'explicit_GICS' and stored:
            included.append(row)
        else:
            reasons = ([] if kind == 'explicit_GICS' else [kind]) + ([] if stored else ['missing_prices'])
            missing.append(dict(row, exclusion_reason=';'.join(reasons)))
    assert len({r['symbol'] for r in included}) == len(included), 'Ambiguous symbol mapping'
    return sorted(included, key=lambda r: r['symbol']), missing

def candidate_pairs(year):
    stocks, _ = stocks_for_year(year)
    all_exp, g_exp, old68, old_all = historical_overlap()
    helpers = original_candidates()
    for a, b in itertools.combinations(stocks, 2):
        leg_a, leg_b = a['symbol'], b['symbol']
        assert leg_a < leg_b
        key = year, (leg_a, leg_b)
        relation = ('cross_sector' if a['sector'] != b['sector'] else
                    'same_subindustry' if a['subindustry'] == b['subindustry'] else 'same_sector')
        yield dict(pair=leg_a + '_' + leg_b, leg_A=leg_a, leg_B=leg_b,
                   ticker_A=a['ticker'], ticker_B=b['ticker'],
                   sector_A=a['sector'], sector_B=b['sector'],
                   subindustry_A=a['subindustry'], subindustry_B=b['subindustry'],
                   sector_relation=relation, dual_class=helpers.dual_class(leg_a, leg_b, a, b),
                   test_year=year, fold=year - 2003, era=era(year),
                   membership_month=a['membership_month'], in_EXP_ALL=key in all_exp,
                   in_EXP_G=key in g_exp, in_old68=(leg_a, leg_b) in old68,
                   previously_traded=key in all_exp or key in old_all,
                   source='fresh_sealed_2000_2025')
