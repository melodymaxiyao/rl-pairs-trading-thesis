"""Frozen FAST conditions, right-censored Kaplan–Meier, caps and groups.

Event extraction, gross and price share are supplied by unchanged imported
functions. This module never reads prices, evaluates trades or fits learners.
"""
import math
from collections import Counter
import numpy as np

def km_quantiles(durations, reverted):
    durations = np.asarray(durations, dtype=int)
    reverted = np.asarray(reverted, dtype=bool)
    assert len(durations) == len(reverted)
    assert np.all(durations >= 0)
    survival, median, p90 = 1.0, math.inf, math.inf
    for duration in np.unique(durations):
        # Reversions precede censor removals for observations tied in time.
        risk = int(np.sum(durations >= duration))
        deaths = int(np.sum((durations == duration) & reverted))
        survival *= 1 - deaths / risk
        if survival <= 0.5 + 1e-12 and math.isinf(median):
            median = float(duration)
        if survival <= 0.1 + 1e-12 and math.isinf(p90):
            p90 = float(duration)
    return median, p90

def conditions(km_med, n_reverted, median_gross_bps, vr5, inner=False, threshold=63):
    return dict(condition_speed=bool(km_med <= threshold),
                condition_evidence=bool(n_reverted >= (3 if inner else 4)),
                condition_gross=bool(median_gross_bps >= 20),
                condition_VR5=bool(vr5 >= 0.3))

def passes(km_med, n_reverted, median_gross_bps, vr5, inner=False, threshold=63):
    return all(conditions(km_med, n_reverted, median_gross_bps, vr5, inner, threshold).values())

def greedy_cap(rows, inner=False):
    prefix = 'inner_' if inner else ''
    ordered = sorted(rows, key=lambda r: (r[prefix + 'KM_med'],
                                          -r[prefix + 'n_reverted'],
                                          -r[prefix + 'median_reverted_gross_bps'],
                                          r['pair']))
    degrees, chosen = Counter(), []
    for row in ordered:
        a, b = row['leg_A'], row['leg_B']
        if degrees[a] < 2 and degrees[b] < 2:
            chosen.append(row)
            degrees[a] += 1
            degrees[b] += 1
    assert max(degrees.values(), default=0) <= 2
    return chosen

def groups(rows, inner=False):
    if not rows:
        return {}
    prefix = 'inner_' if inner else ''
    speed = float(np.median([r[prefix + 'KM_med'] for r in rows]))
    size = float(np.median([r[prefix + 'median_reverted_gross_bps'] for r in rows]))
    return {r['pair']: ('fast' if r[prefix + 'KM_med'] <= speed else 'slow') + '_' +
            ('large' if r[prefix + 'median_reverted_gross_bps'] >= size else 'small')
            for r in rows}

def assign_pools(rows, inner=False):
    prefix = 'inner_' if inner else ''
    passed = [r for r in rows if r.get('inner_eligible' if inner else 'eligible', False) and r[prefix + 'FAST63']]
    members = {prefix + 'FAST': passed,
               prefix + 'FAST_cap': greedy_cap(passed, inner),
               prefix + 'FAST_X_cap': greedy_cap([r for r in passed if r['sector_relation'] == 'cross_sector'], inner),
               prefix + 'FAST_W_cap': greedy_cap([r for r in passed if r['sector_relation'] != 'cross_sector'], inner)}
    for name, pool in members.items():
        chosen = {r['pair'] for r in pool}
        identifiers = groups(pool, inner) if name.endswith('_cap') else {}
        for row in rows:
            row[name] = row['pair'] in chosen
            if name.endswith('_cap'):
                row[name + '_group_id'] = identifiers.get(row['pair'], '')
    return members
