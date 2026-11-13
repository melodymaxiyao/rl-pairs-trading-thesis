"""Governing 2-sigma, non-overlapping events, closed or censored each year."""
import numpy as np
from screen import km_quantiles

def positions(z, anchor=2.0):
    z = np.asarray(z, float)
    assert np.isfinite(z).all()
    opened = None
    direction = None
    for t in range(1, len(z)):
        if opened is not None:
            if z[t] == 0 or np.sign(z[t]) != direction:
                yield opened, t, True
                opened = None
                direction = None
            continue
        if abs(z[t]) >= anchor and abs(z[t - 1]) < anchor:
            opened, direction = t, np.sign(z[t])
    if opened is not None:
        yield opened, len(z) - 1, False

def screen_events(episodes, interval, anchor=2.0):
    result = []
    for episode in sorted(episodes, key=lambda e: e['year']):
        assert episode['window'] == 'train'
        assert all(d.year == episode['year'] for d in episode['dates'])
        for start, end, reverted in positions(episode['z'], anchor):
            # interval is the unchanged original function, accepting the exact
            # entry/exit positions. Its original F gross and price share apply.
            event = interval(episode, start, end, anchor, 'entry', reverted=reverted)
            assert event['entry_date'].year == event['end_date'].year
            result.append(event)
    return result

def summarize(events):
    median, p90 = km_quantiles([e['duration'] for e in events], [e['reverted'] for e in events])
    reverted = [e for e in events if e['reverted']]
    return dict(KM_med=median, KM_p90=p90, n_events=len(events), n_reverted=len(reverted),
                reverted_share=len(reverted) / len(events) if events else np.nan,
                median_reverted_gross_bps=float(np.median([e['ideal_pnl_bps'] for e in reverted])) if reverted else np.nan,
                events_by_year={str(y): sum(e['entry_date'].year == y for e in events)
                                for y in sorted({e['entry_date'].year for e in events})})

def legacy_removals(legacy, nonoverlap, year_starts=()):
    current = {e['entry_date']: e for e in nonoverlap}
    counts = dict(legacy_events=len(legacy), legacy_removed_overlap=0,
                  legacy_removed_year_end=0, legacy_removed_year_start=0,
                  legacy_removed_total=0, legacy_year_end_crossings=0)
    for event in legacy:
        date = event['entry_date']
        crossed = date.year != event['end_date'].year
        counts['legacy_year_end_crossings'] += int(crossed)
        new = current.get(date)
        if new is None:
            # A legacy entry at a year's first session can have a warmup
            # predecessor. Other missing entries are suppressed overlaps.
            counts['legacy_removed_year_start' if date in year_starts else 'legacy_removed_overlap'] += 1
            counts['legacy_removed_total'] += 1
        elif crossed:
            counts['legacy_removed_year_end'] += 1
            counts['legacy_removed_total'] += 1
    return counts
