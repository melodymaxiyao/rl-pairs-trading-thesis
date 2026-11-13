"""Window properties from the unchanged universe statistics plus small additions.

`episode_statistics` (universe numeric.py, unchanged) supplies the 2.0 and 1.5
non-overlapping within-year events, KM medians, VR(5)/VR(10), AR(1), quarterly
beta CV, drift, leg correlation and the |z|>=1.5 share. This module adds only
the properties that function does not return: within-year zero crossings, the
|z|>=2 share, spread-return sd, window OLS beta, entries per year and the 1.5
reverted share. No property reads anything outside the episodes passed in.
"""
import math
import numpy as np

PROPERTIES = [
    ('scale', 'spread_sd_bps'), ('scale', 'share_absz_ge_2'), ('scale', 'share_absz_ge_1_5'),
    ('comovement', 'leg_return_corr'), ('comovement', 'ols_beta_window'), ('comovement', 'quarterly_beta_cv'),
    ('reversion', 'ar1_phi'), ('reversion', 'half_life'), ('reversion', 'VR_5'), ('reversion', 'VR_10'),
    ('reversion', 'zero_crossings_per_year'),
    ('events_2_0', 'entries_per_year_2_0'), ('events_2_0', 'reverted_share_2_0'), ('events_2_0', 'KM_med_2_0'),
    ('events_2_0', 'median_reverted_gross_2_0'),
    ('events_1_5', 'entries_per_year_1_5'), ('events_1_5', 'reverted_share_1_5'), ('events_1_5', 'KM_med_1_5'),
    ('drift', 'drift_bps_per_year'), ('drift', 'drift_t'), ('drift', 'drift_sign'),
]
NAMES = [name for _, name in PROPERTIES]
GROUP = dict((name, group) for group, name in PROPERTIES)
REVERSION_GROUPS = ('reversion', 'events_2_0', 'events_1_5')

# Inventory inner_* column -> episode_statistics key, for window-A reproduction.
INNER_FIELDS = {
    'inner_KM_med': 'KM_med', 'inner_n_events': 'n_events', 'inner_n_reverted': 'n_reverted',
    'inner_reverted_share': 'reverted_share', 'inner_median_reverted_gross_bps': 'median_reverted_gross_bps',
    'inner_KM_med_a15': 'KM_med_a15', 'inner_n_events_a15': 'n_events_a15', 'inner_n_reverted_a15': 'n_reverted_a15',
    'inner_VR_5': 'VR_5', 'inner_VR_10': 'VR_10', 'inner_AR1_phi': 'AR1_phi', 'inner_half_life': 'half_life',
    'inner_quarterly_beta_cv': 'quarterly_beta_cv', 'inner_train_drift_bps_per_year': 'train_drift_bps_per_year',
    'inner_train_drift_t': 'train_drift_t', 'inner_train_drift_sign': 'train_drift_sign',
    'inner_daily_return_correlation': 'daily_return_correlation', 'inner_abs_z_ge_1_5_share': 'abs_z_ge_1_5_share',
    'inner_FAST63': 'FAST63',
}

def zero_crossings_per_year(episodes):
    """Sign changes of the non-zero z sequence inside each calendar year, per year."""
    total = 0
    for episode in episodes:
        assert all(d.year == episode['year'] for d in episode['dates'])
        signs = np.sign(np.asarray(episode['z'], float))
        signs = signs[signs != 0]
        total += int(np.sum(signs[1:] != signs[:-1]))
    return total / len(episodes)

def half_life(phi, original):
    if not math.isfinite(phi):
        return math.nan
    if 0 < phi < 1:
        return float(original)
    return math.inf if phi >= 1 else 0.0

def spread_returns(episodes):
    # Each year's last forward return is the universe's placeholder 0: dropping it
    # keeps only within-year returns, so no annual F reset jump is included.
    return np.concatenate([e['forward_bps'][:-1] for e in episodes])

def window_properties(h, U, episodes, inner):
    stats, _, _ = U.episode_statistics(h, episodes, inner)
    years = len(episodes)
    z = np.concatenate([e['z'] for e in episodes])
    daily = spread_returns(episodes)
    la = np.concatenate([e['log_a'] for e in episodes])
    lb = np.concatenate([e['log_b'] for e in episodes])
    n15, r15 = stats['n_events_a15'], stats['n_reverted_a15']
    values = dict(
        spread_sd_bps=float(daily.std(ddof=1)),
        share_absz_ge_2=float(np.mean(np.abs(z) >= 2)),
        share_absz_ge_1_5=stats['abs_z_ge_1_5_share'],
        leg_return_corr=stats['daily_return_correlation'],
        ols_beta_window=float(h.comp.ols(la, lb)[1]),
        quarterly_beta_cv=stats['quarterly_beta_cv'],
        ar1_phi=stats['AR1_phi'],
        half_life=half_life(stats['AR1_phi'], stats['half_life']),
        VR_5=stats['VR_5'], VR_10=stats['VR_10'],
        zero_crossings_per_year=zero_crossings_per_year(episodes),
        entries_per_year_2_0=stats['n_events'] / years,
        reverted_share_2_0=stats['reverted_share'],
        KM_med_2_0=stats['KM_med'],
        median_reverted_gross_2_0=stats['median_reverted_gross_bps'],
        entries_per_year_1_5=n15 / years,
        reverted_share_1_5=r15 / n15 if n15 else math.nan,
        KM_med_1_5=stats['KM_med_a15'],
        drift_bps_per_year=stats['train_drift_bps_per_year'],
        drift_t=stats['train_drift_t'],
        drift_sign=stats['train_drift_sign'])
    assert list(values) == NAMES
    return {k: float(v) for k, v in values.items()}, stats

def ticker_statistics(h, log_price, dates):
    """Single-stock train statistics over Y-3..Y-1 score sessions."""
    log_price = np.asarray(log_price, float)
    assert np.isfinite(log_price).all()
    r = np.diff(log_price)
    simple = np.abs(np.expm1(r))
    t = np.arange(len(log_price), dtype=float)
    _, _, residual = h.comp.ols(log_price, t)
    x, y = residual[:-1], residual[1:]
    phi = float(np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0][1])
    jump_dates = [str(d.date()) for d in dates[1:][simple > 0.10]]
    return dict(train_vol=float(np.std(r, ddof=1)), max_abs_return=float(simple.max()),
                days_abs_return_gt_10pct=len(jump_dates), jump_dates=jump_dates,
                ar1_phi_detrended=phi, sessions=len(log_price))
