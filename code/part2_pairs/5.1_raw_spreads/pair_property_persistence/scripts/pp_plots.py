"""Four PNG figures from the finished tables (no new statistics except display ranks)."""
from ppio import *
import math
import numpy as np
import pandas as pd
from pp_checkpoints import ready, receipt
from pp_properties import PROPERTIES

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
REAL, NULL = '#2a78d6', '#eb6834'   # validated categorical slots 1-2 (light)
BLUES = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SCATTER = [('KM_med_2_0', 'KM median at 2.0'), ('ar1_phi', 'AR(1) φ'), ('zero_crossings_per_year', 'Zero crossings per year'),
           ('spread_sd_bps', 'Spread-return sd'), ('leg_return_corr', 'Leg return correlation')]

def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
                         'axes.titleweight': 'bold', 'legend.frameon': False})
    return plt

def save(fig, name):
    path = inside(PLOTS / name)
    pending = path.with_name(path.name + '.pending_' + str(time.time_ns()) + '.png')
    fig.savefig(pending, dpi=150, bbox_inches='tight', metadata={'Software': None})
    if path.exists():
        pending.unlink()   # PNG bytes are not compared; the receipt fixes the first render
    else:
        os.link(pending, path); pending.unlink()
    record('PNG_generated', PNG_files_written=1)
    return path

def spearman_plot(plt, P):
    pooled = P[P.stratum_type == 'pooled'].set_index('property')
    names = [n for _, n in PROPERTIES]
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y = np.arange(len(names))[::-1]
    for offset, prefix, color, label in [(.15, '', REAL, 'Real pairs (random sample)'), (-.15, 'null_', NULL, 'Random-walk null')]:
        v = pooled.loc[names, prefix + 'rho'].to_numpy(float)
        lo = pooled.loc[names, prefix + 'rho_lo'].to_numpy(float); hi = pooled.loc[names, prefix + 'rho_hi'].to_numpy(float)
        ax.hlines(y + offset, lo, hi, color=color, linewidth=2)
        ax.plot(v, y + offset, 'o', color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ax.axvline(0, color=INK2, linewidth=.8)
    ax.axvline(.2, color=INK2, linewidth=.8, linestyle='--')
    ax.text(.2, len(names) - .3, ' persistent threshold 0.20', color=INK2, fontsize=8, va='bottom')
    ax.set_yticks(y, [f'{n}  ({dict((b, a) for a, b in PROPERTIES)[n]})' for n in names])
    ax.set_xlabel('Spearman ρ(window A, window B): fold-weighted within-fold mean, 95% year-block interval')
    ax.set_title('Which train properties persist from years 1–2 to year 3?', loc='left')
    ax.legend(loc='lower right')
    return save(fig, '01_spearman_real_vs_null.png')

def fold_ranks(frame, column):
    out = np.full(len(frame), math.nan)
    for _, index in frame.groupby('fi').groups.items():
        values = frame.loc[index, column].to_numpy(float)
        keep = ~np.isnan(values)
        from pp_stats import midranks
        r = np.full(len(values), math.nan); r[keep] = midranks(values[keep]) / keep.sum()
        out[frame.index.get_indexer(index)] = r
    return out

def scatter_plot(plt, real, null, P):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blues', BLUES)
    fig, axes = plt.subplots(len(SCATTER), 2, figsize=(8, 16), sharex=True, sharey=True)
    pooled = P[P.stratum_type == 'pooled'].set_index('property')
    for row, (name, label) in enumerate(SCATTER):
        for col, (frame, title, key) in enumerate([(real, 'Real pairs', 'rho'), (null, 'Random-walk null', 'null_rho')]):
            ax = axes[row, col]
            a, b = fold_ranks(frame, name + '_A'), fold_ranks(frame, name + '_B')
            keep = ~np.isnan(a) & ~np.isnan(b)
            ax.hexbin(a[keep], b[keep], gridsize=25, cmap=cmap, mincnt=1, linewidths=.2, edgecolors=SURFACE)
            ax.set_title(f'{title}: {label}  (ρ = {pooled.loc[name, key]:.2f})', loc='left', fontsize=9)
            ax.grid(False)
            if col == 0: ax.set_ylabel('Window B rank (within fold)')
            if row == len(SCATTER) - 1: ax.set_xlabel('Window A rank (within fold)')
    fig.suptitle('Window A vs window B, within-fold percentile ranks (darker = more pair-folds; shading scaled per panel)', x=.02, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .975))
    return save(fig, '02_rank_scatter_A_vs_B.png')

def G_plot(plt, G):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    eras = ['E1', 'E2', 'E3', 'E4', 'E5', 'all']
    part = G[(G.metric == 'G_pass') & (G.sector_relation == 'all')].set_index('era').loc[eras]
    x = np.arange(len(eras)); width = .38
    for offset, prefix, color, label in [(-width / 2, 'real', REAL, 'Real eligible pair-folds'), (width / 2, 'null', NULL, 'Random-walk null')]:
        v = part[prefix + '_rate'].to_numpy(); lo = part[prefix + '_lo'].to_numpy(); hi = part[prefix + '_hi'].to_numpy()
        left.bar(x + offset, v, width - .04, color=color, label=label)
        left.errorbar(x + offset, v, yerr=[v - lo, hi - v], fmt='none', ecolor=INK2, elinewidth=1, capsize=2)
    left.set_xticks(x, eras); left.set_ylabel('G pass rate')
    left.set_ylim(0, max(part.real_hi.max(), part.null_hi.max()) * 1.08)
    left.set_title('G pass rate by era', loc='left', pad=26)
    left.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    metrics = [('cond_entries', '≥ 6\nevents'), ('cond_reverted_share', 'reverted\nshare ≥ 0.80'), ('cond_price_share', 'price share\n≥ 0.70'),
               ('cond_net_positive', 'net mean\n> 0'), ('G_pass', 'all four\n(G)')]
    part = G[(G.era == 'all') & (G.sector_relation == 'all')].set_index('metric').loc[[m for m, _ in metrics]]
    x = np.arange(len(metrics))
    for offset, prefix, color in [(-width / 2, 'real', REAL), (width / 2, 'null', NULL)]:
        v = part[prefix + '_rate'].to_numpy(); lo = part[prefix + '_lo'].to_numpy(); hi = part[prefix + '_hi'].to_numpy()
        right.bar(x + offset, v, width - .04, color=color)
        right.errorbar(x + offset, v, yerr=[v - lo, hi - v], fmt='none', ecolor=INK2, elinewidth=1, capsize=2)
    right.set_xticks(x, [l for _, l in metrics]); right.set_ylabel('Share of pair-folds passing')
    right.set_title('Each G condition, all folds', loc='left', pad=26)
    fig.tight_layout()
    return save(fig, '03_G_vs_null.png')

def ticker_plot(plt, wide, top):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    z = wide.FAST_z.to_numpy(float); z = z[np.isfinite(z)]
    bins = np.linspace(-4, max(8, min(np.nanmax(z), 30)), 70)
    left.hist(np.clip(z, bins[0], bins[-1]), bins=bins, density=True, color=REAL, edgecolor=SURFACE, linewidth=.5,
              label='Real ticker-folds (FAST63)')
    grid = np.linspace(bins[0], bins[-1], 400)
    left.plot(grid, np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi), color=NULL, linewidth=2,
              label='Binomial expectation, normal approx. (common fold rate)')
    left.set_xlabel('Standardized ticker pass count (k − n·p) / √(n·p·(1−p)); right tail clipped')
    left.set_ylabel('Density'); left.legend(loc='upper right')
    left.set_title('Ticker pass rates vs a common pass rate', loc='left')
    t = top[(top.indicator == 'FAST63') & (top['rank'] > 0)].sort_values('rank', ascending=False)
    right.barh(np.arange(len(t)), t.pooled_degree, color=REAL, height=.75)
    right.set_yticks(np.arange(len(t)), [f'{r.ticker} · {r.sector}' for r in t.itertuples()], fontsize=7.5)
    for i, r in enumerate(t.itertuples()):
        right.text(r.pooled_degree, i, f'  {r.pass_rate:.0%}', va='center', fontsize=7, color=INK2)
    right.set_xlabel('Pooled FAST63 degree over 22 folds (label: pooled pass rate)')
    right.set_title('Top 30 tickers by pooled FAST degree', loc='left'); right.grid(axis='y', visible=False)
    fig.tight_layout()
    return save(fig, '04_ticker_effects.png')

def plots():
    marker = HERE / 'audit/plots_ready.json'
    if ready(marker): return read(marker)
    plt = style()
    P = pd.read_csv(TABLES / 'property_persistence.csv')
    G = pd.read_csv(TABLES / 'G_vs_null.csv')
    real = pd.read_csv(TABLES / 'property_values_sample.csv.gz')
    real = real[real.in_random.astype(bool)].reset_index(drop=True)
    null = pd.read_csv(TABLES / 'property_values_null.csv.gz')
    for frame in (real, null): frame['fi'] = frame.test_year - YEARS[0]
    wide = pd.read_csv(TABLES / 'ticker_effects.csv'); top = pd.read_csv(TABLES / 'top_degree_tickers.csv')
    files = [spearman_plot(plt, P), scatter_plot(plt, real, null, P), G_plot(plt, G), ticker_plot(plt, wide, top)]
    receipt(marker, files, PNGs=len(files))
    return read(marker)
