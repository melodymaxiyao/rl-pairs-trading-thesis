"""Chunk-ready train statistics using original numeric helpers unchanged."""
from universe_io import *
import numpy as np
import pandas as pd
from types import SimpleNamespace
from train_reader import read_window, complete_test
from events import screen_events, summarize, legacy_removals
from screen import conditions, passes

def fold_spec(year):
    return dict(fold=year-2003, train_start=f'{year-3}-01-01',
                train_end=f'{year-1}-12-31', test_start=f'{year}-01-01', test_end=f'{year}-12-31')

class FoldData:
    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k!='h'}

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.h=set_year(self.year)
        assert self.cal.max().year<self.year
        assert all(frame.index.max().year<self.year for frame in self.market.values())

    def __init__(self, year, stocks):
        self.year, self.h = year, set_year(year)
        self.cal = self.h.comp.CAL.copy()
        self.test_cal = self.h.full_cal[self.h.full_cal.year == year]
        self.market, self.logs, self.test_ok, self.single, self.masks = {}, {}, {}, {}, {}
        self.test_dates_ok={}
        self.stocks = stocks
        for stock in stocks:
            ticker = stock['symbol']
            path = note(FRESH / 'data/raw/prices' / (ticker.lower() + '_yahoo.csv'))
            raw, test = read_window(path, year)
            frame = pd.DataFrame(dict(adjusted_close=raw['Adj Close'].reindex(self.cal),
                                      unadjusted_close=raw.Close.reindex(self.cal),
                                      volume=raw.Volume.reindex(self.cal)), index=self.cal)
            self.market[ticker] = frame
            with np.errstate(divide='ignore', invalid='ignore'):
                self.logs[ticker] = np.log(frame.adjusted_close.to_numpy())
            self.test_ok[ticker] = complete_test(test, self.test_cal)
            self.test_dates_ok[ticker]=complete_test([(date,True) for date,_ in test],self.test_cal)
            record('date_filtered_input', test_year=year, source=str(path.relative_to(ROOT)),
                   train_numeric_rows=len(raw), max_numeric_year=int(raw.index.year.max()) if len(raw) else None,
                   test_date_rows=len(test), test_presence_booleans=len(test), test_numeric_rows=0)
        for ticker in self.market:
            frame = pd.DataFrame(dict(log_a=self.logs[ticker], log_b=self.logs[ticker]), index=self.cal)
            frame.index.name = 'date'
            spec = SimpleNamespace(anchor=ticker, candidate=ticker, origin='new_sector', already_in_universe=False)
            for inner in (False, True):
                result = self.h.prepare.train_eligibility(spec, frame, self.market, fold_spec(year), inner)
                self.single[ticker, inner] = result
                end = year - (2 if inner else 1)
                train = self.cal[(self.cal.year >= year-3) & (self.cal.year <= end)]
                start = self.cal.get_loc(train[0])
                required = self.cal[max(0, start-312):self.cal.get_loc(train[-1])+1]
                self.masks[ticker, inner] = pd.isna(frame.reindex(required).log_a.to_numpy())
                record('original_ticker_eligibility', original_eligibility_calls=1, inner=inner, test_rows_used=0)

    def frame(self, a, b):
        result = pd.DataFrame(dict(log_a=self.logs[a], log_b=self.logs[b]), index=self.cal)
        result.index.name = 'date'
        return result

    def eligibility(self, a, b, inner=False):
        x, y = self.single[a, inner], self.single[b, inner]
        available = min(x['available_history'], y['available_history'])
        missing = float(np.mean(self.masks[a, inner] | self.masks[b, inner]))
        stale = max(x['max_stale_run_required'], y['max_stale_run_required'])
        flagged = (x['n_flagged_returns_required'] + y['n_flagged_returns_required']) // 2
        history = bool(x['F_history_eligible'] and y['F_history_eligible'])
        liquid = bool(x['liquidity_gate'] and y['liquidity_gate'])
        quality = missing <= .02 and stale <= 5
        reasons = []
        if not history: reasons.append('stage_A_requires_complete_train_and_F_formation_years')
        if available < 312: reasons.append('insufficient_312_session_R_history')
        if missing > .02: reasons.append('required_window_missing_share_gt_2pct')
        if stale > 5: reasons.append('identical_close_run_gt_5')
        if not liquid: reasons.append('liquidity_gate_failed_or_unavailable')
        result = dict(available_history=available, required_R_history=312, required_gate_history=251,
                      train_sessions=x['train_sessions'], F_history_eligible=history,
                      R_history_eligible=history and available>=312, missing_share_required=missing,
                      max_stale_run_required=stale, n_flagged_returns_required=flagged,
                      quality_review=flagged>0, quality_pass=quality, liquidity_gate=liquid,
                      liquidity_descriptive_only=False, F_eligible=history and quality and liquid,
                      R_eligible=history and available>=312 and quality and liquid,
                      gate_eligible=history and available>=251 and quality and liquid,
                      eligibility_test_data_used=False, drop_reason=';'.join(reasons))
        for leg, original in [('A', x), ('B', y)]:
            for field in ['median_dollar_volume', 'zero_volume_share', 'liquidity_complete', 'liquidity_pass']:
                result[leg+'_'+field] = original['A_'+field]
        return result

def make_episodes(h, cal, log_a, log_b, year, pair, years):
    result = []
    for score_year in years:
        record('F_construction_entered',F_construction_calls=1,score_year=score_year,test_year=year,test_rows_used=0)
        formation = cal.year == score_year-1
        score = cal.year == score_year
        assert formation.sum() and score.sum()
        y, x = log_a[formation], log_b[formation]
        if not np.isfinite(np.r_[y, x, log_a[score], log_b[score]]).all():
            raise ValueError('Incomplete formation or scoring year')
        alpha, beta, residual = h.comp.ols(y, x)
        mu, sigma = float(residual.mean()), float(residual.std(ddof=1))
        if not sigma > 0:
            raise ValueError('zero F scale')
        la, lb = log_a[score], log_b[score]
        z = (la-alpha-beta*lb-mu)/sigma
        previous = float((y[-1]-alpha-beta*x[-1]-mu)/sigma)
        theta = np.tile(np.array([alpha, beta, mu, sigma]), (len(z), 1))
        fa, fb = np.r_[np.diff(la)*10000, 0.], np.r_[np.diff(lb)*10000, 0.]
        result.append(dict(pair=pair, fold=year-2003, year=score_year, window='train', construction='F',
                           dates=cal[score], z=z, beta=theta[:,1], theta=theta, log_a=la, log_b=lb,
                           forward_bps=(fa-beta*fb)/(1+abs(beta)), previous_z=previous,
                           category='equity_peer', origin='new_sector', **{'class':'synthetic'},
                           S2=False, A=True, near_white_noise=False, R_eligible=True, F_eligible=True))
    return result

def episode_statistics(h, es, inner=False):
    record('screen_entered',screen_evaluations=1,inner=inner,test_rows_used=0)
    interval = h.ho.gate().diag.dec.interval
    primary = screen_events(es, interval, 2.0)
    a15 = screen_events(es, interval, 1.5)
    stats = summarize(primary)
    stats.update(n_censored=stats['n_events']-stats['n_reverted'],screen_events_crossing_annual_reset=0,
                 events_by_year={str(e['year']):sum(ev['entry_date'].year==e['year'] for ev in primary) for e in es})
    sens = summarize(a15)
    vr = h.comp.variance_ratios([e['z'] for e in es], qs=(5,10))
    legacy_frame = h.ho.gate_events(es, len(es))
    legacy = legacy_frame.to_dict('records')
    legacy_stats = summarize(legacy)
    assert stats['n_events'] >= stats['n_reverted']
    assert sens['n_reverted'] <= legacy_stats['n_reverted'], 'Non-overlap reverted count exceeded legacy'
    if legacy_frame.empty:
        legacy_frame = pd.DataFrame(columns=['window','anchor','construction','pair','fold','ideal_pnl_bps','reverted','price_share'])
        legacy_frame['reverted'] = legacy_frame.reverted.astype(bool)
    decision = h.ho.decide(legacy_frame, [(es[0]['pair'], es[0]['fold'])]).iloc[0].to_dict()
    stats.update(VR_5=vr['VR_5'], VR_10=vr['VR_10'], near_white_noise=bool(vr['VR_5'] < .3),
                 KM_med_a15=sens['KM_med'], n_events_a15=sens['n_events'], n_reverted_a15=sens['n_reverted'],
                 median_reverted_gross_bps_a15=sens['median_reverted_gross_bps'],
                 KM_med_legacy=legacy_stats['KM_med'], n_reverted_legacy=legacy_stats['n_reverted'],
                 median_reverted_gross_bps_legacy=legacy_stats['median_reverted_gross_bps'])
    for suffix, source in [('',stats), ('_a15',sens), ('_legacy',legacy_stats)]:
        stats['FAST63'+suffix] = passes(source['KM_med'], source['n_reverted'], source['median_reverted_gross_bps'], vr['VR_5'], inner)
    stats['FAST126'] = passes(stats['KM_med'], stats['n_reverted'], stats['median_reverted_gross_bps'], vr['VR_5'], inner, 126)
    stats.update(conditions(stats['KM_med'],stats['n_reverted'],stats['median_reverted_gross_bps'],vr['VR_5'],inner))
    stats.update(G_pass=bool(decision['passes']), G_n_events=int(decision['n_events']),
                 G_reverted_share=decision['reverted_share'],
                 G_median_price_share_reverted=decision['median_price_share_reverted'],
                 G_gross_mean_bps=decision['gross_mean_bps'], G_mean_net_bps=decision['gross_mean_bps']-10)
    stats.update(legacy_removals(legacy, a15, {e['dates'][0] for e in es}))
    ar = h.eda.ar1(es).iloc[0]
    daily = np.concatenate([e['forward_bps'][:-1] for e in es])
    z = np.concatenate([e['z'] for e in es])
    mu, sd = float(daily.mean()), float(daily.std(ddof=1))
    quarterly = []
    for e in es:
        for quarter in range(1,5):
            mask = e['dates'].quarter == quarter
            quarterly.append(h.comp.ols(e['log_a'][mask],e['log_b'][mask])[1])
    qmean = float(np.mean(quarterly))
    la = np.concatenate([e['log_a'] for e in es]); lb = np.concatenate([e['log_b'] for e in es])
    stats.update(AR1_phi=float(ar.phi), half_life=float(ar.half_life),
                 abs_z_ge_1_5_share=float(np.mean(abs(z)>=1.5)), beta=float(np.mean([e['theta'][0,1] for e in es])),
                 beta_by_year={str(e['year']):float(e['theta'][0,1]) for e in es},
                 quarterly_beta_cv=float(np.std(quarterly,ddof=1)/abs(qmean)) if qmean else np.nan,
                 train_drift_bps_per_year=252*mu, train_drift_t=mu/(sd/np.sqrt(len(daily))) if sd else np.nan,
                 train_drift_sign=h.data.h1_sign(es)[es[0]['pair']],
                 daily_return_correlation=float(np.corrcoef(np.diff(la),np.diff(lb))[0,1]))
    prefix='inner_' if inner else 'final_'
    record('screen_statistics_completed',screen_completions=1,AR1_calls=1,VR_calls=1,
           original_entry_event_calls=1,original_G_calls=1,quarterly_OLS_calls=4*len(es),
           custom_interval_calls=len(primary)+len(a15),legacy_interval_calls=len(legacy),
           **{prefix+k:stats[k] for k in ['legacy_removed_overlap','legacy_removed_year_end','legacy_removed_year_start','legacy_year_end_crossings']})
    return stats, primary, legacy

def evaluate_pair(context, candidate, return_episodes=False):
    h, year = context.h, context.year
    a,b = candidate['leg_A'], candidate['leg_B']
    full = context.eligibility(a,b)
    inner = context.eligibility(a,b,True)
    test = context.test_ok[a] and context.test_ok[b]
    row = dict(candidate, **full, eligible=bool(full['F_eligible'] and test),
               inner_eligible=bool(inner['F_eligible'] and test), test_path_complete=test,
               test_dates_complete=context.test_dates_ok[a] and context.test_dates_ok[b],
               exclusion_reason='' if full['F_eligible'] and test else full['drop_reason'] if not full['F_eligible'] else 'incomplete_test_path',
               test_numeric_rows_read=0)
    es = []
    inner_constructible = bool(inner['F_history_eligible'])
    if full['F_eligible'] or inner_constructible:
        ys = range(year-3,year) if full['F_eligible'] else range(year-3,year-1)
        es = make_episodes(h,context.cal,context.logs[a],context.logs[b],year,row['pair'],ys)
    for prefix, selected, ok in [('',es,full['F_eligible']), ('inner_',es[:2],inner_constructible)]:
        if ok:
            stats,_,_ = episode_statistics(h,selected,bool(prefix))
            row.update({prefix+k:v for k,v in stats.items()})
        else:
            row.update({prefix+'FAST63':False,prefix+'FAST126':False,prefix+'FAST63_a15':False,prefix+'FAST63_legacy':False})
    if full['F_eligible']:
        validation = screen_events(es[-1:],h.ho.gate().diag.dec.interval,2.0)
        row['validation_KM_med'] = summarize(validation)['KM_med']
    else:
        row['validation_KM_med'] = np.inf
    return (row,es) if return_episodes else row
