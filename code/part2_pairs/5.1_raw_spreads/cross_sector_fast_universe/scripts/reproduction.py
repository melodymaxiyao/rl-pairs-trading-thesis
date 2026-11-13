"""Required real-data reproduction checks, invoked by the user's run command."""
from universe_io import *
from candidates import stocks_for_year, candidate_pairs, unordered
from numeric import FoldData, make_episodes, evaluate_pair, fold_spec, episode_statistics
import numpy as np
import pandas as pd
from types import SimpleNamespace

def same(a,b):
    if a is None or (isinstance(a,float) and np.isnan(a)):
        return b is None or (isinstance(b,float) and np.isnan(b))
    return a == b

def cross_sector_sample():
    marker=HERE/'audit/cross_sector_reproduction.json'
    if marker.exists():
        saved=read(marker); assert saved['status']=='PASS' and saved['pair_folds']>=2000
        return saved
    count=0; max_error=0.; rejected=0
    for year in range(2004,2026):
        stocks,_=stocks_for_year(year); context=FoldData(year,stocks)
        rng=np.random.default_rng(np.random.SeedSequence([20261020,year,2000]))
        wanted=91 if year<2024 else 90
        chosen=set(); trials=0
        while len(chosen)<wanted:
            trials+=1
            assert trials<1000000, ('Not enough eligible cross-sector pairs',year)
            ia,ib=sorted(rng.choice(len(stocks),2,replace=False))
            a,b=stocks[ia],stocks[ib]
            if a['sector']==b['sector'] or (ia,ib) in chosen: continue
            ea,eb=a['symbol'],b['symbol']
            eligibility=context.eligibility(ea,eb)
            if not eligibility['F_eligible'] or not(context.test_ok[ea] and context.test_ok[eb]):
                rejected+=1;continue
            chosen.add((ia,ib))
            spec=SimpleNamespace(anchor=ea,candidate=eb,origin='new_sector',already_in_universe=False)
            scalar=context.h.prepare.train_eligibility(spec,context.frame(ea,eb),context.market,fold_spec(year))
            record('scalar_eligibility_reference_completed',scalar_eligibility_reference_calls=1,test_rows_used=0)
            assert set(scalar)==set(eligibility)
            for key in scalar:
                assert same(scalar[key],eligibility[key]),('Eligibility mismatch',year,ea,eb,key,scalar[key],eligibility[key])
            vector=make_episodes(context.h,context.cal,context.logs[ea],context.logs[eb],year,ea+'_'+eb,range(year-3,year))
            scalar_es=[]
            for e in vector:
                path=context.h.comp.frozen_year(context.frame(ea,eb),e['year'])
                record('scalar_F_reference_completed',scalar_F_reference_calls=1,test_rows_used=0)
                assert path.index.max().year<year
                se=context.h.ho.episode(dict(pair=e['pair'],category='equity_peer',cls='synthetic'),year-2003,e['year'],'train',path)
                scalar_es.append(se)
                err=float(np.max(abs(e['z']-se['z'])));max_error=max(max_error,err)
                assert err<=1e-10,('F z mismatch',year,ea,eb,e['year'],err)
            vg=context.h.ho.gate_events(vector,3);sg=context.h.ho.gate_events(scalar_es,3)
            pd.testing.assert_frame_equal(vg,sg,check_exact=True)
            for inner in (False,True):
                vstats,ve,_=episode_statistics(context.h,vector[:2] if inner else vector,inner)
                sstats,se,_=episode_statistics(context.h,scalar_es[:2] if inner else scalar_es,inner)
                assert clean(vstats)==clean(sstats),('Screen/G inputs mismatch',year,ea,eb,inner)
                for va,sa in zip(ve,se):
                    assert clean(va)==clean(sa)
            count+=1
        print('Reproduction sample:',year,wanted,'eligible cross-sector pair-folds; total',count,flush=True)
    result=dict(status='PASS',pair_folds=count,max_abs_F_z_error=max_error,
                random_seed=20261020,rejected_ineligible_draws=rejected,
                eligibility_exact=True,legacy_events_exact=True,G_inputs_exact=True,
                primary_and_inner_screen_events_exact=True,test_numeric_rows_read=0)
    write(marker,result);return result

def compare_sector_fold(year,rows):
    old=[r for r in read(note(SECTOR/'outputs/tables/pair_fold_inventory.json')) if r['test_year']==year]
    current={unordered(r['leg_A'],r['leg_B']):r for r in rows}
    exact=0;legacy=[];missing_candidate=0
    fields={'G_n_events':'n_events','G_reverted_share':'reverted_share',
            'G_median_price_share_reverted':'median_price_share_reverted','G_gross_mean_bps':'gross_mean_bps'}
    for original in old:
        key=unordered(original['anchor'],original['candidate']); row=current.get(key)
        if original['existing']:
            candidate_present=row is not None
            if row is None:
                row=dict(pair='_'.join(key),leg_A=key[0],leg_B=key[1],F_eligible=False,G_pass=False)
            entry=dict(test_year=year,pair=row['pair'],recorded_pair=original['pair'],
                       candidate_under_new_rules=candidate_present,
                       fresh_exclusion_reason='' if candidate_present else 'not_a_PIT_explicit_GICS_stored_price_candidate_pair',
                       recorded_source=original.get('source'),fresh_source='fresh_sealed_2000_2025',
                       recorded_leg_order=original['anchor']+' / '+original['candidate'],
                       alphabetical_leg_order=row['leg_A']+' / '+row['leg_B'],
                       recorded_eligible=original.get('F_eligible',original['eligible']),
                       recomputed_eligible=row['F_eligible'],recorded_G_pass=original['G_pass'],
                       recomputed_G_pass=row.get('G_pass',False))
            for new,old_field in fields.items():
                entry['recorded_'+new]=original.get(old_field)
                entry['recomputed_'+new]=row.get(new)
            entry['eligibility_differs']=entry['recorded_eligible']!=entry['recomputed_eligible']
            entry['G_differs']=entry['recorded_G_pass']!=entry['recomputed_G_pass'] or any(
                not same(entry['recorded_'+f],entry['recomputed_'+f]) for f in fields)
            legacy.append(entry);continue
        assert original['source']=='fresh_sealed_2000_2025'
        assert original['anchor']<original['candidate']
        if row is None:
            assert not original['eligible'],('Required old new pair absent but eligible',year,key)
            missing_candidate+=1;exact+=1;continue
        assert row['sector_relation']=='same_subindustry'
        assert bool(row['F_eligible'])==bool(original.get('F_eligible',original['eligible'])),('Same-subindustry train eligibility mismatch',year,key)
        old_eligible=bool(row['F_eligible'] and row['test_dates_complete'])
        assert old_eligible==bool(original['eligible']),('Original date-only eligibility mismatch',year,key)
        reproduced_G=bool(old_eligible and row.get('G_pass',False))
        assert reproduced_G==bool(original['G_pass']),('Original G membership mismatch',year,key)
        if original['eligible']:
            for new,old_field in fields.items():
                assert same(row.get(new),original[old_field]),('Same-subindustry G input mismatch',year,key,new,row.get(new),original[old_field])
            assert bool(row['G_pass'])==bool(original['G_pass']),('Same-subindustry G mismatch',year,key)
        else:
            assert not original['G_pass']
        exact+=1
    return dict(test_year=year,exact_new_pair_folds=exact,missing_price_candidates=missing_candidate,
                legacy_rows=len(legacy),legacy_eligibility_differences=sum(r['eligibility_differs'] for r in legacy),
                legacy_G_differences=sum(r['G_differs'] for r in legacy),status='PASS'),legacy
