"""Event-IPW statistics and paired fold-block contrasts; no pooled-ratio shortcut."""
from aio import *
from ah_stats import matrix,mean_draws,median_draws,ratio_draws,summarise,holm
from originals import product_limit
from engine import ANCHORS,INFEASIBLE,ADAPTIVE,BINS,HORIZONS
import numpy as np
import pandas as pd

COLUMNS=['year','relation','alignment','weight','duration','reverted','gross',*[f'forward_{h}' for h in HORIZONS],
         'driver_num','driver_den','price_part','anchor_part','scale_part','delta_d']
INDEX={k:i for i,k in enumerate(COLUMNS)}
STRATA=[('pooled','all','all','all'),('X','X','all','all'),('W','W','all','all'),
        ('same_sector','same_sector','all','all'),('same_subindustry','same_subindustry','all','all')]
STRATA += [(e,'all',e,'all') for e in ERAS]
STRATA += [('drift_'+a,'all','all',a) for a in ['aligned','counter','unknown']]
STRATA += [(rel+'_drift_'+a,rel,'all',a) for rel in ['X','W'] for a in ['aligned','counter','unknown']]

def relation_mask(rel,code):
    return np.ones(len(code),bool) if rel=='all' else code==0 if rel=='X' else code!=0 if rel=='W' else code==1 if rel=='same_sector' else code==2

def select(array,relation,period,alignment):
    mask=relation_mask(relation,array[:,INDEX['relation']])
    if period!='all':
        a,b=ERAS[period];mask&=(array[:,0]>=a)&(array[:,0]<=b)
    x=array[:,INDEX['alignment']]
    if alignment!='all':mask&=x==1 if alignment=='aligned' else x==-1 if alignment=='counter' else np.isnan(x)
    return array[mask]

def difference(a,b):
    with np.errstate(invalid='ignore'):return a[0]-b[0],a[1]-b[1],min(a[2],b[2]),min(a[3],b[3])

def empty(M):return np.nan,np.full(len(M),np.nan),0,0

class Analysis:
    def __init__(self,pairs,years=YEARS,draws=2000):
        self.pairs=pairs;self.years=years;self.draws=draws;self.Ms={};self.stats={};self.rows=[];self.exclusions=[];self.curves=[]
    def M(self,period):
        years=self.years if period=='all' else [y for y in self.years if ERAS[period][0]<=y<=ERAS[period][1]]
        key=tuple(years or self.years)
        if key not in self.Ms:self.Ms[key]=matrix(key,self.draws)
        return self.Ms[key]
    def row(self,key,metric,result,**extra):
        anchor,regime,dataset,stratum=key
        return dict(anchor=anchor,feasibility='INFEASIBLE' if anchor in INFEASIBLE else 'CAUSAL',regime=regime,dataset=dataset,stratum=stratum,
            metric=metric,**summarise(*result),status='not_estimable' if np.isnan(result[0]) else 'ok',**extra)
    def pair_mass(self,dataset,rel,period):
        p=self.pairs[(self.pairs.dataset==dataset)&(self.pairs.status=='ok')]
        code=p.relation.map({'cross_sector':0,'same_sector':1,'same_subindustry':2}).to_numpy()
        mask=relation_mask(rel,code)
        if period!='all':
            a,b=ERAS[period];mask&=(p.year.to_numpy()>=a)&(p.year.to_numpy()<=b)
        p=p[mask];return np.bincount(p.year.to_numpy(int)-2004,weights=p.weight,minlength=22)*3
    def cell(self,anchor,regime,dataset,array):
        assert array.shape[1]==len(COLUMNS)
        for stratum,rel,period,alignment in STRATA:
            M=self.M(period);a=select(array,rel,period,alignment);n=len(a)
            year=a[:,0].astype(int);w=a[:,INDEX['weight']];folds=len(np.unique(year));f=year-2004
            key=(anchor,regime,dataset,stratum);results={}
            def mean(values):return mean_draws(values,w,year,M)
            def median(values,p=.5):return median_draws(values,w,year,M,p)
            def ratio(num,den):
                keep=np.isfinite(num)&np.isfinite(den)&(den>0)
                numerator=np.bincount(f[keep],weights=w[keep]*np.asarray(num)[keep],minlength=22)
                denominator=np.bincount(f[keep],weights=w[keep]*np.asarray(den)[keep],minlength=22)
                point,draws=ratio_draws(numerator,denominator,M)
                return point,draws,int(keep.sum()),len(np.unique(f[keep]))
            gross=a[:,INDEX['gross']];rev=a[:,INDEX['reverted']]
            results['mean_gross']=mean(gross);results['mean_net']=tuple([results['mean_gross'][0]-10,results['mean_gross'][1]-10,*results['mean_gross'][2:]])
            results['median_gross']=median(gross);results['median_net']=tuple([results['median_gross'][0]-10,results['median_gross'][1]-10,*results['median_gross'][2:]])
            results['hit_rate']=mean((gross>10).astype(float));results['reverted_share']=mean(rev)
            pairmass=self.pair_mass(dataset,rel,period);eventmass=np.bincount(f,weights=w,minlength=22);point,draws=ratio_draws(eventmass,pairmass,M)
            results['entries_per_pair_year']=(point,draws,n,int((pairmass>0).sum()))
            d=np.zeros((22,BINS));c=d.copy()
            for flag,target in [(1.,d),(0.,c)]:
                indices=rev==flag;np.add.at(target,(f[indices],a[indices,INDEX['duration']].astype(int)),w[indices])
            _,med=product_limit(d.sum(axis=0),c.sum(axis=0));_,boot=product_limit(M@d,M@c)
            results['KM_median']=(med[0],boot,n,folds)
            if anchor=='A_F' and stratum=='pooled':
                deaths=d.sum(axis=0);censors=c.sum(axis=0);risk=np.cumsum((deaths+censors)[::-1])[::-1]
                survival=np.cumprod(1-np.divide(deaths,risk,out=np.zeros(BINS),where=risk>0))
                for t in np.flatnonzero(risk>0):self.curves.append(dict(anchor=anchor,feasibility='CAUSAL',regime=regime,dataset=dataset,duration=int(t),survival=float(survival[t]),weighted_at_risk=float(risk[t])))
            for h in HORIZONS:results[f'mean_forward_{h}']=mean(a[:,INDEX[f'forward_{h}']])
            if anchor in ADAPTIVE:
                numerator=a[:,INDEX['driver_num']];denominator=a[:,INDEX['driver_den']]
                valid=(rev==1)&np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0)
                num=np.where(valid,numerator,np.nan);den=np.where(valid,denominator,np.nan)
                results['price_share_ratio']=ratio(num,den)
                share=np.divide(num,den,out=np.full(n,np.nan),where=valid)
                for p,name in [(.25,'price_share_q25'),(.5,'price_share_median'),(.75,'price_share_q75')]:results[name]=median(share,p)
                results['price_share_IQR']=difference(results['price_share_q75'],results['price_share_q25'])
                results['outside_unit_interval_share']=mean(np.where(valid,((share<0)|(share>1)).astype(float),np.nan))
                results['anchor_and_scale_share_ratio']=tuple([1-results['price_share_ratio'][0],1-results['price_share_ratio'][1],*results['price_share_ratio'][2:]])
                results['driver_reverted_coverage']=ratio(valid.astype(float),np.where(rev==1,1.,np.nan))
                delta=a[:,INDEX['delta_d']];orientation=np.sign(delta)
                results['total_deviation_change_signed']=mean(np.where(valid,orientation*delta,np.nan))
                for part in ['price','anchor','scale']:
                    raw=a[:,INDEX[part+'_part']]
                    results[part+'_deviation_part_signed']=mean(np.where(valid,orientation*raw,np.nan))
                    results[part+'_component_share']=mean(np.divide(raw,delta,out=np.full(n,np.nan),where=valid))
            self.stats[key]=results
            for metric,value in results.items():self.rows.append(self.row(key,metric,value))
            for h in HORIZONS:
                missing=~np.isfinite(a[:,INDEX[f'forward_{h}']])
                self.exclusions.append(dict(anchor=anchor,feasibility='INFEASIBLE' if anchor in INFEASIBLE else 'CAUSAL',regime=regime,dataset=dataset,stratum=stratum,h=h,
                    n_events=n,forward_excluded=int(missing.sum()),weighted_forward_excluded=float(w[missing].sum()),
                    censored_events=int((rev==0).sum()),weighted_censored=float(w[rev==0].sum()),
                    driver_censored_excluded=int((rev==0).sum()) if anchor in ADAPTIVE else 0,
                    driver_unavailable_reverted=int(((rev==1)&~np.isfinite(a[:,INDEX['driver_den']])).sum()) if anchor in ADAPTIVE else 0,
                    unknown_alignment=int(np.isnan(a[:,INDEX['alignment']]).sum()),
                    forward_exclusion_reason='year_end' if regime=='within-year' else 'window_end'))
    def contrasts(self):
        rows=[]
        for key,results in self.stats.items():
            anchor,regime,dataset,stratum=key
            if dataset!='real':continue
            for null in ['N2','N1']:
                other=self.stats[(anchor,regime,null,stratum)]
                for metric,value in results.items():rows.append(self.row((anchor,regime,'real-'+null,stratum),metric,difference(value,other[metric])))
        return rows
    def get(self,anchor,regime,dataset,stratum,metric):return self.stats[(anchor,regime,dataset,stratum)][metric]
    def gap(self,anchor,regime,stratum='pooled',metric='mean_net',null='N2'):
        return difference(self.get(anchor,regime,'real',stratum,metric),self.get(anchor,regime,null,stratum,metric))
    def predictions(self):
        family={}
        for name,anchor in [('Q2-F','A_F'),('Q2-R252','A_R252'),('Q2-R63','A_R63'),('Q2-TR','A_TR')]:family[name]=self.gap(anchor,'within-year')
        family['Q3']=difference(self.gap('A_F','episode-free'),self.gap('A_F','within-year'))
        family['Q6']=difference(self.gap('A_F','within-year','drift_aligned'),self.gap('A_F','within-year','drift_counter'))
        info={k:summarise(*v) for k,v in family.items()};adjusted=holm([r['p_positive'] for r in info.values()]);rows=[]
        for (name,r),p in zip(info.items(),adjusted):rows.append(dict(prediction=name,family='primary',feasibility='CAUSAL',**r,p_raw=r['p_positive'],p_Holm=p,
            label='not_estimable' if np.isnan(r['estimate']) else 'supported' if r['estimate']>0 and p<=.05 else 'not_supported'))
        def add(name,result,anchor,regime,stratum='pooled',metric='mean_net',label=None,**extra):
            r=summarise(*result);rows.append(dict(prediction=name,family='descriptive',anchor=anchor,feasibility='INFEASIBLE' if anchor in INFEASIBLE else 'CAUSAL',
                regime=regime,stratum=stratum,metric=metric,**r,p_raw=np.nan,p_Holm=np.nan,
                label=label or ('not_estimable' if np.isnan(r['estimate']) else 'positive' if r['estimate']>0 else 'nonpositive'),**extra))
        for anchor in ['H_Y','H_W']:
            for regime in ['within-year','episode-free']:add('Q1',self.gap(anchor,regime),anchor,regime)
        for anchor in ['F_orig','A_F']:
            a=self.gap(anchor,'within-year',metric='KM_median',null='N1');b=self.gap(anchor,'within-year',metric='KM_median',null='N2')
            label='not_estimable' if not np.isfinite([a[0],b[0]]).all() else 'consistent_with_disappearance' if a[0]>0 and b[0]<=0 else 'not_consistent_with_disappearance'
            for null,result in [('N1',a),('N2',b)]:add('Q4',result,anchor,'within-year',metric='KM_median',label=label,comparison='real-'+null,role='primary_continuity' if anchor=='F_orig' else 'fixed_beta_companion')
        for anchor in ANCHORS:
            for regime in ['within-year','episode-free']:
                if anchor=='F_orig' and regime=='episode-free':continue
                for h in HORIZONS:add('Q5',self.gap(anchor,regime,metric=f'mean_forward_{h}'),anchor,regime,metric=f'mean_forward_{h}')
                for rel in ['X','W']:
                    add('Q7-Q2',self.gap(anchor,regime,rel),anchor,regime,rel)
                    aligned=self.gap(anchor,regime,rel+'_drift_aligned');counter=self.gap(anchor,regime,rel+'_drift_counter')
                    add('Q7-Q6',difference(aligned,counter),anchor,regime,rel,metric='aligned_minus_counter_net')
                if anchor in ADAPTIVE:
                    for metric in ['price_share_ratio','price_share_median','price_share_q25','price_share_q75','outside_unit_interval_share']:
                        add('Q8',self.gap(anchor,regime,metric=metric),anchor,regime,metric=metric)
        return rows
