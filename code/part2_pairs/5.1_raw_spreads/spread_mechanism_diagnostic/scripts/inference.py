"""Frozen IPW estimands; every contrast reuses the same year-block draws."""
from mio import *
from jobs import VIEWS,RELATIONS
from diagnostics import DURATION_BINS
from sources import weighted_helpers
from statistics import matrix,median_draws,mean_draws,ratio_draws,km_draws,summarise,holm
import numpy as np
import pandas as pd

def masks(frame,partial=False):
    for view in VIEWS:
        selected=frame.in_random.astype(bool) if view=='sample' else frame[view].astype(bool)
        base=frame.loc[selected].copy()
        base['weight']=pd.to_numeric(base['partial_weight' if partial and view=='sample' else 'ipw'],errors='coerce') if view=='sample' else 1.
        if partial:base=base[base.partial_weight.notna()]
        for relation in ['all',*RELATIONS]:
            r=base if relation=='all' else base[base.relation==relation]
            for period in ['all',*ERAS]:
                part=r if period=='all' else r[r.era==period]
                if len(part):yield dict(view=view,relation=relation,era=period),part

def empty(M):return np.nan,np.full(len(M),np.nan),0,0

def estimate(frame,column,M,how='median'):
    if not len(frame) or column not in frame:return empty(M)
    values=pd.to_numeric(frame[column],errors='coerce').to_numpy(float)
    return (median_draws if how=='median' else mean_draws)(values,frame.weight,frame.year,M)

def difference(a,b):return a[0]-b[0],a[1]-b[1],min(a[2],b[2]),min(a[3],b[3])

class Analysis:
    def __init__(self,years=YEARS,draws=2000):
        self.years=list(years);self.draws=draws;self.matrices={};self.pred={};self.tables={}
    def M(self,years):
        years=tuple(sorted(set(int(y) for y in years)))
        if not years:years=tuple(self.years)
        if years not in self.matrices:self.matrices[years]=matrix(years,self.draws)
        return self.matrices[years]
    def row(self,meta,metric,result,**extra):
        point,draws,n,folds=result
        return dict(meta,metric=metric,**summarise(point,draws,n,folds),status='ok' if not np.isnan(point) else 'not_estimable',**extra)
    def vr(self,basic):
        rows=[]
        for meta,frame in masks(basic):
            M=self.M(frame.year);sides={d:frame[frame.dataset==d] for d in ['real','null']}
            for q in [2,5,10,20,60,120,250]:
                col=f'VR_{q}';res={d:estimate(s,col,M) for d,s in sides.items()}
                for d,v in res.items():
                    rows.append(self.row(dict(meta,dataset=d,q=q),'median_VR',v))
                    s=sides[d].copy();s['_below']=np.where(pd.to_numeric(s[col],errors='coerce').notna(),s[col]<1,np.nan)
                    rows.append(self.row(dict(meta,dataset=d,q=q),'share_VR_below_1',estimate(s,'_below',M,'mean')))
                diff=difference(res['real'],res['null']);rows.append(self.row(dict(meta,dataset='real-null',q=q),'median_VR',diff))
                if meta==dict(view='sample',relation='all',era='all') and q in [10,120]:self.pred['P1' if q==10 else 'P2']=diff
        self.tables['vr_term_structure']=rows
        M=self.M(basic.year)
        sample=basic[basic.in_random].copy();sample['weight']=sample.ipw
        def gap(rels):
            f=sample[sample.relation.isin(rels)]
            return difference(estimate(f[f.dataset=='real'],'VR_120',M),estimate(f[f.dataset=='null'],'VR_120',M))
        self.pred['P8']=difference(gap(['cross_sector']),gap(['same_sector','same_subindustry']))
        self.pred['P8_same_sector_only']=difference(gap(['cross_sector']),gap(['same_sector']))
    def association(self,frame,x,y,M):
        h=weighted_helpers();fold_values=[];weights=[];years=[]
        valid=~np.isnan(pd.to_numeric(frame[x],errors='coerce'))&~np.isnan(pd.to_numeric(frame[y],errors='coerce'))
        frame=frame[valid]
        for year,part in frame.groupby('year'):
            fold_values.append(h.wspearman(part[x],part[y],part.weight));weights.append(part.weight.sum());years.append(year)
        res=mean_draws(fold_values,weights,years,M) if years else empty(M)
        return res,h.wspearman(frame[x],frame[y],frame.weight) if len(frame) else np.nan
    def partial(self,partial):
        rows=[]
        # Missing fit fields remain explicit NaN, never a successful zero estimate.
        for c in ['R2_MR','half_life_order','half_life','envelope_half_life','sd_M','swing_bps','swing_ge_20','rho_M','oscillatory','boundary']:
            if c not in partial:partial[c]=np.nan
        p4={}
        for meta,frame in masks(partial,True):
            M=self.M(frame.year)
            for window in ['full','A','B']:
                win=frame[frame.window==window]
                for boundary_policy in ['all_converged','without_boundaries']:
                    results={}
                    for dataset in ['real','null']:
                        attempted=win[win.dataset==dataset];ok=attempted[attempted.converged.astype(bool)]
                        if boundary_policy=='without_boundaries':ok=ok[~ok.boundary.fillna(True).astype(bool)]
                        for col,how in [('R2_MR','median'),('half_life_order','median'),('sd_M','median'),('swing_bps','median'),
                                        ('swing_ge_20','mean'),('oscillatory','mean'),('boundary','mean'),('envelope_half_life','median')]:
                            res=estimate(ok,col,M,how);results[(dataset,col)]=res
                            rows.append(self.row(dict(meta,dataset=dataset,window=window,boundary_policy=boundary_policy),col,res,
                                attempted=len(attempted),converged=int(attempted.converged.sum()),used=len(ok),
                                coverage=len(ok)/len(attempted) if len(attempted) else np.nan,
                                display='<1 session' if col=='half_life_order' and np.isneginf(res[0]) else None))
                        pos=ok[ok.rho_M>0];res=estimate(pos,'half_life',M)
                        rows.append(self.row(dict(meta,dataset=dataset,window=window,boundary_policy=boundary_policy),'half_life_positive_only',res,
                                             positive_rho_coverage=len(pos)/len(ok) if len(ok) else np.nan))
                    if window=='full':
                        diff=difference(results[('real','R2_MR')],results[('null','R2_MR')]);rows.append(self.row(dict(meta,dataset='real-null',window=window,boundary_policy=boundary_policy),'R2_MR',diff))
                        if meta==dict(view='sample',relation='all',era='all') and boundary_policy=='all_converged':
                            self.pred['P3']=diff;p4={k:results[('real',k)] for k in ['half_life_order','swing_ge_20']}
            for dataset in ['real','null']:
                for policy in ['all_converged','without_boundaries']:
                    ok=frame[(frame.dataset==dataset)&frame.converged.astype(bool)]
                    if policy=='without_boundaries':ok=ok[~ok.boundary.fillna(True).astype(bool)]
                    for col in ['R2_MR','half_life_order']:
                        a=ok[ok.window=='A'][['pair','year','weight',col]]
                        b=ok[ok.window=='B'][['pair','year',col]]
                        joined=a.merge(b,on=['pair','year'],suffixes=('_A','_B'),validate='one_to_one')
                        res,pooled=self.association(joined,col+'_A',col+'_B',M)
                        rows.append(self.row(dict(meta,dataset=dataset,window='A-to-B',boundary_policy=policy),col+'_Spearman',res,pooled_weighted_Spearman=pooled,
                                             paired_fits=len(joined)))
        self.p4=p4;self.tables['partial_ci_summary']=rows
    def drift(self,basic):
        rows=[];p5={}
        edges=[0,.5,1,1.5,2,np.inf]
        for meta,frame in masks(basic):
            M=self.M(frame.year)
            for predictor in ['lag2','lag1','last6months','A']:
                for bucket in ['all',0,1,2,3,4]:
                    strength=np.abs(frame[predictor+'_t'])
                    part=frame if bucket=='all' else frame[(strength>=edges[bucket])&(strength<edges[bucket+1] if bucket<4 else strength<=np.inf)]
                    res={}
                    for dataset in ['real','null']:
                        s=part[part.dataset==dataset].copy()
                        s=s[(s[predictor+'_sign']!=0)&(s.outcome_sign!=0)]
                        s['agreement']=np.where(s[predictor+'_sign']==s.outcome_sign,.5,-.5)
                        res[dataset]=estimate(s,'agreement',M,'mean')
                        rows.append(self.row(dict(meta,dataset=dataset,predictor=predictor,bucket=bucket),'sign_agreement_minus_half',res[dataset]))
                    rows.append(self.row(dict(meta,dataset='real-null',predictor=predictor,bucket=bucket),'sign_agreement_minus_half',difference(res['real'],res['null'])))
                    if meta==dict(view='sample',relation='all',era='all'):p5[(predictor,bucket)]=res['real'][0]
                for dataset in ['real','null']:
                    s=frame[frame.dataset==dataset];res,pooled=self.association(s,predictor+'_mean','outcome_mean',M)
                    rows.append(self.row(dict(meta,dataset=dataset,predictor=predictor,bucket='all'),'Spearman',res,pooled_weighted_Spearman=pooled))
        self.p5=p5;self.tables['drift_persistence_horizons']=rows
    def aggregate_bank(self,bank,view,relations,dataset,version,years):
        names=['deaths','censors','pair_mass','pairs'];out={n:np.zeros((22,DURATION_BINS)) if n in ['deaths','censors'] else np.zeros(22) for n in names}
        for relation in relations:
            value=bank.get((view,relation,dataset,version))
            if value:
                for n in names:out[n]+=value[n]
        selected=np.isin(np.arange(2004,2026),list(years))
        for n in names:out[n][~selected]=0
        return out
    def bank_stats(self,value,M,metric):
        d=value['deaths'];c=value['censors'];deaths=d.sum(axis=1);total=(d+c).sum(axis=1)
        n=int(value['pairs'].sum());folds=int((value['pairs']>0).sum())
        if metric=='KM63':point,draws,_,_=km_draws(d,c,M)
        elif metric=='KM_median':_,_,point,draws=km_draws(d,c,M)
        else:
            numerator={'reverted_share':deaths,'direct_within63_share':d[:,:64].sum(axis=1),
                       'censored_before63_share':c[:,:63].sum(axis=1),'events_per_pair_year':total/3}[metric]
            denominator=value['pair_mass'] if metric=='events_per_pair_year' else total
            point,draws=ratio_draws(numerator,denominator,M)
        return point,draws,n,folds
    def triggers(self,bank,gross):
        rows=[]
        for view in VIEWS:
            for relation in ['all',*RELATIONS]:
                rels=RELATIONS if relation=='all' else [relation]
                for period in ['all',*ERAS]:
                    years=self.years if period=='all' else [y for y in self.years if ERAS[period][0]<=y<=ERAS[period][1]]
                    M=self.M(years)
                    for dataset in ['real','null']:
                        for kind in ['volume-spike','jump','gradual']:
                            if dataset=='null' and kind=='volume-spike':continue
                            value=self.aggregate_bank(bank,view,rels,dataset,kind,years)
                            meta=dict(view=view,relation=relation,era=period,dataset=dataset,trigger=kind)
                            for metric in ['KM63','KM_median','direct_within63_share','censored_before63_share','events_per_pair_year']:
                                res=self.bank_stats(value,M,metric)
                                total=float((value['deaths']+value['censors']).sum())
                                rows.append(self.row(meta,metric,res,weighted_events=total,
                                    followup_at_least63_share=float((value['deaths'][:,63:]+value['censors'][:,63:]).sum()/total) if total else np.nan,
                                    KM63_identifiable=not np.isnan(self.bank_stats(value,M,'KM63')[0])))
                                if view=='sample' and relation=='all' and period=='all' and dataset=='real' and metric=='KM63':self.pred['P6_'+kind]=res
                            g=gross[(gross.view==view)&gross.relation.isin(rels)&(gross.dataset==dataset)&(gross.trigger==kind)&gross.year.isin(years)]
                            rows.append(self.row(meta,'median_reverted_gross_bps',estimate(g,'gross_bps',M)))
        self.pred['P6']=difference(self.pred.get('P6_volume-spike',empty(self.M(self.years))),self.pred.get('P6_gradual',empty(self.M(self.years))))
        self.tables['event_trigger_summary']=rows
    def p7(self,bank,control,years,version='factor_only',metric='reverted_share',view='sample'):
        M=self.M(years);cells={}
        for group,rels in [('X',['cross_sector']),('W',control)]:
            for dataset in ['real','null']:
                values={v:self.bank_stats(self.aggregate_bank(bank,view,rels,dataset,v,years),M,metric) for v in ['raw',version]}
                cells[(group,dataset)]=difference(values[version],values['raw'])
        return difference(difference(cells[('X','real')],cells[('W','real')]),difference(cells[('X','null')],cells[('W','null')]))
    def factor(self,frame,bank):
        rows=[]
        for meta,part in masks(frame):
            M=self.M(part.year)
            for dataset in ['real','null']:
                side=part[part.dataset==dataset]
                for version in ['raw','factor_only','factor_alpha']:
                    s=side[side.version==version];rels=RELATIONS if meta['relation']=='all' else [meta['relation']]
                    value=self.aggregate_bank(bank,meta['view'],rels,dataset,version,part.year.unique())
                    for metric in ['reverted_share','KM63','KM_median','events_per_pair_year']:
                        rows.append(self.row(dict(meta,dataset=dataset,version=version),metric,self.bank_stats(value,M,metric),
                            attempted=len(s),used=int((s.status=='ok').sum()),weighted_events=float((value['deaths']+value['censors']).sum())))
                    for q in [10,60,120]:rows.append(self.row(dict(meta,dataset=dataset,version=version),f'VR_{q}',estimate(s,f'VR_{q}',M)))
                    if version=='raw':continue
                    raw=side[side.version=='raw'];keys=['pair','year']
                    for q in [10,60,120]:
                        col=f'VR_{q}'
                        rows.append(self.row(dict(meta,dataset=dataset,version=version),'residual_minus_raw_'+col,
                                             difference(estimate(s,col,M),estimate(raw,col,M))))
                    joined=raw.merge(s,on=keys,suffixes=('_raw','_residual'),validate='one_to_one')
                    joined['weight']=joined['weight_raw']
                    for col in ['VR_10','VR_60','VR_120','reverted_share']:
                        for c in [col+'_raw',col+'_residual']:
                            if c not in joined:joined[c]=np.nan
                        joined['_change']=joined[col+'_residual']-joined[col+'_raw']
                        res=estimate(joined,'_change',M,'mean')
                        valid=np.isfinite(joined['_change']);total=joined.weight.sum()
                        rows.append(self.row(dict(meta,dataset=dataset,version=version),'paired_pairfold_mean_change_'+col,res,
                            dropped_undefined=int((~valid).sum()),dropped_share=float((~valid).mean()) if len(joined) else np.nan,
                            dropped_IPW_share=float(joined.loc[~valid,'weight'].sum()/total) if total else np.nan))
                    for metric in ['reverted_share','KM63','KM_median','events_per_pair_year']:
                        a=self.bank_stats(value,M,metric);b=self.bank_stats(self.aggregate_bank(bank,meta['view'],rels,dataset,'raw',part.year.unique()),M,metric)
                        rows.append(self.row(dict(meta,dataset=dataset,version=version),'residual_minus_raw_'+metric,difference(a,b)))
        self.pred['P7']=self.p7(bank,['same_sector','same_subindustry'],self.years)
        variants=[('P7_same_sector_only',['same_sector'],self.years,'factor_only','reverted_share'),
                  ('P7_without_2004',['same_sector','same_subindustry'],[y for y in self.years if y!=2004],'factor_only','reverted_share'),
                  ('P7_KM63',['same_sector','same_subindustry'],self.years,'factor_only','KM63'),
                  ('P7_factor_alpha',['same_sector','same_subindustry'],self.years,'factor_alpha','reverted_share')]
        for name,control,years,version,metric in variants:self.pred[name]=self.p7(bank,control,years,version,metric)
        # Complete section-5 pooled sensitivity without 2004 (all three relations retained).
        for view in VIEWS:
            for relation in ['all',*RELATIONS]:
                rels=RELATIONS if relation=='all' else [relation];years=[y for y in self.years if y!=2004];M=self.M(years)
                for dataset in ['real','null']:
                    for version in ['raw','factor_only','factor_alpha']:
                        f=frame[frame.year.isin(years)&(frame.dataset==dataset)&(frame.version==version)&frame.relation.isin(rels)].copy()
                        f=f[f.in_random if view=='sample' else f[view]];f['weight']=f.ipw if view=='sample' else 1.
                        meta=dict(view=view,relation=relation,era='all_without_2004',dataset=dataset,version=version)
                        value=self.aggregate_bank(bank,view,rels,dataset,version,years)
                        for metric in ['reverted_share','KM63','KM_median','events_per_pair_year']:rows.append(self.row(meta,metric,self.bank_stats(value,M,metric)))
                        for q in [10,60,120]:rows.append(self.row(meta,f'VR_{q}',estimate(f,f'VR_{q}',M)))
        # Paired share-change DID sensitivity, with per-cell missingness above.
        M=self.M(self.years);s=frame[frame.in_random].copy();s['weight']=s.ipw
        joined=s[s.version=='raw'].merge(s[s.version=='factor_only'],on=['pair','year','dataset','relation'],suffixes=('_raw','_residual'),validate='one_to_one')
        joined['weight']=joined.weight_raw;joined['_change']=joined.reverted_share_residual-joined.reverted_share_raw
        def paired(rels,d):return estimate(joined[joined.relation.isin(rels)&(joined.dataset==d)],'_change',M,'mean')
        self.pred['P7_paired_pairfold_sensitivity']=difference(difference(paired(['cross_sector'],'real'),paired(['same_sector','same_subindustry'],'real')),difference(paired(['cross_sector'],'null'),paired(['same_sector','same_subindustry'],'null')))
        self.p7_dropped=dict(dropped_undefined=int(joined['_change'].isna().sum()),
                            dropped_share=float(joined['_change'].isna().mean()) if len(joined) else np.nan)
        self.tables['factor_residual']=rows
    def predictions(self):
        rows=[];directions={'P1':-1,'P2':1,'P3':1,'P6':-1,'P7':1,'P8':1}
        primary=['P2','P3','P6','P7'];adjusted=holm([summarise(*self.pred.get(p,empty(self.M(self.years))))['p_negative' if directions[p]<0 else 'p_positive'] for p in primary])
        adj=dict(zip(primary,adjusted))
        for name in ['P1','P2','P3','P6','P7','P8',*sorted(k for k in self.pred if k.startswith('P7_') or k=='P8_same_sector_only')]:
            result=self.pred.get(name,empty(self.M(self.years)));row=self.row(dict(prediction=name),'contrast',result)
            direction=directions.get(name,1);family='primary' if name in primary else 'descriptive' if name in directions else 'sensitivity'
            p=row['p_positive' if direction>0 else 'p_negative'];point=row['estimate']
            row.update(family=family,p_raw=p if family=='primary' else np.nan,p_Holm=adj.get(name,np.nan),
                       label='not_estimable' if np.isnan(point) else ('supported' if point*direction>0 and adj[name]<=.05 else 'not_supported') if family=='primary' else 'consistent' if point*direction>0 else 'not_consistent')
            if name=='P7_paired_pairfold_sensitivity':row.update(self.p7_dropped)
            rows.append(row)
        half=self.p4.get('half_life_order',empty(self.M(self.years)));swing=self.p4.get('swing_ge_20',empty(self.M(self.years)))
        valid=not np.isnan(half[0]) and np.isfinite(swing[0]);consistent=valid and half[0]<=10 and swing[0]<.5
        rows.append(dict(prediction='P4',family='descriptive',metric='half_life_and_swing',estimate=half[0],swing_ge20_share=swing[0],display='<1 session' if np.isneginf(half[0]) else None,label='consistent' if consistent else 'not_consistent' if valid else 'not_estimable'))
        buckets=[self.p5.get(('A',i),np.nan) for i in range(5)];lag1=self.p5.get(('lag1','all'),np.nan);lag2=self.p5.get(('lag2','all'),np.nan)
        valid=np.isfinite([*buckets,lag1,lag2]).all();consistent=valid and np.all(np.diff(buckets)>=0) and buckets[-1]>buckets[0] and lag1>=lag2
        rows.append(dict(prediction='P5',family='descriptive',metric='strength_and_horizon',estimate=lag1-lag2,A_bucket_estimates=json.dumps(clean(buckets)),label='consistent' if consistent else 'not_consistent' if valid else 'not_estimable'))
        self.tables['predictions']=rows
        return rows
