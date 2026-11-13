"""Exact sampled rows, NEW EXP cohort and newly frozen quality selection."""
from aio import *
from store import cached,commit
from sources import anchor_sample,inventory
from market import context
import numpy as np
from scipy.stats import rankdata

def quality_pool(rows,properties):
    valid=[r for r in rows if all(np.isfinite(properties[r['pair']][k]) for k in ['leg_return_corr','spread_sd','quarterly_beta_cv'])]
    scores={}
    if valid:
        a=np.array([[properties[r['pair']]['leg_return_corr'],-properties[r['pair']]['spread_sd'],-properties[r['pair']]['quarterly_beta_cv']] for r in valid])
        ranks=np.column_stack([rankdata(a[:,j],method='average')/len(a) for j in range(3)]);score=ranks.mean(axis=1)
        scores={r['pair']:float(v) for r,v in zip(valid,score)}
    ordered=sorted(valid,key=lambda r:(-scores[r['pair']],r['pair']))[:int(np.ceil(.2*len(valid)))];degrees={};chosen=[]
    for row in ordered:
        a,b=row['leg_A'],row['leg_B']
        if degrees.get(a,0)<2 and degrees.get(b,0)<2:
            chosen.append(row);degrees[a]=degrees.get(a,0)+1;degrees[b]=degrees.get(b,0)+1
    return chosen,scores,len(rows)-len(valid)

def stratified(rows,size,year):
    relations=['cross_sector','same_sector','same_subindustry'];groups=[sorted([r for r in rows if r['sector_relation']==v],key=lambda r:r['pair']) for v in relations]
    counts=np.array([len(g) for g in groups]);quota=counts*min(size,len(rows))/max(1,len(rows));n=np.floor(quota).astype(int)
    for i in np.argsort(-(quota-n),kind='stable')[:min(size,len(rows))-int(n.sum())]:n[i]+=1
    out=[]
    for i,g in enumerate(groups):
        random=np.random.default_rng(np.random.SeedSequence([20261026,year,i]))
        out.extend(g[int(j)] for j in sorted(random.choice(len(g),int(n[i]),replace=False)))
    return sorted(out,key=lambda r:r['pair'])

def prepare(year):
    stem=f'runtime/pools/{year}';old=cached(stem)
    if old is not None:return old
    rows=anchor_sample(year);assert len(rows)==3000
    rand=[dict(r,weight=float(r['weight'])) for r in rows]
    exp=[]
    for r in inventory():
        if r['test_year']!=year:continue
        exp.append(dict(r,leg_A=r['anchor'],leg_B=r['candidate'],weight=1.,sector_relation=r.get('sector_relation','same_subindustry')))
    properties={r['pair']:context(r,year)['quality'] for r in rand}
    qual,scores,missing=quality_pool(rand,properties)
    record('quality_definition',quality_missing_pair_folds=missing,quality_eligible_pair_folds=len(scores),quality_selected_pair_folds=len(qual),test_rows_used=0)
    result=dict(P_RAND=rand,P_EXP=sorted(exp,key=lambda r:r['pair']),P_QUAL=qual,quality_scores=scores,training_subset=[r['pair'] for r in stratified(rand,600,year)])
    return commit(stem,result,test_year=year,test_rows_used=0)
