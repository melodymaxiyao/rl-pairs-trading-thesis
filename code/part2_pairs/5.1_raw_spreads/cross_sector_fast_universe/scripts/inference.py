"""Year-block persistence bootstrap with exact weighted Spearman ties.

An infinite KM median remains a tied highest rank. Joint duration histograms
avoid repeatedly expanding and sorting millions of pair-fold observations.
"""
import numpy as np

def weighted_spearman(joint):
    joint=np.asarray(joint,float)
    left,right=joint.sum(axis=1),joint.sum(axis=0)
    n=left.sum()
    if n<2:return np.nan
    rx=np.cumsum(left)-.5*left+.5
    ry=np.cumsum(right)-.5*right+.5
    rx-=np.dot(left,rx)/n;ry-=np.dot(right,ry)/n
    vx=np.dot(left,rx*rx);vy=np.dot(right,ry*ry)
    return float(rx@joint@ry/np.sqrt(vx*vy)) if vx>0 and vy>0 else np.nan

def estimate(totals,joint):
    # totals: innerFAST n, innerFAST validation successes, other n, other successes
    ni,si,no,so=totals
    p=si/ni if ni else np.nan;q=so/no if no else np.nan
    return np.array([p,q,p-q,weighted_spearman(joint)])

def histogram_blocks(blocks,years):
    values=sorted({float(v) for b in blocks.values() for key in ['inner','validation'] for v in b[key]})
    if not values:values=[float('inf')]
    totals=[];joints=[]
    for year in years:
        block=blocks.get(year)
        j=np.zeros((len(values),len(values)),np.int64)
        if block is None:
            totals.append([0,0,0,0]);joints.append(j);continue
        inner=np.asarray(block['inner'],float);validation=np.asarray(block['validation'],float)
        selected=np.asarray(block['fast'],bool);success=validation<=63
        totals.append([selected.sum(),(selected&success).sum(),(~selected).sum(),((~selected)&success).sum()])
        np.add.at(j,(np.searchsorted(values,inner),np.searchsorted(values,validation)),1)
        joints.append(j)
    return np.asarray(totals,float),np.asarray(joints,float)

def persistence(blocks,years,seed=20261020,draws=2000):
    totals,joints=histogram_blocks(blocks,years)
    point=estimate(totals.sum(axis=0),joints.sum(axis=0))
    rng=np.random.default_rng(seed)
    bootstrap=np.full((draws,4),np.nan)
    for draw in range(draws):
        indices=rng.integers(0,len(years),len(years))
        weights=np.bincount(indices,minlength=len(years))
        bootstrap[draw]=estimate(weights@totals,np.tensordot(weights,joints,axes=1))
    return point,bootstrap,totals.sum(axis=0)

def interval(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return (float(np.quantile(values,.025)),float(np.quantile(values,.975)),len(values)) if len(values) else (np.nan,np.nan,0)
