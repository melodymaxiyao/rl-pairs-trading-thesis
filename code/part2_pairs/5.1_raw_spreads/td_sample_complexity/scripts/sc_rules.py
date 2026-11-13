"""Original B3 execution grid plus the two frozen pair-cluster selectors."""
from sc_common import *
from sc_kernel import descriptor,batch
import numpy as np
GRID=[descriptor()]+[descriptor(entry=e,exit=x) for e in [1.5,2.,2.5] for x in [0.,.5]]
DEFAULT=3

def clustered(values,pairs):
    x=np.asarray(values,float);pairs=np.asarray(pairs);names=np.unique(pairs);G=len(names)
    assert G>=2 and len(x)==len(pairs) and np.isfinite(x).all()
    mean=float(x.mean());sums=np.array([(x[pairs==g]-mean).sum() for g in names])
    se=float(np.sqrt(G/(G-1)*np.sum(sums*sums)/len(x)**2));return mean,se

def select_matrix(values,pairs):
    assert np.asarray(values).shape[0]==7
    stats=[clustered(v,pairs) for v in values]
    alternatives=[i for i in range(7) if i!=DEFAULT]
    diffs={i:clustered(values[i]-values[DEFAULT],pairs) for i in alternatives}
    best=max(alternatives,key=lambda i:diffs[i][0]);mu,se=diffs[best]
    ratio=mu/se if se>0 else np.inf if mu>0 else -np.inf if mu<0 else 0.
    return dict(B3=max(range(7),key=lambda i:stats[i][0]),B3_shrunk=best if ratio>1.645 else DEFAULT,B3_lcb=max(range(7),key=lambda i:stats[i][0]-1.645*stats[i][1]),statistics=stats,differences=diffs,shrunk_best_alternative=best,shrunk_t=float(ratio))

def choose(es,risk):
    assert all(e['window']=='train' for e in es)
    values,_=batch(es,GRID,risk);out=select_matrix(values[:,:,0],[e['pair'] for e in es]);count('B3_selections',3);return out
