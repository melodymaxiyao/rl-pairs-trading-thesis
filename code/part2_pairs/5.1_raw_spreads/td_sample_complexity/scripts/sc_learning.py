"""Original v3 inner selection and tuned final ensembles, checkpointed per fit."""
from sc_common import *
from sc_sources import native,selection
from sc_kernel import fit as tuned_fit
import numpy as np

def arrays_hash(es):
    h=hashlib.sha256()
    for e in es:
        h.update(json.dumps([e['pair'],e['year']]).encode())
        for k in ['z','drift','strong','forward_bps']:
            a=np.ascontiguousarray(e[k]);h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()

def fit_cached(es,val,alpha,gamma,seed,risk,tuned=False,**kw):
    sig=dict(train=arrays_hash(es),val=arrays_hash(val),alpha=alpha,gamma=gamma,seed=int(seed),risk=risk,tuned=tuned,kwargs=kw)
    digest=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();stem='runtime/fits/'+digest;old=cached(stem)
    if old is not None:return old
    started=time.monotonic()
    result=(tuned_fit(es,val,alpha,gamma,seed,risk,True,**kw) if tuned else native().fit(es,val,alpha,gamma,seed,risk,**kw))
    result['meta']['elapsed_seconds']=time.monotonic()-started
    result['signature']=sig;return save(stem,result)

def v3(es,risks,gammas=(.9,.99)):
    seconds=[]
    def adapter(a,b,alpha,gamma,seed,label,risk=None,**kw):
        out=fit_cached(a,b,alpha,gamma,seed,risk,False,**kw);seconds.append(out['meta']['elapsed_seconds']);return out
    out=selection(es,risks,adapter,gammas)
    return dict(q=np.asarray(out['models']).mean(axis=0),models=np.asarray(out['models']),selected=out['selected'],convergence=out['conv'],fit_seconds=sum(seconds))

def tuned(es,risk,seeds=10,gamma=1.):
    tables=[];metas=[]
    for s in range(seeds):
        seed=int(np.random.SeedSequence([20261020,1,s,1,0]).generate_state(1,dtype=np.uint64)[0])
        out=fit_cached(es,[],0.,gamma,seed,risk,True);tables.append(out['q']);metas.append(out['meta'])
    tables=np.asarray(tables);return dict(q=tables.mean(axis=0),models=tables,selected=dict(gamma=gamma,alpha='1/n^.7',seeds=seeds),convergence=metas,fit_seconds=sum(m['elapsed_seconds'] for m in metas))

def table_metrics(model):
    tables=np.asarray(model['models']);greedy=(tables[:,:,1]>tables[:,:,0]).astype(int)
    return dict(abstention=float((greedy[:,::2].sum(axis=1)==0).mean()),exit_everywhere=float((greedy[:,1::2].sum(axis=1)==4).mean()))
