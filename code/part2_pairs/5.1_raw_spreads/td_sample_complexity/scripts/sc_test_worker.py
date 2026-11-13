def worker(seed):
    import hashlib
    from sc_simulation import episodes
    from sc_kernel import fit
    es=[e for e in episodes(10,.1,119.,['worker_fixture'],seed) if e['window']=='train']
    out=fit(es,[],.05,.99,seed,dict(H=20,stop=4.,lock=True),False,maxepochs=20,early=False)
    h=hashlib.sha256()
    for k in ['q','curves','snapshots','visits']:h.update(out[k].tobytes())
    return h.hexdigest()
