"""Read-only sealed numerical imports; IO facades point only at this task."""
from common import *
import importlib.util,types
import numpy as np,pandas as pd

def load(p,name):
 key=str(Path(p).relative_to(ROOT));assert sha(p)==read(HERE/'audit/source_registration.json')[key]
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@lru_cache(None)
def setup():
 # None of the sc_* common IO from the input is imported or executed.
 sys.modules['sc_common']=sys.modules['common']
 m=load(SYN/'scripts/sc_sources.py','sc_sources')
 sim=load(SYN/'scripts/sc_simulation.py','sc_simulation')
 kernel=load(SYN/'scripts/sc_kernel.py','sc_kernel')
 # Reuse the sealed binary in read-only mode; never build in an input root.
 import ctypes
 oldlib=ctypes.CDLL(str(SYN/'runtime/tuned_kernel.so'));n=m.native()
 oldlib.sc_batch.argtypes=[ctypes.c_int,ctypes.c_int,n.DP,n.DP,n.DP,ctypes.c_int,n.DP,n.IP,n.DP,n.DP,ctypes.c_int,ctypes.c_double,ctypes.c_int,n.DP,n.LP];oldlib.sc_batch.restype=None
 kernel.kernel=lambda:oldlib
 rules=load(SYN/'scripts/sc_rules.py','sc_rules')
 return types.SimpleNamespace(sources=m,sim=sim,old_kernel=kernel,rules=rules,native=m.native())

def params(k):
 cal=read(note(SYN/'outputs/tables/calibration.json'));assert cal['chosen']['h']==40 and cal['chosen']['w']==.281
 return 40,min(k*.281,.95),119.31662154836822

def training(world,k):
 h,w,sigma=params(k);return setup().sim.training(h,w,sigma,['world_train','gaussian',k,world],600)
def testing(world,k,n=5000):
 h,w,sigma=params(k);return setup().sim.test_batches(h,w,sigma,['world_test','gaussian',k,world],n)
def oracle_batches(k,n=50000):
 h,w,sigma=params(k);return setup().sim.test_batches(h,w,sigma,['oracle','gaussian',k],n)
def seed(j,inner=False):return int(np.random.SeedSequence([20261020,1,j,0 if inner else 1,0]).generate_state(1,dtype=np.uint64)[0])
