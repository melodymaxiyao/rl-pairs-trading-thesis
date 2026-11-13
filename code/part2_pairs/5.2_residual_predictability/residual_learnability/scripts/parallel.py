from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

def initialize():
 configure();guard()

def execute(function,jobs,workers):
 assert workers>=1
 if workers==1:
  for job in jobs:yield function(job)
  return
 executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize)
 futures=[]
 try:
  futures=[executor.submit(function,j) for j in jobs]
  for future in as_completed(futures):yield future.result()
 except BaseException:
  for future in futures:future.cancel()
  # A safe interruption never kills a process during an atomic publish.
  executor.shutdown(wait=True,cancel_futures=True);raise
 else:executor.shutdown(wait=True)

def deterministic_fixture(job):
 import numpy as np
 from statistics import all_stats
 random=np.random.default_rng(stable_seed('workers_fixture',job));S=np.cumsum(random.standard_normal(340));d=(S-S.mean())/S.std();value=all_stats(S,d)
 return dict(job=job,values=value)
