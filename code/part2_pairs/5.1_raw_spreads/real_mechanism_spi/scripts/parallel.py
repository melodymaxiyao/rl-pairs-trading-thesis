"""Per-job seeds and checkpoint-based cooperative interruption."""
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp,signal
def initialize():
 signal.signal(signal.SIGINT,signal.SIG_IGN);guard()
def run(tasks,fn,workers,first_completed=None):
 executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize)
 futures={executor.submit(fn,task):j for j,task in enumerate(tasks)};results={}
 try:
  for future in as_completed(futures):
   result=future.result();results[futures[future]]=result
   if len(results)==1 and first_completed:first_completed(result)
 except BaseException:
  stop=Path(os.environ.get('RMS_STOP_FILE',str(HERE/f'runtime/tmp/stop_{time.time_ns()}.json')))
  if not stop.exists():write(stop,dict(reason='controller interruption or worker failure'))
  for future in futures:future.cancel()
  # Running workers finish their current native call, save it, and stop at the
  # next guard. Joining prevents a resumed command racing an old worker.
  executor.shutdown(wait=True,cancel_futures=True);raise
 else:executor.shutdown(wait=True)
 return [results[i] for i in range(len(tasks))]
def fixture_worker(job):
 from fixtures import tiny
 from real_sources import setup
 pool,y,j=job;out=setup().kernel.fit(tiny(pool,y),[],'C01',1.,stable_seed('fixture',pool,y,j),dict(H=12,stop=4.,lock=True),maxepochs=20,early=False)
 return hashlib.sha256(out['q'].tobytes()+out['visits'].tobytes()+out['diagnostics'].tobytes()).hexdigest()
