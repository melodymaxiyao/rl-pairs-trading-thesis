"""Journaled bounded append of new, unsealed output; no duplicated full dataset."""
from common import *
import csv,io
PARAM_COLUMNS=['definition','world','replicate','series','test_year','date','weight','b','b_se','kappa','half_life','m','sigma_epsilon','sigma_eq','s','increment_predictive_r2','valid','reverting','fast','no_reversion','nonpositive_b','significantly_below_one']
def param_shard(job,rows):
 directory=RUN/'runtime/shards';directory.mkdir(exist_ok=True);p=directory/(job+'.csv.gz.pending')
 if p.exists():
  assert not (RUN/f'runtime/jobs/{job}.ready.json').exists(),'Completed job transport must be resumed, not overwritten'
  write(event_path('abandoned_uncommitted_shard'),dict(path=str(p.relative_to(RUN)),sha256=sha(p),size=p.stat().st_size,job=job,reason='Uncommitted interrupted fit; no completed job receipt; remove only this temporary transport'))
  p.unlink()
 with p.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as zipped:
   import io
   with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
    w=csv.DictWriter(text,fieldnames=PARAM_COLUMNS,extrasaction='ignore');w.writerows(rows)
 return dict(path=str(p.relative_to(RUN)),sha256=sha(p),size=p.stat().st_size)
class ParamStream:
 def __init__(self):
  self.path=RUN/'runtime/streams/ou_params.csv.gz';self.path.parent.mkdir(exist_ok=True);self.receipts=RUN/'runtime/streams/receipts';self.receipts.mkdir(exist_ok=True)
  sio=io.StringIO();csv.writer(sio,lineterminator='\n').writerow(PARAM_COLUMNS);header=gzip.compress(sio.getvalue().encode(),mtime=0);self.header=header
  if not self.path.exists():publish(self.path,header)
  with self.path.open('rb') as f:assert f.read(len(header))==header
  self.done={};self.end=len(header)
  for p in sorted(self.receipts.glob('*.json')):
   r=json.loads(p.read_text());assert r['start']==self.end
   with self.path.open('rb') as f:f.seek(r['start']);data=f.read(r['end']-r['start'])
   assert hashlib.sha256(data).hexdigest()==r['sha256'],('Committed parameter stream changed',r['job'])
   self.end=r['end'];self.done[r['job']]=r
  if self.path.stat().st_size>self.end:
   # Roll back only our unpublished append tail, preserving every committed byte.
   write(event_path('uncommitted_stream_tail'),dict(path=str(self.path.relative_to(RUN)),before=self.path.stat().st_size,committed=self.end,action='truncate only incomplete uncommitted append'))
   with self.path.open('r+b') as f:f.truncate(self.end);f.flush();os.fsync(f.fileno())
  assert self.path.stat().st_size==self.end
 def append(self,job,shard):
  if job in self.done:return self.done[job]
  checkpoint_guard();p=inside(RUN/shard['path']);assert p.name.endswith('.pending') and sha(p)==shard['sha256'] and p.stat().st_size==shard['size']
  if shutil.disk_usage(RUN).free-p.stat().st_size<5*2**30:raise Pause('Output append would cross5GiB; completed job and shard retained')
  start=self.end
  with self.path.open('ab') as f,p.open('rb') as source:shutil.copyfileobj(source,f,1024*1024);f.flush();os.fsync(f.fileno())
  record=dict(job=job,start=start,end=self.path.stat().st_size,sha256=shard['sha256']);receipt=self.receipts/f'{len(self.done):06d}.json';write(receipt,record);self.end=record['end'];self.done[job]=record
  # The immutable receipt and committed stream now replace this temporary transport.
  p.unlink();return record
 def publish(self):
  target=RUN/'outputs/tables/ou_params.csv.gz';target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.stat().st_ino==self.path.stat().st_ino
  else:os.link(self.path,target)
  return sha(target)
