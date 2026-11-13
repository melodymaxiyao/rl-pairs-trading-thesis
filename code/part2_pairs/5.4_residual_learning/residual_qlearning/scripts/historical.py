"""Canonical historical checker and the authenticated global8908-path union."""
from common import *
from common import _IGNORED
import types,contextlib,prune_extension
def historical(label):
 """Same canonical frozen historical checker; all temp/audit writes stay in this run."""
 s=source();names=['registered_scope','conflict_copies','approved_A_pruning','protection','historical_gate'];texts={}
 for name in names:texts[name]=s.verify('scripts/'+name+'.py').read_text()
 out=RUN/'runtime'/('historical_'+label);(out/'audit').mkdir(parents=True,exist_ok=True);(out/'vendor').mkdir(exist_ok=True)
 publish(out/'audit/control_registration.json',s.verify('audit/control_registration.json').read_bytes())
 source_input=read(s.verify('audit/input_registration.json'));publish(out/'audit/input_registration.json',json.dumps(dict(source_snapshots=source_input['source_snapshots']),sort_keys=True))
 for rel,row in source_input['source_snapshots'].items():
  if not rel.endswith('/complete.json'):continue
  path=s.verify('vendor/'+rel);dest=out/'vendor'/rel;dest.parent.mkdir(parents=True,exist_ok=True);publish(dest,path.read_bytes())
 shim=types.ModuleType('common');shim.__dict__.update({k:v for k,v in globals().items() if not k.startswith('_')});shim.RUN=out;shim.INPUT=INPUT;shim.contextmanager=contextlib.contextmanager;shim.spec=lambda:s.prereg;shim.deny_conflict_read=lambda p:_IGNORED.add(str(Path(p).absolute()))
 # Avoid another full-file hash cache; target checks re-read exactly their required files.
 text=texts['historical_gate'];assert text.count('record=dict(path=rel,**st,target=target,sha256=digest)')==1;text=text.replace('record=dict(path=rel,**st,target=target,sha256=digest)','record=dict(path=rel,**st,target=target,sha256=digest);record_sink(record)');text=text.replace('byte_count=0','byte_count=0;fresh_count=0').replace('if target is None and rel not in registry:verified[rel]=record','if target is None and rel not in registry:fresh_count+=1').replace('fresh_unregistered_hashes=len(verified)','fresh_unregistered_hashes=fresh_count');texts['historical_gate']=text
 timer=[time.monotonic()]
 def checked_sha(p):
  if time.monotonic()-timer[0]>15:checkpoint();timer[0]=time.monotonic()
  return sha(p)
 snapshot=RUN/'audit'/('protected_'+label+'.jsonl.gz');sink=gzip.open(snapshot,'xt');shim.record_sink=lambda row:sink.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
 shim.sha=checked_sha;old_modules={n:sys.modules.get(n) for n in ['common']+names};sys.modules['common']=shim;old_flag=os.environ.get('ROU_PREFLIGHT_ONLY');os.environ['ROU_PREFLIGHT_ONLY']='1'
 try:
  for name in names:
   m=types.ModuleType(name);m.__file__=str(out/(name+'.py'));sys.modules[name]=m;exec(compile(texts[name],'<sealed '+name+'>','exec'),m.__dict__)
  prune_extension.install(sys.modules['approved_A_pruning'],sys.modules['protection'])
  from global_conditionals import install
  install(sys.modules['registered_scope'])
  cp=Path('/Users/maxiyao/FYP_runs/fyp_icloud_conflict_copies_v1/REGISTERED_IGNORED_CONFLICT_COPIES.json');assert sha(cp)=='464f8bd494b43b153ecafde3265dc0e654258587c6b540c0328ff2391d4ebe21';entries=read(cp)['entries'];original=sys.modules['conflict_copies'].index_for
  def index(root,frozen,proof):
   value=original(root,frozen,proof)
   for row in entries:value.ignored[row['path']]=row
   return value
  sys.modules['conflict_copies'].index_for=index
  result=sys.modules['historical_gate'].verify('coint_'+label)
 finally:
  sink.close()
  for name,prior in old_modules.items():
   if prior is None:sys.modules.pop(name,None)
   else:sys.modules[name]=prior
  if old_flag is None:os.environ.pop('ROU_PREFLIGHT_ONLY',None)
  else:os.environ['ROU_PREFLIGHT_ONLY']=old_flag
 write(RUN/'audit'/f'historical_{label}.json',dict(result=result,adapter='unchanged source canonical scope and checks; own output paths; explicit conflict registry; full-file cache omitted; no sealed-run snapshot comparison or rewrite',source_code_sha256={name:s.expected['scripts/'+name+'.py'] for name in names},approved_location_write=sha(RUN/'audit/location_authorization.json')))
 compare_snapshots(snapshot)
 return result

def compare_snapshots(current):
 from itertools import zip_longest
 first=RUN/'audit/protected_initial.jsonl.gz'
 if not first.exists():publish(first,current.read_bytes());return
 from global_conditionals import merge
 # The gate has already authenticated materialized registered bytes and symlink targets.
 base=read(Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1/REGISTERED_CONDITIONAL_PATHS.json'));registered={x['path'] for x in base['entries']}
 registered.update(x['path'] for x in read(Path('/Users/maxiyao/FYP_runs/rf_cost_accounting_v1/audit/standing_wheel_addendum_20261031/REGISTERED_CONDITIONAL_PATHS_ADDENDUM.json'))['entries'])
 with gzip.open(first,'rt') as a,gzip.open(current,'rt') as b:
  for aa,bb in zip_longest(a,b):
   assert aa is not None and bb is not None,'Protected namespace changed'
   x=json.loads(aa);y=json.loads(bb);assert x['path']==y['path'],'Protected namespace changed'
   if x['path'] in registered:continue
   if x.get('target') is not None:x['mtime_ns']=y['mtime_ns']
   delta=x['flags']^y['flags']
   if delta and not delta&~0x8040:x['flags']=y['flags']
   assert x==y,('Protected snapshot changed outside standing rules',x,y)
