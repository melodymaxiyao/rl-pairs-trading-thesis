"""Standing policies and unchanged sealed artifact checks, with own-run output redirection."""
from common import *
from common import _IGNORED
import types,contextlib
CONDITIONAL=Path('/Users/maxiyao/FYP_runs/fyp_registered_conditionals_v1/REGISTERED_CONDITIONAL_PATHS.json')
CONDITIONAL_SHA='8bc237d1f1daca88249d305b7388f576842e125c0c431986aeb9c895de0357e8'

def location():
 a=read(RUN/'audit/location_authorization.json');assert sha(INPUT/'experiments/15_pair_trading/RUN_LOCATIONS.json')==a['after_sha256']
 before=read(RUN/'audit/run_locations.before.json');after=read(RUN/'audit/run_locations.after.json');copy=json.loads(json.dumps(after));entry=copy['runs'].pop(spec()['task']);assert copy==before and entry==a['addition'];assert RUN.resolve()==RUN and not RUN.resolve().is_relative_to(INPUT.parent)

def verify_inputs(label,original=False):
 checkpoint();s=source();location();assert sha(CONDITIONAL)==CONDITIONAL_SHA
 from standing_wheel import authenticate
 _,_,observations=authenticate();write(event("standing_wheel_"+label),dict(status="PASS",global_exact_paths=8908,observations=observations,source_policy_writes=2,scientific_source_writes=0))
 if original:
  cmd=[str(INPUT/'.venv/bin/python'),'-I','-B',str(SOURCE/'verify_residual_ou.py')];p=subprocess.run(cmd,capture_output=True,text=True)
  publish(RUN/'audit'/f'original_verifier_{label}.log',p.stdout+p.stderr);assert p.returncode==0,('Original artifact verifier failed',p.stdout[-1000:],p.stderr[-1500:])
 # Fixed input provenance also makes resumed accounting outputs deterministic.
 required=[]
 for rel in sorted(s.expected):
  if rel in s.registration['files'] or rel.startswith(('runtime/jobs/RF_','runtime/jobs/RP_','runtime/aggregates/RF.','runtime/aggregates/RP_','runtime/data/calendar_v2/')) or rel in ['outputs/tables/rule_daily.csv.gz','outputs/tables/rule_summary.csv']:
   required.append(rel)
 records={}
 for rel in required:
  path=SOURCE/rel;m=meta(path)
  if original:
   # The unchanged just-completed original verifier checked all artifact/code hashes.
   s.checked[rel]=dict(sha256=s.expected[rel],metadata=m)
  else:s.verify(rel)
  records[rel]=dict(sha256=s.expected[rel],metadata=m)
 input_path=RUN/'audit/input_registration.json'
 if input_path.exists():assert read(input_path)['files']==records,'Selected input metadata changed on resume'
 else:write(input_path,dict(source_completion_sha256=spec()['sealed_completion_sha256'],files=records))
 # Exact sealed input/price declarations. Current byte hashes are never adopted as new seals.
 reg=read(s.verify('audit/input_registration.json'));conditionals={v['path']:v for v in read(CONDITIONAL)['entries']};checked=0;conditional=[];flags=[]
 prefix='experiments/15_pair_trading/fresh_holdout_data_v1/'
 for rel,row in reg['selected'].items():
  if not (rel.startswith(prefix) and (rel.endswith('.csv') or rel.endswith('/complete.json'))):continue
  path=INPUT/rel
  try:st=path.lstat()
  except FileNotFoundError:st=None
  if st is None or st.st_flags&DATALESS:
   assert rel in conditionals;conditional.append(rel);continue
  expected=row['sha256'];assert sha(path)==expected and st.st_size==int(row['size']),('Sealed price/input changed',rel)
  old=reg.get('selected_metadata',{}).get(rel)
  if rel not in conditionals and old:
   current=meta(path);assert all(current[k]==old[k] for k in ['size','mode','mtime_ns','ino']),('Unregistered price metadata changed',rel,old,current)
   delta=old['flags']^current['flags'];assert not delta&~0x8040
   if delta:flags.append(dict(path=rel,old=old['flags'],new=current['flags']))
  checked+=1
 write(event('input_verification_'+label),dict(status='PASS',sealed_completion_sha256=sha(SOURCE/'complete.json'),price_or_membership_files=checked,conditionally_unavailable_provenance=conditional,flag_only_changes=flags,cumulative_trials=552,new_trials=0))

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
 text=texts['historical_gate'];text=text.replace('byte_count=0','byte_count=0;fresh_count=0').replace('if target is None and rel not in registry:verified[rel]=record','if target is None and rel not in registry:fresh_count+=1').replace('fresh_unregistered_hashes=len(verified)','fresh_unregistered_hashes=fresh_count');texts['historical_gate']=text
 timer=[time.monotonic()]
 def checked_sha(p):
  if time.monotonic()-timer[0]>15:checkpoint();timer[0]=time.monotonic()
  return sha(p)
 shim.sha=checked_sha;old_modules={n:sys.modules.get(n) for n in ['common']+names};sys.modules['common']=shim;old_flag=os.environ.get('ROU_PREFLIGHT_ONLY');os.environ['ROU_PREFLIGHT_ONLY']='1'
 try:
  for name in names:
   m=types.ModuleType(name);m.__file__=str(out/(name+'.py'));sys.modules[name]=m;exec(compile(texts[name],'<sealed '+name+'>','exec'),m.__dict__)
  from standing_wheel import install
  install(sys.modules['registered_scope'])
  cp=Path('/Users/maxiyao/FYP_runs/fyp_icloud_conflict_copies_v1/REGISTERED_IGNORED_CONFLICT_COPIES.json');assert sha(cp)=='464f8bd494b43b153ecafde3265dc0e654258587c6b540c0328ff2391d4ebe21';entries=read(cp)['entries'];original=sys.modules['conflict_copies'].index_for
  def index(root,frozen,proof):
   value=original(root,frozen,proof)
   for row in entries:value.ignored[row['path']]=row
   return value
  sys.modules['conflict_copies'].index_for=index
  result=sys.modules['historical_gate'].verify('cost_accounting_'+label)
 finally:
  for name,prior in old_modules.items():
   if prior is None:sys.modules.pop(name,None)
   else:sys.modules[name]=prior
  if old_flag is None:os.environ.pop('ROU_PREFLIGHT_ONLY',None)
  else:os.environ['ROU_PREFLIGHT_ONLY']=old_flag
 write(RUN/'audit'/f'historical_{label}.json',dict(result=result,adapter='unchanged source canonical scope and checks; own output paths; explicit conflict registry; full-file cache omitted; no sealed-run snapshot comparison or rewrite',source_code_sha256={name:s.expected['scripts/'+name+'.py'] for name in names},approved_location_write=sha(RUN/'audit/location_authorization.json')))
 return result
