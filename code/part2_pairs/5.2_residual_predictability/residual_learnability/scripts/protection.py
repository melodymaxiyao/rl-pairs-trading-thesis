"""Sequential protected SHA, scoped historical checks, immutable final seal."""
from common import *
import subprocess
INPUTS={'trend_anchor_td_tuned_v1':'verify_td_tuned_resume_v3.py','trend_anchor_trading_v1':'verify_trend_anchor.py','td_sample_complexity_v1':'verify_td_sample_complexity.py','fresh_holdout_data_v1':'verify_fresh_data.py'}
def registered():
 r=read(HERE/'audit/code_registration.json')
 import importlib.metadata
 assert sys.version.split()[0]==r['python_version'], 'Prepared Python version changed'
 for name,value in r['package_versions'].items():assert importlib.metadata.version(name)==value,('Prepared package version changed',name)
 if (HERE/'addendum_1_design_decisions.json').exists():
  receipt=read(HERE/'audit/addendum_1_registration.json');assert sha(HERE/'addendum_1_design_decisions.json')==receipt['sha256'], 'Design addendum not registered'
  assert receipt['base_preregistration_sha256']==sha(HERE/'preregistration.json')
 for k,v in r['files'].items():assert sha(HERE/k)==v,('Prepared code changed',k)
 assert hashlib.sha256((HERE/'README.md').read_bytes()[:r['README_top_bytes']]).hexdigest()==r['README_top_sha256']
 for key,value in read(HERE/'audit/source_registration.json').items():assert sha(ROOT/key)==value,('Imported source changed',key)
 for name,v in spec()['input_seals'].items():assert sha(PT/name/'complete.json')==v,('Input seal changed',name)
 pass # Imported bytes are registered; all other inputs are checked by the full protected snapshot.
 assert sha(ROOT/spec()['price_input']['path'])==spec()['price_input']['sha256'], 'Transitive price input seal changed'
 return r

def snapshot():
 files={};dirs=[];finder={};last=time.monotonic()
 for parent,children,names in os.walk(ROOT,followlinks=False):
  parent=Path(parent);children[:]=sorted(n for n in children if n not in ['.git','.venv','node_modules'] and parent/n not in [HERE,ROOT/'experiments/turning_point_detection'])
  for n in children.copy():
   p=parent/n
   if p.is_symlink():files[str(p.relative_to(ROOT))]=fingerprint(p);children.remove(n)
   else:dirs.append(str(p.relative_to(ROOT)))
  for n in sorted(names):
   p=parent/n;(finder if n=='.DS_Store' else files)[str(p.relative_to(ROOT))]=fingerprint(p)
   if time.monotonic()-last>30:print('Protected SHA files',len(files),flush=True);last=time.monotonic()
 return dict(files=files,directories=sorted(dirs)),finder

def scope_compare(before,after,own_exclusion=None):
 expected=before['files'];actual=after['files'];changed=[]
 for k,v in expected.items():
  if Path(k).name=='.DS_Store' or k.startswith('experiments/turning_point_detection/'):continue
  assert actual.get(k)==v,('Historical/protected file changed or missing',k)
 assert set(before.get('directories',[]))<=set(after.get('directories',[])), 'Protected directory missing'
 # Only directories already sealed at the historical snapshot can prohibit
 # later additions. A new later experiment outside those inputs is permitted.
 sealed={str(Path(k).parent) for k in expected if Path(k).name=='complete.json'}
 if own_exclusion:sealed.discard(own_exclusion)
 additions=sorted(k for k in actual if k not in expected and not (own_exclusion and k.startswith(own_exclusion+'/')))
 for k in additions:assert not any(k.startswith(p+'/') for p in sealed),('Added inside historical sealed input',k)
 return dict(status='PASS',original_files_exact=sum(Path(k).name!='.DS_Store' for k in expected),added_by_top_directory=dict(Counter(k.split('/')[0] for k in additions)),added_paths=additions)

def input_check(name,current,finder):
 d=PT/name;c=read(d/'complete.json');assert c['status']=='complete';m=d/c.get('manifest_file','manifest.json');assert sha(m)==c['manifest_sha256']
 mf=read(m)['files']
 if isinstance(mf,list):mf={v['path']:dict(sha256=v['sha256'],size=v['bytes'],mtime_ns=v['mtime_ns']) for v in mf}
 actual={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p!=m and p.name not in ['complete.json','pipeline.lock','.DS_Store']}
 assert actual==set(mf),('Input inventory mismatch',name)
 for k,v in mf.items():
  a=current['files'].get(str((d/k).relative_to(ROOT)));assert a is not None
  assert (a.get('sha256')==v if isinstance(v,str) else a==v),('Sealed manifest mismatch',name,k)
 expected=read(d/'audit/protected_before.json');result=scope_compare(expected,current,str(d.relative_to(ROOT)))
 historical_finder={k:v for k,v in expected['files'].items() if Path(k).name=='.DS_Store'}
 finder_path=d/'audit/finder_before.json'
 if finder_path.exists():historical_finder.update(read(finder_path))
 result['changed_DS_Store_paths']=sorted(k for k,v in historical_finder.items() if finder.get(k)!=v)
 return result

def preflight():
 registered()
 assert not spec().get('pending_decisions'), 'Await frozen user decisions; no main computation permitted' 
 # A retry rehashes the current workspace and reruns each verifier; it never
 # trusts a stale preflight receipt after an interruption.
 before,finder=snapshot();p=HERE/'audit/protected_before.json'
 if p.exists():scope_compare(read(p),before)
 else:write(p,before);write(HERE/'audit/finder_before.json',finder)
 info={}
 for name in INPUTS:
  info[name]=input_check(name,before,finder);print('Historical non-Finder exact:',name,'Finder changes',info[name]['changed_DS_Store_paths'],'additions',info[name]['added_by_top_directory'],flush=True)
  log=HERE/f'audit/input_{name}_{time.time_ns()}.log'
  with log.open('x') as f:r=subprocess.run([sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,('Original verifier failed; log preserved',log)
  print('PASS original verifier',name,flush=True)
 write(HERE/f'audit/preflight_{time.time_ns()}.json',info)

@contextmanager
def locks():
 f=(HERE/'pipeline.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Never create or alter another project's lock; probe existing files only.
 try:
  for p in PT.glob('*/pipeline.lock'):
   if p.parent==HERE:continue
   with p.open('r') as q:
    try:fcntl.flock(q,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(q,fcntl.LOCK_UN)
    except BlockingIOError:raise RuntimeError('Another workspace pipeline is active: '+str(p))
  yield
 finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()

def seal():
 registered();attempt=HERE/f'audit/seal_{time.time_ns()}';attempt.mkdir()
 after,finder=snapshot();write(attempt/'protected_after.json',after);write(attempt/'finder_after.json',finder)
 result=scope_compare(read(HERE/'audit/protected_before.json'),after)
 assert after==read(HERE/'audit/protected_before.json'), 'Current protected workspace changed; preserve failure and stop without sealing'
 # Also verify all three complete manifests after computation.
 for name in INPUTS:input_check(name,after,finder)
 pass # Full before/after equality covers every protected source.
 bf=read(HERE/'audit/finder_before.json');finder_changes=sorted(k for k in set(bf)|set(finder) if bf.get(k)!=finder.get(k))
 l=ledger();assert l['downloads']==l['rl_fits']==0 and l['total']['trading_evaluations']==1
 write(attempt/'operation_ledger.json',l);write(attempt/'audit.json',dict(status='PASS',protected_scope=result,changed_DS_Store_paths=finder_changes,statistics_only=True,new_real_strategy_trials=0,rl_fits=0,trading_evaluations=1,input_seals=spec()['input_seals']))
 # Stable root pointers allow an interrupted seal to retry without overwrites.
 publish(HERE/'audit.json','{"authoritative_file": "complete.json:audit_file", "status": "see authoritative audit"}\n')
 publish(HERE/'operation_ledger.json','{"authoritative_file": "complete.json:operation_ledger_file", "rl_fits": 0, "downloads": 0}\n')
 manifest=attempt/'manifest.json'
 files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p!=manifest and p.name not in ['complete.json','pipeline.lock','.DS_Store'] and '.pending_' not in p.name}
 write(manifest,dict(files=files))
 c=dict(status='complete',verifier='verify_residual_learnability.py',statistics_only=True,new_real_strategy_trials=0,rl_fits=0,trading_evaluations=1,input_seals=spec()['input_seals'],manifest_file=str(manifest.relative_to(HERE)),manifest_sha256=sha(manifest),audit_file=str((attempt/'audit.json').relative_to(HERE)),audit_sha256=sha(attempt/'audit.json'),operation_ledger_file=str((attempt/'operation_ledger.json').relative_to(HERE)),operation_ledger_sha256=sha(attempt/'operation_ledger.json'),after_file=str((attempt/'protected_after.json').relative_to(HERE)),finder_after_file=str((attempt/'finder_after.json').relative_to(HERE)),preregistration_sha256=sha(HERE/'preregistration.json'))
 write(HERE/'complete.json',c)
