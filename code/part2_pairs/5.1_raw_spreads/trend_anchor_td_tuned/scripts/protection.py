from common import *
import subprocess
INPUTS={'trend_anchor_trading_v1':'verify_trend_anchor.py','td_sample_complexity_v1':'verify_td_sample_complexity.py'}
def registered():
 r=read(HERE/'audit/code_registration.json')
 assert sha(HERE/'audit/source_registration.json')==r['source_registration_sha256']
 for k,v in r['files'].items():assert sha(HERE/k)==v,('Prepared code changed',k)
 for k,v in read(HERE/'audit/source_registration.json').items():assert sha(ROOT/k)==v,('Imported source changed',k)
 for n,v in spec()['input_seals'].items():assert sha(PT/n/'complete.json')==v,('Input seal changed',n)
 assert sha(HERE/'preregistration.json')==r['preregistration_sha256']
 assert hashlib.sha256((HERE/'README.md').read_bytes()[:r['README_top_bytes']]).hexdigest()==r['README_top_sha256']
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
def scoped_input(name,current):
 d=PT/name;c=read(d/'complete.json');assert c['status']=='complete'
 assert sha(d/'manifest.json')==c['manifest_sha256']
 manifest=read(d/'manifest.json')['files'];assert isinstance(manifest,dict)
 for k,v in manifest.items():
  if Path(k).name=='.DS_Store':continue
  actual=current['files'].get(str((d/k).relative_to(ROOT)))
  assert (actual.get('sha256')==v if isinstance(v,str) and actual else actual==v),('Input output mismatch',name,k)
 # Mirror the sealed verifier's inventory convention, then also freeze all
 # ignored interrupted-file names against the preparation inventory.
 actual={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json','pipeline.lock','.DS_Store'] and '.pending_' not in p.name}
 assert actual==set(manifest),('File added or missing inside sealed input',name)
 expected=read(d/'audit/protected_before.json')['files'];changed=[]
 for k,v in expected.items():
  if Path(k).name=='.DS_Store':
   if not (ROOT/k).exists() or fingerprint(ROOT/k)!=v:changed.append(k)
  elif k.startswith('experiments/turning_point_detection/'):continue
  else:assert current['files'].get(k)==v,('Historical non-Finder changed/missing',name,k)
 additions=Counter(k.split('/')[0] for k in current['files'] if k not in expected and not k.startswith(str(d.relative_to(ROOT))+'/'))
 return dict(status='PASS',changed_DS_Store_paths=changed,added_by_top_directory=dict(additions))
def preflight():
 registered()
 if (HERE/'audit/preflight_ready.json').exists():return
 print('Full protected hashing and original read-only input verifiers; one process.',flush=True)
 before,finder=snapshot()
 if (HERE/'audit/protected_before.json').exists():assert before==read(HERE/'audit/protected_before.json')
 else:write(HERE/'audit/protected_before.json',before);write(HERE/'audit/finder_before.json',finder)
 info={}
 for name in INPUTS:
  info[name]=scoped_input(name,before);print('Scoped verification',name,info[name],flush=True)
  log=HERE/f'audit/input_{name}_{time.time_ns()}.log'
  with log.open('x') as f:p=subprocess.run([sys.executable,'-B',str(HERE/'scripts/verifier_entry.py'),name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,('Original verifier failed; failure retained',log)
  print('PASS original verifier',name,flush=True)
 write(HERE/'audit/preflight_ready.json',dict(status='PASS',inputs=info))
def finish():
 registered();after,finder=snapshot();assert after==read(HERE/'audit/protected_before.json'),'Protected workspace changed; stop without sealing'
 write(HERE/'audit/protected_after.json',after);write(HERE/'audit/finder_after.json',finder)
 b=read(HERE/'audit/finder_before.json');write(HERE/'audit/finder_changes.json',dict(changed_paths=sorted(k for k in set(b)|set(finder) if b.get(k)!=finder.get(k))))
 for p in (HERE/'audit').glob('events_*.jsonl'):
  for line in p.read_text().splitlines():
   r=json.loads(line)
   if r['operation']=='input_pin':assert fingerprint(ROOT/r['path'])==r['fingerprint'],('Imported/read input changed',r['path'])
 l=ledger();l.update(cumulative_trials=544,production_expected_final_fits=660,source_inputs=spec()['input_seals']);write(HERE/'operation_ledger.json',l)
 write(HERE/'audit.json',dict(status='PASS',protected_match=True,input_seals=spec()['input_seals'],development_only=True,preregistered_before_real_tuned_fit=True))
 files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ['manifest.json','complete.json','pipeline.lock','.DS_Store'] and '.pending_' not in p.name}
 write(HERE/'manifest.json',dict(files=files))
 write(HERE/'complete.json',dict(status='complete',sealed_at_utc=utc(),cumulative_trials=544,**{k+'_sha256':sha(HERE/(k+'.json')) for k in ['manifest','audit','operation_ledger','preregistration']}))
