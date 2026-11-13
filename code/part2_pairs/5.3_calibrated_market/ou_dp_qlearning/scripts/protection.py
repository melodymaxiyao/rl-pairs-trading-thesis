"""Strict before/after gates. Unavailable provenance is never silently waived."""
from common import *
import subprocess,csv,ast,runpy
PYTHON=str(INPUT/'.venv/bin/python');PT=INPUT/'experiments/15_pair_trading';COMPACT=PT/'storage_compaction_v1';PRUNE=PT/'tav1_runtime_prune_v1'
VERIFIERS={'td_mechanism_spi_v1':'verify_td_mechanism_spi.py','trend_anchor_td_tuned_v1':'verify_td_tuned_resume_v3.py'}
def environment():
 import importlib.metadata as md,platform
 assert sys.executable==PYTHON,('Wrong interpreter',sys.executable,PYTHON)
 expected={}
 for line in (RUN/'audit/pip_freeze_all.txt').read_text().splitlines():
  assert '==' in line,('Unsupported sealed requirement',line)
  name,version=line.split('==',1);expected[name.lower().replace('_','-')]=version
 actual={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions() if d.metadata['Name']}
 differences={k:dict(expected=expected.get(k),actual=actual.get(k)) for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)}
 assert not differences,('Environment differs; no install allowed',differences)
 assert platform.python_version()=='3.13.2' and actual['pandas']=='3.0.5' and actual['numpy']=='2.5.2'
 evidence=json.loads((RUN/'audit/sealed_environment_records.json').read_text())
 for r in evidence['records'].values():
  for name,version in r['packages'].items():assert actual[name.lower().replace('_','-')]==version,('Sealed named environment mismatch',name)
 assert evidence['restored_full_freeze_sha256']==sha(RUN/'audit/pip_freeze_all.txt')
 return dict(status='PASS',sys_executable=sys.executable,python=platform.python_version(),packages=len(actual),pandas=actual['pandas'],numpy=actual['numpy'],sealed_freeze_sha256=sha(RUN/'audit/pip_freeze_all.txt'),named_environment_records=len(evidence['records']))
def metadata(p):
 s=p.lstat();return dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,flags=s.st_flags,ino=s.st_ino)
def selected_inputs():
 reg=json.loads((RUN/'audit/input_registration.json').read_text());out={}
 for rel,r in reg['source_snapshots'].items():
  p=RUN/'vendor'/rel;assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],('Source snapshot changed',rel);out[rel]=r['sha256']
 r=reg['fixture'];p=INPUT/r['path'];st=metadata(p)
 assert all(st[k]==r['metadata'][k] for k in st),('Fixture metadata drift; stop, no automatic repair',str(p),st,r['metadata'])
 assert not st['flags']&DATALESS and sha(p)==r['sha256'];out[r['path']]=r['sha256'];return out
def availability():
 from provenance import snapshot
 states=snapshot();conditional=[r for r in states.values() if r['content_hash_excluded']]
 return dict(status='PASS_WITH_REGISTERED_CONDITIONALS' if conditional else 'AVAILABLE_REQUIRES_FULL_VERIFICATION',issues=[],files=states,conditionally_unavailable_provenance=conditional,source_hydration=0,new_exclusions=len(conditional),new_exclusions_authorization='standing registered conditional rule, 2026-10-29',computation_inputs_excluded=0)
def registry_rows():
 registry=PT/'PRUNE_REGISTRY.json';pins=json.loads((RUN/'audit/control_registration.json').read_text());assert sha(registry)==pins['files'][str(registry.relative_to(INPUT))]
 rows={}
 for e in json.loads(registry.read_text())['entries']:
  if e['status']!='active_registered_pruning':continue
  for name,key in [('manifest','manifest_sha256'),('recipe','recipe_sha256')]:assert sha(INPUT/e[name])==e[key]
  assert sha(INPUT/e['audit_directory']/'audit.json')==e['audit_sha256']
  with gzip.open(INPUT/e['manifest'],'rt',newline='') as f:rr=list(csv.DictReader(f))
  assert len(rr)==e['files'] and len({r['path'] for r in rr})==len(rr)
  assert hashlib.sha256(('\n'.join(sorted(r['path'] for r in rr))+'\n').encode()).hexdigest()==e['paths_sha256']
  for r in rr:
   rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts and r['path'] not in rows;rows[r['path']]=r
 return rows
def canonical_scope():
 """Execute only the original frozen exclusion predicate and constants."""
 p=COMPACT/'storage_compaction.py';pins=json.loads((RUN/'audit/control_registration.json').read_text());assert sha(p)==pins['files'][str(p.relative_to(INPUT))]
 tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='excluded');assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXCLUDED' for t in n.targets));ns=dict(ROOT=INPUT,HERE=COMPACT,EXCLUDED=ast.literal_eval(assign.value));exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);return ns['excluded']
def walk():
 excluded=canonical_scope()
 for parent,dirs,names in os.walk(INPUT,followlinks=False):
  parent=Path(parent);assert not parent.lstat().st_flags&DATALESS,('Dataless directory; do not expand',str(parent));dirs[:]=sorted(d for d in dirs if not excluded(parent/d))
  for d in list(dirs):
   if (parent/d).is_symlink():names.append(d);dirs.remove(d)
   elif (parent/d).lstat().st_flags&DATALESS:raise RuntimeError('Unapproved dataless directory: '+str(parent/d))
  for n in sorted(names):
   p=parent/n
   if not excluded(p):yield p
def conditionals():
 reg=json.loads((COMPACT/'addendum_2_registration.json').read_text())
 for name,key in [('addendum_2_scope.md','sha256'),('addendum_2_metadata_baseline.json','metadata_baseline_sha256'),('hydration_authorization.json','hydration_authorization_sha256')]:assert sha(COMPACT/name)==reg[key]
 base=json.loads((COMPACT/'addendum_2_metadata_baseline.json').read_text());rms=PT/'real_mechanism_spi_v1';out=json.loads((rms/'addendum_005_gitignore_hydration_outcome.json').read_text());assert sha(rms/'addendum_004_gitignore_hydration_authorization.json')==out['authorization_sha256'];p=rms/out['baseline_file'];assert sha(p)==out['baseline_sha256'];extra=json.loads(p.read_text());assert set(extra)=={'.gitignore','experiments/15_pair_trading/economic_peer_residual_screen_rolling/.gitignore'} and not base.keys()&extra.keys();base.update(extra);assert len(base)==30;return base
def log_reset(rel,before,after):
 p=RUN/'symlink_timestamp_resets.csv';exists=p.exists()
 with p.open('a',newline='') as f:
  w=csv.writer(f)
  if not exists:w.writerow(['path','frozen_mtime_ns','observed_mtime_ns','target_content_check'])
  w.writerow([rel,before,after,'available target members verified; exact registered unavailable members recorded conditionally']);f.flush();os.fsync(f.fileno())
def target_file_check(path,rel,row,registered):
 entry=registered.get(rel);is_conditional=entry is not None and ('physical' in entry or entry.get('content_hash_excluded',False))
 if not path.exists():assert entry is not None,('Unregistered missing symlink target member',rel);return
 st=metadata(path)
 if st['flags']&DATALESS:assert is_conditional,('Unregistered dataless symlink target',rel);return
 expected_sha=entry.get('sealed_sha256') if is_conditional else row['sha256'];expected_sha=expected_sha or row['sha256'];expected_size=int(entry.get('size',row['size'])) if is_conditional else int(row['size'])
 cached_value=globals().get('_TARGET_HASH_CACHE',{}).get(rel) if globals().get('_TARGET_CACHE_ROOT')==INPUT else None
 cached_metadata=cached_value.get('metadata',{k:cached_value[k] for k in st if k in cached_value}) if cached_value else None
 if not is_conditional:
  assert all(st[k]==int(row[k]) for k in ['size','mode','mtime_ns']),('Unregistered target metadata changed',rel)
  if cached_value:assert cached_metadata==st,('Unregistered target changed during gate',rel)
 digest=cached_value['sha256'] if cached_value and cached_metadata==st else sha(path)
 assert st['size']==expected_size and digest==expected_sha,('Symlink target content or size changed',rel)
def target_check(p,expected,registered):
 target=p.resolve(strict=False)
 if target.is_relative_to(INPUT):
  rel=str(target.relative_to(INPUT))
  if not target.exists():assert rel in registered,('Unregistered missing symlink target',rel);return
  if target.is_file():assert rel in expected;target_file_check(target,rel,expected[rel],registered)
  else:
   key=(str(INPUT),id(expected),rel);done=globals().setdefault('_CHECKED_TARGET_DIRECTORIES',set())
   if key in done:return
   members={k:r for k,r in expected.items() if k.startswith(rel+'/')}
   if not members:
    archive=PT/'fyp_relocation_v1';c=json.loads((archive/'complete.json').read_text());a=json.loads((archive/'audit.json').read_text());assert sha(archive/'audit.json')==c['audit_sha256'];manifest=archive/'tree_manifest.csv.gz';assert sha(manifest)==a['artifact_sha256'][manifest.name]
    dirs=globals().get('_FROZEN_DIRECTORY_ROWS')
    if dirs is None:
     dirs={};children={}
     with gzip.open(manifest,'rt',newline='') as f:
      for v in csv.DictReader(f):
       if v['kind']=='directory':dirs[v['path']]=int(v['mode'])
       parent=v['path'].rpartition('/')[0];children[parent]=children.get(parent,0)+1
     globals()['_FROZEN_DIRECTORY_ROWS']=dirs;globals()['_FROZEN_DIRECTORY_CHILD_COUNTS']=children
    assert rel in dirs and stat.S_IMODE(target.stat().st_mode)==dirs[rel] and globals()['_FROZEN_DIRECTORY_CHILD_COUNTS'].get(rel,0)==0,('No authenticated empty-directory baseline',rel)
    assert not target.lstat().st_flags&DATALESS,('Dataless target directory prohibited',rel)
    assert not list(target.iterdir()),('Frozen empty target directory has members',rel)
    done.add(key);return
   for k,r in members.items():
    q=INPUT/k
    if not q.exists():assert k in registered;continue
    if r['kind']=='symlink':assert q.is_symlink() and os.readlink(q)==r['target'] and stat.S_IMODE(q.lstat().st_mode)==int(r['mode'])
    else:target_file_check(q,k,r,registered)
   done.add(key)
 else:
  external_target_check(target)
def external_target_check(target):
 target=Path(target);key=str(target);cache=globals().setdefault('_CHECKED_EXTERNAL_TARGETS',{})
 if key in cache:return cache[key]
 archive=PT/'fyp_relocation_v1';c=json.loads((archive/'complete.json').read_text());a=json.loads((archive/'audit.json').read_text());assert sha(archive/'audit.json')==c['audit_sha256'];p=archive/'external_symlink_target_hashes.json';assert sha(p)==a['artifact_sha256'][p.name];old=json.loads(p.read_text());assert key in old,('External target lacks frozen digest',key);r=old[key]
 if r['kind']=='file_content_sha256':value=dict(sha256=sha(target),file_bytes=target.stat().st_size)
 else:
  assert r['kind']=='directory_member_content_sha256';h=hashlib.sha256();entries=0;size=0
  for parent,dirs,names in os.walk(target,followlinks=False):
   parent=Path(parent);assert not parent.lstat().st_flags&DATALESS;dirs.sort();names.sort()
   for name in dirs+names:
    q=parent/name;kind='symlink' if q.is_symlink() else 'directory' if q.is_dir() else 'file'
    digest=hashlib.sha256(os.fsencode(os.readlink(q))).hexdigest() if kind=='symlink' else sha(q) if kind=='file' else ''
    if kind=='file':size+=q.stat().st_size
    h.update(json.dumps([str(q.relative_to(target)),kind,digest],separators=(',',':')).encode()+b'\n');entries+=1
   dirs[:]=[d for d in dirs if not (parent/d).is_symlink()]
  value=dict(sha256=h.hexdigest(),entries=entries,file_bytes=size)
 assert value['sha256']==r['sha256'] and ('entries' not in r or value['entries']==r['entries']),('External target content differs',key,value,r)
 cache[key]=value;return value
def additional_frozen_symlinks(expected,registered):
 archive=PT/'fyp_relocation_v1';c=json.loads((archive/'complete.json').read_text());a=json.loads((archive/'audit.json').read_text());assert sha(archive/'audit.json')==c['audit_sha256'];assert sha(archive/'symlinks.json')==a['artifact_sha256']['symlinks.json'];resets=[];counted=0;extra_inventory=None
 for r in json.loads((archive/'symlinks.json').read_text()):
  if r['path'] in expected:continue
  p=INPUT/r['path'];s=p.lstat();resolved=p.resolve(strict=False);conditional_target=resolved.is_relative_to(INPUT) and str(resolved.relative_to(INPUT)) in registered
  assert p.is_symlink() and os.readlink(p)==r['target'] and stat.S_IMODE(s.st_mode)==r['lstat']['mode'] and (p.exists()==r['target_exists_before'] or conditional_target),('Additional frozen symlink changed',r['path']);counted+=1
  changed=s.st_mtime_ns!=r['lstat']['mtime_ns']
  if p.exists() and not p.resolve().is_relative_to(INPUT):
   assert r['target_hash_kind']=='file_content_sha256' and sha(p.resolve())==r['target_hash'],('External interpreter/link target changed',r['path'])
  if changed:
   assert p.exists() or conditional_target,('Cannot accept timestamp drift with unregistered unavailable target',r['path'])
   if p.resolve().is_relative_to(INPUT):
    if extra_inventory is None:
     manifest=archive/'tree_manifest.csv.gz';assert sha(manifest)==a['artifact_sha256'][manifest.name]
     with gzip.open(manifest,'rt',newline='') as f:extra_inventory={v['path']:v for v in csv.DictReader(f) if v['kind'] in ['file','symlink']}
    target_check(p,extra_inventory,registered)
   resets.append((r['path'],r['lstat']['mtime_ns'],s.st_mtime_ns))
 return counted,resets
def historical(label):
 from historical_gate import verify
 return verify(label)
def original_verifiers(label):
 from provenance import snapshot
 states=snapshot()
 results=[]
 for name,script in VERIFIERS.items():
  rows=[states[f'experiments/15_pair_trading/{name}/{f}'] for f in ['complete.json',script]]
  conditional=[r for r in rows if r['content_hash_excluded']]
  for rel,r in globals().get('_REGISTERED_CURRENT_STATES',{}).items():
   if rel.startswith(f'experiments/15_pair_trading/{name}/') and any(a['physical_path']==str(INPUT/rel) and a['content_hash_excluded'] for a in r['physical']):
    if not any(a['path']==rel for a in conditional):conditional.append(dict(path=rel,status='REGISTERED_UNAVAILABLE_OR_DATALESS',sealed_sha256=r['sealed_sha256'],content_hash_excluded=True))
  if conditional:
   log=RUN/f'audit/original_{label}_{name}_{time.time_ns()}.json';write(log,dict(status='CONDITIONALLY_UNAVAILABLE_PROVENANCE',name=name,unavailable=conditional,original_verifier_executed=False,full_fresh_upstream_verifier_pass_claimed=False,computation_inputs_excluded=0,authorization='standing registered conditional rule, 2026-10-29'))
   results.append(dict(name=name,status='CONDITIONALLY_UNAVAILABLE_PROVENANCE',original_verifier_executed=False,log=str(log.relative_to(RUN)),sha256=sha(log)));continue
  p=PT/name/script;rel=str(p.relative_to(INPUT));r=json.loads((RUN/'audit/input_registration.json').read_text())['source_snapshots'][rel];assert sha(p)==r['sha256'];assert sha(PT/name/'complete.json')==spec()['input_seals'][name]
  log=RUN/f'audit/original_{label}_{name}_{time.time_ns()}.log'
  env=os.environ.copy();ignored=globals().get('_IGNORED_CONFLICT_COPIES_PROOF')
  if ignored:env.update(FYP_IGNORED_CONFLICT_PROOF=str(RUN/ignored['file']),FYP_IGNORED_CONFLICT_PROOF_SHA256=ignored['sha256'])
  with log.open('x') as f:process=subprocess.run([PYTHON,'-I','-B',str(RUN/'scripts/readonly_entry.py'),str(p)],cwd=RUN,stdout=f,stderr=subprocess.STDOUT,env=env)
  assert process.returncode==0,('Original verifier failed; sealed verifier unchanged',name,str(log))
  for line in log.read_text().splitlines():
   if line.startswith('SYMLINK_TIMESTAMP_ACCEPTED '):
    reset=json.loads(line.split(' ',1)[1]);log_reset(reset['path'],reset['frozen_mtime_ns'],reset['observed_mtime_ns'])
  results.append(dict(name=name,status='PASS',original_verifier_executed=True,log=str(log.relative_to(RUN)),sha256=sha(log)))
 return results
def no_other_task():
 from multiprocessing import resource_tracker
 from process_guard import own_resource_tracker,process_rows
 conflicts=[];helpers=[];owner=os.getpid();registered_pid=resource_tracker._resource_tracker._pid
 output=subprocess.run(['/bin/ps','-ww','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True).stdout
 for pid,ppid,cmd,line in process_rows(output):
  if pid in {owner,os.getppid()}:continue
  if own_resource_tracker(pid,ppid,cmd,owner,registered_pid,PYTHON):
   helpers.append(dict(pid=pid,ppid=ppid,command=cmd,registered_tracker_pid=registered_pid,kind='own multiprocessing.resource_tracker',ownership='exact registered PID and direct parent PID and stdlib command'));continue
  if str(RUN) in cmd:continue
  if 'python' not in cmd.lower() or not (str(INPUT) in cmd or str(RUN.parent) in cmd or 'experiments/15_pair_trading/' in cmd):continue
  if '/python-env-tools/' in cmd:continue
  if cmd in {str(INPUT/'.venv/bin'/s) for s in ['python','python3','python3.13']}:
   status=subprocess.run(['/bin/ps','-p',str(pid),'-o','%cpu=,state='],capture_output=True,text=True).stdout.split();opened=subprocess.run(['/usr/sbin/lsof','-nP','-p',str(pid),'-Ffn'],capture_output=True,text=True).stdout
   if len(status)==2 and float(status[0])==0 and status[1].startswith('S') and not any(s.startswith('n'+str(INPUT)) for s in opened.splitlines()) and '\nf0\nn/dev/tty' in opened:continue
  conflicts.append(line)
 report=dict(status='PASS' if not conflicts else 'BLOCKED',coordinator_pid=owner,registered_tracker_pid=registered_pid,owned_multiprocessing_helpers=helpers,conflicts=conflicts)
 write(RUN/f'audit/process_guard_{time.time_ns()}.json',report)
 assert not conflicts,('Other FYP process; checkpointed work retained',conflicts)
 return report
def main_gate(label,phase="preflight"):
 from disk_policy import require_space
 assert phase in ("preflight","checkpoint")
 assert RUN.resolve()==Path(spec()['run_directory']) and not RUN.resolve().is_relative_to(INPUT)
 free=require_space(phase)
 env=environment();selected=selected_inputs();no_other_task();a=availability();write(RUN/f'audit/availability_{label}_{time.time_ns()}.json',a);assert a['status']!='BLOCKED',('Upstream original-path verification unavailable',a['issues'])
 hist=historical(label);verifiers=original_verifiers(label)
 final_free=require_space(phase)
 conditional=hist['registered_conditionals']['conditional_paths']>0
 report=dict(status='PASS_WITH_REGISTERED_CONDITIONALS' if conditional else 'PASS',label=label,disk_guard_phase=phase,minimum_free_gib=8 if phase=='preflight' else 5,environment=env,selected_inputs=selected,free_bytes_before=free,free_bytes=final_free,original_verifiers=verifiers,historical=hist,provenance_limitation=dict(policy='standing registered conditional rule, 2026-10-29',proof=hist['registered_conditionals']) if conditional else None,full_fresh_original_verifiers_claimed=all(v['original_verifier_executed'] for v in verifiers))
 p=RUN/f'audit/full_gate_{label}_{time.time_ns()}.json';write(p,report);return dict(file=str(p.relative_to(RUN)),sha256=sha(p))
