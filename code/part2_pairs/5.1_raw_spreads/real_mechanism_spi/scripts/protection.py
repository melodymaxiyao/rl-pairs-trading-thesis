"""Frozen input checks, exact metadata drift reporting, and final task seal."""
from common import *
import csv,subprocess,inspect
INPUTS={'trend_anchor_trading_v1':'verify_trend_anchor.py','trend_anchor_td_tuned_v1':'verify_td_tuned_resume_v3.py',
 'td_mechanism_spi_v1':'verify_td_mechanism_spi.py','residual_learnability_v1':'verify_residual_learnability_native_v5.py',
 'fresh_holdout_data_v1':'verify_fresh_data.py','td_sample_complexity_v1':'verify_td_sample_complexity.py','thesis_tables_v1':'verify_thesis_tables.py'}
def registry_rows():
 reg=read(PT/'PRUNE_REGISTRY.json');rows={}
 for entry in reg['entries']:
  assert sha(ROOT/entry['manifest'])==entry['manifest_sha256'];assert sha(ROOT/entry['recipe'])==entry['recipe_sha256']
  assert sha(ROOT/entry['audit_directory']/'audit.json')==entry['audit_sha256']
  with gzip.open(ROOT/entry['manifest'],'rt',newline='') as f:rr=list(csv.DictReader(f))
  assert len(rr)==entry['files'] and len({r['path'] for r in rr})==len(rr)
  assert hashlib.sha256(('\n'.join(sorted(r['path'] for r in rr))+'\n').encode()).hexdigest()==entry['paths_sha256']
  for r in rr:
   assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
   assert r['path'] not in rows;rows[r['path']]=r
 return rows
def frozen_inventory():
 audit=read(PRUNE/'audit.json');assert sha(PRUNE/'audit.json')==read(PRUNE/'complete.json')['audit_sha256']
 p=COMPACT/'inventory.csv.gz';assert sha(p)==audit['source_code_sha256'][str(p.relative_to(ROOT))]
 return p
def metadata_check(label):
 from conditional_scope import allowed_paths,state
 approved=registry_rows();auth=allowed_paths();diff=[];resets=[];devices=set();frozen_link_paths=set();seen=0;links=0;last=time.monotonic()
 with gzip.open(frozen_inventory(),'rt',newline='') as f:
  for row in csv.DictReader(f):
   rel=row['path']
   if Path(rel).name=='.DS_Store':continue
   p=ROOT/rel
   try:st=p.lstat()
   except FileNotFoundError:
    assert rel in approved,('Unregistered missing historical path',rel);continue
   fields=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode))
   expected={k:int(row[k]) for k in fields}
   if row['kind']=='symlink':
    links+=1;frozen_link_paths.add(rel);assert p.is_symlink() and p.exists(),('Symlink type/existence changed',rel)
    fields.update(flags=st.st_flags,ino=st.st_ino,nlink=st.st_nlink,target=os.readlink(p))
    expected.update(flags=int(row['flags']),ino=int(row['ino']),nlink=int(row['nlink']),target=row['target'])
    devices.add((int(row['dev']),st.st_dev));assert st.st_dev==ROOT.stat().st_dev,('Link moved off the workspace volume',rel)
   if fields!=expected:
    changed={k:dict(before=expected[k],after=fields[k]) for k in fields if fields[k]!=expected[k]}
    target_equal=row['kind']=='symlink' and os.readlink(p)==row['target']
    content_equal=target_equal or (row['kind']=='file' and rel not in auth and sha(p)==row['sha256'])
    item=dict(path=rel,differences=changed,content_or_target_unchanged=content_equal,timestamp_only=set(changed)=={'mtime_ns'})
    if row['kind']=='symlink' and item['timestamp_only'] and content_equal:resets.append(item)
    else:diff.append(item)
   seen+=1
   if time.monotonic()-last>25:print('Metadata check',label,seen,flush=True);last=time.monotonic()
 from symlink_scope import validate,verify_external_targets
 relocation=PT/'fyp_relocation_v1';rc=read(relocation/'complete.json');ra=read(relocation/'audit.json')
 assert sha(relocation/'audit.json')==rc['audit_sha256'] and sha(relocation/'symlinks.json')==ra['artifact_sha256']['symlinks.json']
 extra=0
 for row in read(relocation/'symlinks.json'):
  rel=row['path']
  if rel in frozen_link_paths:continue
  extra+=1;p=ROOT/rel;st=p.lstat();old=row['lstat']
  assert p.is_symlink() and p.exists()==row['target_exists_before'] and os.readlink(p)==row['target'] and stat.S_IMODE(st.st_mode)==old['mode'],('Additional frozen symlink changed',rel)
  if st.st_mtime_ns!=old['mtime_ns']:resets.append(dict(path=rel,differences=dict(mtime_ns=dict(before=old['mtime_ns'],after=st.st_mtime_ns)),content_or_target_unchanged=True,timestamp_only=True))
 external=verify_external_targets(ROOT);write(HERE/f'audit/external_symlink_targets_{label}_{time.time_ns()}.json',external)
 if resets:
  proof=HERE/f'audit/target_validation_{label}_{time.time_ns()}.json'
  validate(ROOT,[r['path'] for r in resets],proof,allow_outside_additions=True)
  log=HERE/'symlink_timestamp_resets.csv';exists=log.exists()
  with log.open('a',newline='') as f:
   writer=csv.writer(f)
   if not exists:writer.writerow(['path','frozen_mtime_ns','observed_mtime_ns','action','target_proof','target_proof_sha256'])
   for r in resets:writer.writerow([r['path'],r['differences']['mtime_ns']['before'],r['differences']['mtime_ns']['after'],'ACCEPTED_UNDER_ADDENDUM_002',str(proof.relative_to(HERE)),sha(proof)])
   f.flush();os.fsync(f.fileno())
 report=dict(status='PASS' if not diff else 'STOP_METADATA_DRIFT',historical_present=seen,frozen_symlinks=links,additional_frozen_symlinks=extra,differences=diff,permitted_symlink_timestamp_resets=resets,mount_device_number_pairs=sorted(devices),repairs=0,standing_scope_addendum='addendum_002_symlink_scope.json')
 path=HERE/f'audit/metadata_{label}_{time.time_ns()}.json';write(path,report)
 if diff:raise RuntimeError('Metadata drift; no repair performed. Exact paths in '+str(path))
 # Authenticate and apply the exact dataless metadata/flags policy, without content reads.
 conditional=state(persist=True)
 write(HERE/f'audit/dataless_{label}_{time.time_ns()}.json',conditional)
 return report
@contextmanager
def locks():
 with (HERE/'pipeline.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for path in list(PT.glob('*/pipeline.lock'))+list(PT.glob('*/coordinator.lock')):
   if path.parent==HERE:continue
   with path.open('rb') as old:
    fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(old,fcntl.LOCK_UN)
  yield
def unrelated_interactive_python(pid,command):
 # A virtualenv interpreter alone is not evidence of a workspace task. Require
 # a sleeping bare REPL, external cwd, terminal stdin, and no open FYP files.
 if command not in {str(ROOT/'.venv/bin'/n) for n in ['python','python3','python3.13']}:return None
 status=subprocess.run(['/bin/ps','-p',str(pid),'-o','%cpu=,state='],capture_output=True,text=True,check=False)
 fields=status.stdout.split()
 if len(fields)!=2 or float(fields[0])!=0 or not fields[1].startswith('S'):return None
 opened=subprocess.run(['/usr/sbin/lsof','-nP','-p',str(pid),'-Ffn'],capture_output=True,text=True,check=False)
 if opened.returncode!=0:return None
 fd=None;cwd=None;stdin=None;names=[]
 for line in opened.stdout.splitlines():
  if line.startswith('f'):fd=line[1:]
  elif line.startswith('n'):
   name=line[1:];names.append(name)
   if fd=='cwd':cwd=name
   elif fd=='0':stdin=name
 if not cwd or Path(cwd).is_relative_to(ROOT) or not stdin or not stdin.startswith('/dev/tty'):return None
 if any(Path(n).is_relative_to(ROOT) for n in names if n.startswith('/')):return None
 return dict(pid=pid,command=command,cwd=cwd,stdin=stdin,state=fields[1],cpu_percent=float(fields[0]),open_FYP_files=0)
def quick_preflight():
 free=shutil.disk_usage(ROOT).free;assert free>=8*2**30,('Preflight stop: less than 8 GiB free',free/2**30)
 check_environment()
 ps=subprocess.run(['/bin/ps','-axo','pid=,command='],capture_output=True,text=True,check=True).stdout
 conflicts=[];unrelated=[]
 for line in ps.splitlines():
  parts=line.split(None,1)
  if len(parts)<2 or int(parts[0]) in {os.getpid(),os.getppid()}:continue
  text=parts[1]
  if 'python' in text.lower() and (str(ROOT) in text or 'experiments/15_pair_trading/' in text) and str(HERE) not in text and '/python-env-tools/' not in text:
   evidence=unrelated_interactive_python(int(parts[0]),text)
   if evidence is None:conflicts.append(line)
   else:unrelated.append(evidence)
 assert not conflicts,('Other FYP task active',conflicts)
 for name,digest in spec()['input_seals'].items():
  c=PT/name/'complete.json';assert sha(c)==digest and read(c).get('status')=='complete',('Input completion changed',name)
 write(HERE/f'audit/preflight_{time.time_ns()}.json',dict(status='PASS',free_GiB=free/2**30,minimum_GiB=8,checkpoint_minimum_GiB=5,process_conflicts=conflicts,unrelated_external_interactive_interpreters=unrelated,
  df=subprocess.run(['/bin/df','-h',str(ROOT)],capture_output=True,text=True,check=True).stdout))
def original_verifiers(label):
 results=[]
 for name,script in INPUTS.items():
  log=HERE/f'audit/input_{label}_{name}_{time.time_ns()}.log'
  env=dict(os.environ,RMS_INTEGRITY='1')
  with log.open('x') as f:r=subprocess.run([sys.executable,'-B',str(HERE/'scripts/input_entry.py'),name,script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
  log_timestamp_acceptances(log)
  assert r.returncode==0,('Read-only input verifier failed; log preserved',name,log)
  results.append(dict(name=name,status='EXPECTED_REGISTERED_ABSENCE' if name=='trend_anchor_trading_v1' else 'PASS',script_sha256=sha(PT/name/script),log=str(log.relative_to(HERE)),log_sha256=sha(log)))
  checkpoint_guard();print('Input checked',name,flush=True)
 write(HERE/f'audit/input_verifiers_{label}_{time.time_ns()}.json',results);return results
def log_timestamp_acceptances(log):
 from conditional_scope import capture_log
 capture_log(log)
 accepted=[json.loads(line.split(' ',1)[1]) for line in log.read_text().splitlines() if line.startswith('SYMLINK_TIMESTAMP_ACCEPTED ')]
 if accepted:
  with (HERE/'symlink_timestamp_resets.csv').open('a',newline='') as f:
   writer=csv.writer(f)
   for row in accepted:writer.writerow([row['path'],row['frozen_mtime_ns'],row['observed_mtime_ns'],'ACCEPTED_VERIFIED_MTIME_READ_VIEW',str(log.relative_to(HERE)),sha(log)])
   f.flush();os.fsync(f.fileno())
def historical(label):
 log=HERE/f'audit/historical_{label}_{time.time_ns()}.log';env=dict(os.environ,RMS_INTEGRITY='1')
 with log.open('x') as f:r=subprocess.run([sys.executable,'-B',str(HERE/'scripts/historical_entry.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
 log_timestamp_acceptances(log)
 assert r.returncode==0,('Prune-aware full historical verification failed',log)
 result=json.loads(log.read_text().splitlines()[-1]);write(HERE/f'audit/historical_{label}_{time.time_ns()}.ready.json',dict(status='PASS',log=str(log.relative_to(HERE)),log_sha256=sha(log),remaining_exact=result['historical_scope']['remaining_exact'],registered_pruned=result['registered_pruned_files']))
 return result
def protected_paths():
 """Use the sealed historical traversal scope, plus this task's output exclusion."""
 from conditional_scope import original_policy
 canonical=original_policy().s
 def excluded(path):
  return path.is_relative_to(HERE) or canonical.excluded(canonical.ROOT/path.relative_to(ROOT))
 for parent,dirs,names in os.walk(ROOT,followlinks=False):
  parent=Path(parent);dirs[:]=sorted(d for d in dirs if not excluded(parent/d))
  for name in list(dirs):
   if (parent/name).is_symlink():names.append(name);dirs.remove(name)
  for name in sorted(names):
   path=parent/name
   if not excluded(path):yield path
def protected_snapshot(label):
 from conditional_scope import state
 conditional=state(persist=True);exclusions=set(conditional['active_content_hash_exclusions'])
 # Historical SHA values may be reused only immediately after the full historical gate.
 known={}
 with gzip.open(frozen_inventory(),'rt',newline='') as f:
  for r in csv.DictReader(f):known[r['path']]=(r['sha256'],int(r['size']),int(r['mtime_ns']),int(r['mode']))
 p=HERE/f'audit/protected_{label}.csv.gz';finder=[];last=time.monotonic();n=0
 fields=['path','kind','sha256','size','mtime_ns','mode','flags','ino','dev','nlink','target']
 if p.exists():
  # Every resume still runs the full gate; preserve and compare the first snapshot.
  return compare_snapshot(p,known)
 pending=p.with_name(p.name+'.pending_'+str(time.time_ns()))
 with gzip.open(pending,'xt',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  for path in protected_paths():
   rel=str(path.relative_to(ROOT))
   if path.name=='.DS_Store':finder.append(rel);continue
   st=path.lstat();kind='symlink' if path.is_symlink() else 'file';old=known.get(rel);target=os.readlink(path) if kind=='symlink' else ''
   if kind=='symlink':digest=hashlib.sha256(os.fsencode(target)).hexdigest()
   elif st.st_flags&DATALESS:
    assert rel in exclusions,('Unapproved dataless file in protected snapshot',rel,dict(size=st.st_size,mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns,flags=st.st_flags))
    digest='CONDITIONAL_DATALESS_CONTENT_HASH_EXCLUDED'
   elif old and (st.st_size,st.st_mtime_ns,stat.S_IMODE(st.st_mode))==old[1:]:digest=old[0]
   else:digest=sha(path)
   writer.writerow(dict(path=rel,kind=kind,sha256=digest,size=st.st_size,mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode),flags=st.st_flags,ino=st.st_ino,dev=st.st_dev,nlink=st.st_nlink,target=target));n+=1
   if time.monotonic()-last>25:print('Protected inventory',label,n,flush=True);last=time.monotonic()
 with pending.open('rb') as f:os.fsync(f.fileno())
 os.link(pending,p);pending.unlink()
 write(HERE/f'audit/finder_{label}.json',finder)
 return dict(status='PASS',paths=n,sha256=sha(p))
def compare_snapshot(path,known=None):
 from conditional_scope import state,materialization_transition
 exclusions=set(state(persist=True)['active_content_hash_exclusions'])
 differences=[];resets=[];transitions=[];seen=0
 with gzip.open(path,'rt',newline='') as f:
  for row in csv.DictReader(f):
   p=ROOT/row['path']
   try:st=p.lstat()
   except FileNotFoundError:differences.append(dict(path=row['path'],reason='missing'));continue
   actual=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode),flags=st.st_flags,ino=st.st_ino,dev=st.st_dev,nlink=st.st_nlink)
   changes={k:dict(before=int(row[k]),after=v) for k,v in actual.items() if int(row[k])!=v}
   # st_dev is mount-local; a global renumbering is recorded at preflight.
   if 'dev' in changes and st.st_dev==ROOT.stat().st_dev:changes.pop('dev')
   if p.is_symlink():digest=hashlib.sha256(os.fsencode(os.readlink(p))).hexdigest()
   elif st.st_flags&DATALESS:
    assert row['path'] in exclusions,('Unapproved dataless protected path',row['path'])
    digest='CONDITIONAL_DATALESS_CONTENT_HASH_EXCLUDED'
   elif known and row['path'] in known and tuple(actual[k] for k in ['size','mtime_ns','mode'])==known[row['path']][1:]:digest=known[row['path']][0]
   else:digest=sha(p)
   if row['kind']=='symlink' and set(changes)=={'mtime_ns'} and digest==row['sha256']:
    resets.append(dict(path=row['path'],before=int(row['mtime_ns']),after=st.st_mtime_ns));changes={}
   transition=materialization_transition(row,dict(actual,path=row['path'],sha256=digest)) if 'flags' in changes else False
   if transition:changes.pop('flags');transitions.append(row['path'])
   if changes or (digest!=row['sha256'] and not transition):differences.append(dict(path=row['path'],metadata_changes=changes,content_unchanged=digest==row['sha256']))
   seen+=1
 if resets:
  from symlink_scope import validate
  proof=HERE/f'audit/snapshot_target_validation_{time.time_ns()}.json';validate(ROOT,[r['path'] for r in resets],proof,allow_outside_additions=True)
  with (HERE/'symlink_timestamp_resets.csv').open('a',newline='') as f:
   writer=csv.writer(f)
   for row in resets:writer.writerow([row['path'],row['before'],row['after'],'ACCEPTED_UNDER_ADDENDUM_002',str(proof.relative_to(HERE)),sha(proof)])
 report=dict(status='PASS' if not differences else 'STOP_INPUT_DRIFT',paths=seen,differences=differences,permitted_symlink_timestamp_resets=resets,verified_native_materializations=transitions)
 write(HERE/f'audit/protected_comparison_{time.time_ns()}.json',report)
 assert not differences,('Protected source drift; no repairs',differences[:10]);return report
def registered():
 check_environment()
 r=read(HERE/'audit/code_registration.json')
 for rel,digest in r['files'].items():assert sha(HERE/rel)==digest,('Prepared code changed',rel)
 assert sha(HERE/'README.md')==r['README_sha256']
 for rel,digest in registration().items():assert sha(ROOT/rel)==digest,('Imported source changed',rel)
 for name,digest in spec()['input_seals'].items():assert sha(PT/name/'complete.json')==digest
 return r
def verify_opened_inputs():
 pinned={}
 for p in (HERE/'audit').glob('opened_inputs_*.jsonl'):
  for line in p.read_text().splitlines():
   if not line:continue
   row=json.loads(line);key=row['path'];expected=row['fingerprint']
   if key in pinned:assert pinned[key]==expected,('Conflicting source fingerprints',key)
   else:pinned[key]=expected
 for key,expected in pinned.items():assert fingerprint(ROOT/key)==expected,('Opened input changed',key)
 return dict(status='PASS',files=len(pinned))
def compare_snapshots(before,after):
 """Disk-backed exact comparison, keeping the report process memory bounded."""
 import sqlite3
 from conditional_scope import materialization_transition
 db=HERE/f'runtime/tmp/protected_compare_{time.time_ns()}.sqlite3';connection=sqlite3.connect(db)
 connection.execute('PRAGMA journal_mode=OFF');connection.execute('CREATE TABLE original(path TEXT PRIMARY KEY, value TEXT)')
 sealed=set();batch=[]
 with gzip.open(before,'rt',newline='') as f:
  for row in csv.DictReader(f):
   if Path(row['path']).name=='complete.json':sealed.add(str(Path(row['path']).parent))
   batch.append((row['path'],json.dumps(row,sort_keys=True)))
   if len(batch)==5000:connection.executemany('INSERT INTO original VALUES(?,?)',batch);batch=[]
  if batch:connection.executemany('INSERT INTO original VALUES(?,?)',batch)
 connection.commit();differences=[];resets=[];transitions=[];added=[];seen=0
 with gzip.open(after,'rt',newline='') as f:
  for row in csv.DictReader(f):
   found=connection.execute('SELECT value FROM original WHERE path=?',(row['path'],)).fetchone()
   if found is None:
    assert not any(str(p) in sealed for p in Path(row['path']).parents),('Added inside sealed input',row['path']);added.append(row['path']);continue
   old=json.loads(found[0]);changes={k:dict(before=v,after=row[k]) for k,v in old.items() if k!='dev' and row[k]!=v}
   if 'flags' in changes and materialization_transition(old,row):
    changes.pop('flags');changes.pop('sha256',None);transitions.append(row['path'])
   if old['kind']=='symlink' and set(changes)=={'mtime_ns'}:
    resets.append(dict(path=row['path'],before=int(old['mtime_ns']),after=int(row['mtime_ns'])));changes={}
   if changes:differences.append(dict(path=row['path'],changes=changes))
   connection.execute('DELETE FROM original WHERE path=?',(row['path'],));seen+=1
   if seen%10000==0:connection.commit();checkpoint_guard()
 missing=[x[0] for x in connection.execute('SELECT path FROM original')];connection.close()
 if resets:
  from symlink_scope import validate
  proof=HERE/f'audit/protected_target_validation_{time.time_ns()}.json';validate(ROOT,[r['path'] for r in resets],proof,allow_outside_additions=True)
  with (HERE/'symlink_timestamp_resets.csv').open('a',newline='') as f:
   writer=csv.writer(f)
   for row in resets:writer.writerow([row['path'],row['before'],row['after'],'ACCEPTED_UNDER_ADDENDUM_002',str(proof.relative_to(HERE)),sha(proof)])
 result=dict(status='PASS' if not differences and not missing else 'STOP_PROTECTED_CHANGE',remaining_paths_exact=seen,
             differences=differences,missing=missing,added_outside_sealed_inputs=added,permitted_symlink_timestamp_resets=resets,verified_native_materializations=transitions)
 write(HERE/f'audit/protected_comparison_{time.time_ns()}.json',result)
 assert result['status']=='PASS',('Protected input changed; report preserved',result['differences'][:3],missing[:3])
 return result
