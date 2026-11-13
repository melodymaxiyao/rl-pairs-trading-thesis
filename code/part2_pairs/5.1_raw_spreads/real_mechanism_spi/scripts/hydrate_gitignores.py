"""Exact two-file native hydration, preserving sealed compaction artifacts."""
from common import *
from protection import locks,quick_preflight
import subprocess,runpy,argparse,ctypes
FOLDER=HERE/'audit/gitignore_hydration'

def authorization():
 a=read(HERE/'addendum_004_gitignore_hydration_authorization.json')
 for name,key in [('baseline_file','baseline_sha256'),('authorization_file','authorization_sha256')]:assert sha(HERE/a[name])==a[key]
 return a,read(HERE/a['baseline_file'])

def metadata_status(rel,expected):
 p=ROOT/rel;s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve()==p
 assert (s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(expected['size'],expected['mode'],expected['mtime_ns']),('Hydration metadata changed',rel)
 value=dict(path=rel,size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,flags=s.st_flags,label=expected['label'])
 if s.st_flags&DATALESS:
  assert s.st_flags==expected['flags'],('Dataless flags changed',rel)
  return dict(value,status='DATALESS_CONTENT_HASH_EXCLUDED',content_hash_excluded=True)
 assert s.st_flags&~(DATALESS|0x20)==expected['flags']&~(DATALESS|0x20),('Non-availability flags changed',rel)
 digest=sha(p);assert digest==expected['sealed_sha256'],('Materialized SHA-256 mismatch',rel)
 return dict(value,status='MATERIALIZED_SEALED_HASH_PASS',sha256=digest,content_hash_excluded=False)

def native_setup():
 a,b=authorization();facade=types.ModuleType('storage_compaction')
 facade.HERE=FOLDER;facade.ROOT=ROOT;facade.sha=sha
 original=load(COMPACT/'storage_compaction.py','_rms_native_xattrs')
 facade.list_xattrs=original.list_xattrs;facade.get_xattr=original.get_xattr
 facade.journal=lambda event,**fields:None
 facade.write_json=lambda name,v:write(FOLDER/(Path(name).stem+'_'+str(time.time_ns())+'.json'),v)
 with aliases(storage_compaction=facade):m=load(COMPACT/'hydrate_authorized_files.py','_rms_native_hydration')
 return m,facade

def coordinate(rel):
 a,b=authorization();assert rel in b
 m,facade=native_setup()
 with aliases(storage_compaction=facade,hydrate_authorized_files=m):
  sys.argv=[str(COMPACT/'coordinated_hydration.py'),rel]
  assert sha(COMPACT/'coordinated_hydration.py')==registration()[str((COMPACT/'coordinated_hydration.py').relative_to(ROOT))]
  runpy.run_path(str(COMPACT/'coordinated_hydration.py'),run_name='__main__')

def main():
 a,b=authorization();outcome=HERE/'addendum_005_gitignore_hydration_outcome.json'
 if outcome.exists():
  for rel,expected in b.items():metadata_status(rel,expected)
  print('Hydration outcome already frozen; no additional requests');return
 with locks():
  quick_preflight();m,facade=native_setup();attempt=FOLDER/f'attempt_{time.time_ns()}';attempt.mkdir();events=[]
  def record(v):
   events.append(v);write(attempt/f'{len(events):03d}.json',v);print(json.dumps(v,sort_keys=True),flush=True)
  for rel,expected in b.items():
   current=metadata_status(rel,expected)
   if current['content_hash_excluded']:
    count('system_hydration_requests');record(dict(stage='native_initial',result=m.request(ROOT/rel,journal=False)))
  for rel,expected in b.items():
   if not metadata_status(rel,expected)['content_hash_excluded']:continue
   count('system_hydration_requests')
   try:
    r=subprocess.run(['/usr/bin/fileproviderctl','materialize',str(ROOT/rel)],capture_output=True,text=True,timeout=20)
    value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   except subprocess.TimeoutExpired:value=dict(timeout_seconds=20)
   record(dict(stage='exact_fileproviderctl',path=rel,**value))
  for rel,expected in b.items():
   if not metadata_status(rel,expected)['content_hash_excluded']:continue
   log=attempt/('coordinate_'+str(list(b).index(rel))+'.log');count('system_hydration_requests')
   with log.open('x') as f:
    try:r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--coordinate',rel],stdout=f,stderr=subprocess.STDOUT,timeout=30);value=dict(returncode=r.returncode)
    except subprocess.TimeoutExpired:value=dict(timeout_seconds=30)
   record(dict(stage='native_coordinated_read',path=rel,log=str(log.relative_to(HERE)),log_sha256=sha(log),**value))
  for rel,expected in b.items():
   if metadata_status(rel,expected)['content_hash_excluded']:
    count('system_hydration_requests');record(dict(stage='native_retry',result=m.request(ROOT/rel,journal=False)))
  # Match the prior six-poll active native run-loop retry; no indefinite wait.
  send_double=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double)(('objc_msgSend',m.objc))
  for index in range(6):
   pending=[rel for rel,e in b.items() if metadata_status(rel,e)['content_hash_excluded']]
   if not pending:break
   loop=m.send0(m.objc.objc_getClass(b'NSRunLoop'),m.selector('currentRunLoop'))
   until=send_double(m.objc.objc_getClass(b'NSDate'),m.selector('dateWithTimeIntervalSinceNow:'),5.)
   m.send1(loop,m.selector('runUntilDate:'),until)
   record(dict(stage='active_native_runloop_poll',poll=index+1,pending=pending))
  files=[metadata_status(rel,e) for rel,e in b.items()];pending=[r['path'] for r in files if r['content_hash_excluded']]
  materialized=[r['path'] for r in files if not r['content_hash_excluded']]
  count('system_hydration_materialized_files',len(materialized));count('system_hydration_materialized_bytes',sum(r['size'] for r in files if not r['content_hash_excluded']))
  write(outcome,dict(status='FROZEN_HYDRATION_OUTCOME',authorization_sha256=sha(HERE/'addendum_004_gitignore_hydration_authorization.json'),baseline_file=a['baseline_file'],baseline_sha256=a['baseline_sha256'],label='non-scientific .gitignore, non-live',files=files,fallback_paths=pending,materialized_verified_paths=materialized,retry_evidence_directory=str(attempt.relative_to(HERE)),events=len(events),source_file_content_writes=0,storage_compaction_addendum_2_extension=True,metadata_policy='Exact path/size/mode/mtime/flags; only native availability flags may clear upon materialization; no content read while dataless; sealed SHA-256 and size required once materialized; previously verified materialized content cannot return to the exception.'))
  print(json.dumps(dict(status='PASS',materialized=materialized,conditional_fallback=pending)),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--coordinate');args=parser.parse_args();guard()
 if args.coordinate:coordinate(args.coordinate)
 else:main()
