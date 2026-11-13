"""Call the unchanged 22-path journaled repair tool for ten authorized cases."""
import csv,fcntl,gzip,hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];PT=HERE.parent
TOOL=PT/'tav1_runtime_prune_v1/repairs/historical_symlink_mtime_v1/repair.py'
INVENTORY=PT/'storage_compaction_v1/inventory.csv.gz'
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
 data=json.dumps(value,indent=2,sort_keys=True)+'\n'
 if p.exists():assert p.read_text()==data;return
 with p.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
def main():
 with (HERE/'pipeline.lock').open('a+') as lock,(PT/'tav1_runtime_prune_v1/pipeline.lock').open('rb') as old_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=list(csv.DictReader((HERE/'audit/symlink_timestamp_resets.csv').open()))
  paths=sorted(r['path'] for r in rows);assert len(paths)==len(set(paths))==220
  audit=read(PT/'tav1_runtime_prune_v1/audit.json');complete=read(PT/'tav1_runtime_prune_v1/complete.json')
  assert sha(PT/'tav1_runtime_prune_v1/audit.json')==complete['audit_sha256']
  digest=audit['source_code_sha256'][str(INVENTORY.relative_to(ROOT))];assert sha(INVENTORY)==digest
  permitted={str(ROOT/p) for p in paths}
  def guard(event,args):
   if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network/downloads')
   writes=[]
   if event=='open':
    p,mode,flags=args
    if isinstance(p,(str,bytes,os.PathLike)):
     q=Path(os.fsdecode(p)).absolute()
     if q.exists() and q.lstat().st_flags&0x40000000:raise PermissionError('No dataless target reads')
    if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):writes=[p]
   elif event=='os.utime':
    q=Path(args[0]).absolute()
    if str(q) in permitted and q.is_symlink():return
    writes=[args[0]]
   elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.chflags']:writes=[args[0]]
   elif event in ['os.rename','os.link']:writes=args[:2]
   elif event=='os.symlink':writes=[args[1]]
   for p in writes:
    if isinstance(p,(str,bytes,os.PathLike)) and not Path(os.fsdecode(p)).absolute().resolve().is_relative_to(HERE):raise PermissionError('Unauthorized source write: '+str(p))
  sys.addaudithook(guard)
  from symlink_scope import validate
  folder=HERE/'audit/symlink_repair';folder.mkdir(exist_ok=True)
  proof=folder/'targets_before.json'
  if proof.exists():
   # Resume still revalidates targets; preserve every prior proof.
   proof=folder/f'targets_before_resume_{time.time_ns()}.json'
  validate(ROOT,paths,proof)
  original_tool_sha=sha(TOOL)
  for batch in range(10):
   case=folder/f'batch_{batch:03d}';case.mkdir(exist_ok=True)
   subset=paths[22*batch:22*(batch+1)]
   case_audit=dict(adapter='task-local case for unchanged original repair tool',authorized_by='addendum_002_symlink_scope.json',original_prune_audit_sha256=complete['audit_sha256'],source_code_sha256={str(INVENTORY.relative_to(ROOT)):digest})
   write(case/'audit.json',case_audit);write(case/'complete.json',dict(audit_sha256=sha(case/'audit.json')))
   write(case/'metadata_diagnosis.json',dict(mismatches=[dict(path=p) for p in subset]))
   spec=importlib.util.spec_from_file_location('_unchanged_journaled_repair_'+str(batch),TOOL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   module.HERE=case;module.AUDIT=case;module.ROOT=ROOT;module.INVENTORY=INVENTORY
   old_argv=sys.argv;sys.argv=[str(TOOL),'--apply']
   try:module.main()
   finally:sys.argv=old_argv
   assert sha(TOOL)==original_tool_sha
  report=dict(status='PASS',authorized_paths=220,batches=10,original_tool=str(TOOL.relative_to(ROOT)),original_tool_sha256=original_tool_sha,
    original_tool_unchanged=True,follow_symlinks=False,target_validation_sha256=sha(proof),source_file_content_writes=0)
  write(folder/'result.json',report);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()
