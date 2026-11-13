"""Refresh only this task's preparation controls after the scoped snapshot repair."""
from common import *
from protection import locks
import prepare
FOLDER=HERE/'audit/repairs/protected_scope_v1'
CONTROLS={'audit/code_registration.json','audit/preparation_ready.json','audit/preparation_ledger.json','audit.json','operation_ledger.json'}

def main():
 guard()
 with locks():
  assert not (HERE/'complete.json').exists()
  measured=ledger()
  for scope in ['production','acceptance']:
   assert measured['scopes'].get(scope,{}).get('TD_fits',0)==0 and measured['scopes'].get(scope,{}).get('switch_tests',0)==0
  diagnosis=read(FOLDER/'diagnosis.json');original=read(FOLDER/'before/audit/code_registration.json')
  for rel,digest in diagnosis['before_files'].items():assert sha(FOLDER/'before'/rel)==digest
  for rel,digest in original['files'].items():
   p=FOLDER/'before'/rel if rel=='scripts/protection.py' else HERE/rel
   assert sha(p)==digest,('Unrelated prepared file changed',rel)
  assert sha(FOLDER/'before/README.md')==original['README_sha256']
  validation=read(FOLDER/'validation.json');assert validation['status']=='PASS'
  assert sha(HERE/validation['snapshot_file'])==validation['snapshot_sha256']
  assert sha(HERE/validation['historical_receipt'])==validation['historical_receipt_sha256']
  proof=passing_tests('preparation')
  old_write,old_publish=prepare.write,prepare.publish
  journal=FOLDER/'refresh_journal.jsonl'
  def event(v):
   with journal.open('a') as f:f.write(json.dumps(dict(time_ns=time.time_ns(),**v))+'\n');f.flush();os.fsync(f.fileno())
  def publish_control(path,data):
   path=Path(path);rel=str(path.relative_to(HERE))
   if isinstance(data,str):data=data.encode()
   if rel not in CONTROLS:return old_publish(path,data)
   if path.exists() and path.read_bytes()==data:return
   assert sha(path)==diagnosis['before_files'][rel],('Unexpected control revision; preserved',rel)
   digest=hashlib.sha256(data).hexdigest();event(dict(event='replace_intent',path=rel,before_sha256=sha(path),after_sha256=digest,archived_before=str((FOLDER/'before'/rel).relative_to(HERE))))
   pending=path.with_name(path.name+'.pending_scope_repair_'+str(time.time_ns()));old_publish(pending,data);os.replace(pending,path)
   fd=os.open(path.parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
   assert sha(path)==digest;event(dict(event='replaced',path=rel,after_sha256=digest))
  def write_control(path,value):
   if Path(path)==HERE/'audit/code_registration.json':
    extra=[p for p in (FOLDER/'before').rglob('*') if p.is_file()]
    extra += [FOLDER/'diagnosis.json',FOLDER/'validation.json',HERE/validation['historical_receipt'],HERE/validation['snapshot_file']]
    value=dict(value,repair='addendum_008_protected_snapshot_scope_repair.json',previous_registration_sha256=diagnosis['original_registration_sha256'])
    value['files'].update({str(p.relative_to(HERE)):sha(p) for p in extra})
   publish_control(path,json.dumps(clean(value),sort_keys=True,indent=2,allow_nan=False)+'\n')
  prepare.write,prepare.publish=write_control,publish_control
  try:prepare.finish()
  finally:prepare.write,prepare.publish=old_write,old_publish
  write(FOLDER/'refresh_receipt.json',dict(status='PASS',original_registration_sha256=diagnosis['original_registration_sha256'],current_registration_sha256=sha(HERE/'audit/code_registration.json'),current_preparation_ready_sha256=sha(HERE/'audit/preparation_ready.json'),short_test_receipt=proof,validation='audit/repairs/protected_scope_v1/validation.json',source_content_writes=0,production_run=False))
  print('PASS: scoped preparation repair sealed; production not run',flush=True)

if __name__=='__main__':main()
