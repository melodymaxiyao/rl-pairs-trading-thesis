"""One canonical traversal; fresh hashes once per available unregistered file."""
from common import *
import csv
import registered_scope as conditional
def verify(label):
 import protection as p
 p._CHECKED_TARGET_DIRECTORIES=set();p._CHECKED_EXTERNAL_TARGETS={};p._FROZEN_DIRECTORY_ROWS=None
 pruned=p.registry_rows();pins=json.loads((RUN/'audit/control_registration.json').read_text());inventory=p.COMPACT/'inventory.csv.gz';assert sha(inventory)==pins['files'][str(inventory.relative_to(p.INPUT))]
 expected={};modes={}
 with gzip.open(inventory,'rt',newline='') as f:
  for r in csv.DictReader(f):
   if Path(r['path']).name=='.DS_Store':continue
   mode=modes.setdefault(r['mode'],int(r['mode']))
   expected[r['path']]=dict(kind=sys.intern(r['kind']),size=int(r['size']),mode=mode,mtime_ns=int(r['mtime_ns']),sha256=r['sha256'],target=r['target'])
 from approved_A_pruning import expected_registry_record
 expected['experiments/15_pair_trading/PRUNE_REGISTRY.json']=expected_registry_record(expected.get('experiments/15_pair_trading/PRUNE_REGISTRY.json'))
 canonical=p.canonical_scope();excluded=lambda path:canonical(path) or path.name=='.DS_Store';registry={rel:r for rel,r in conditional.rows(refresh=True).items() if not excluded(p.INPUT/rel)}
 states=conditional.snapshot(excluded=excluded);assert set(states)==set(registry)
 from conflict_copies import index_for
 conflicts=index_for(p.INPUT,expected,dict(path=str(inventory),sha256=pins['files'][str(inventory.relative_to(p.INPUT))]))
 for rel,r in registry.items():
  if rel in expected and r['sealed_sha256']:
   assert r['sealed_sha256']==expected[rel]['sha256'] and r['size']==int(expected[rel]['size']),('Conditional registry differs from frozen inventory',rel)
 for rel,r in pruned.items():assert rel in expected and all(str(r[k])==str(expected[rel][k]) for k in ['sha256','size','mode','mtime_ns'])
 errors=[];last=time.monotonic();unseen=set(expected)-registry.keys()-pruned.keys();sealed={str(Path(rel).parent) for rel in expected if Path(rel).name=='complete.json'};metadata_files=0
 # Validate the complete current namespace before hashing large historical data.
 # This catches a new unregistered placeholder even when it has no inventory row.
 try:
  for path in p.walk():
   rel=str(path.relative_to(p.INPUT))
   if path.name=='.DS_Store':continue
   if conflicts.classify(path) is not None:continue
   unseen.discard(rel);metadata_files+=1
   if rel in registry:continue
   s=path.lstat();r=expected.get(rel);issues=[];drift={}
   if r is None and any(str(q.relative_to(p.INPUT)) in sealed for q in path.parents if q!=p.INPUT and q.is_relative_to(p.INPUT)):issues.append('unregistered addition inside sealed historical run')
   if s.st_flags&DATALESS:issues.append('unregistered dataless path; no content read')
   if r:
    a=dict(size=s.st_size,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns);drift={k:dict(sealed=int(r[k]),actual=v) for k,v in a.items() if v!=int(r[k]) and not (k=='mtime_ns' and r['kind']=='symlink')}
    if drift:issues.append('unregistered historical metadata drift')
   if issues:errors.append(dict(path=rel,error='; '.join(issues),differences=drift,flags=s.st_flags,size=s.st_size))
   if time.monotonic()-last>20:print('Historical metadata files',metadata_files,flush=True);last=time.monotonic()
 except (AssertionError,RuntimeError,OSError) as exc:errors.append(dict(path='current canonical traversal',error=str(exc)))
 for rel in unseen:errors.append(dict(path=rel,error='Unregistered historical absence'))
 proof=conditional.persist(label,states)
 if errors:
  r=dict(status='BLOCKED',phase='metadata_before_content_hashes',historical_content_hashes_performed=0,metadata_files=metadata_files,errors=errors,registered_conditionals=proof,ignored_conflict_copies=conflicts.persist(label+'_blocked'));path=RUN/f'audit/historical_{label}_{time.time_ns()}.json';write(path,r);raise RuntimeError(f'Unregistered historical differences ({len(errors)}); details: {path}')
 seen=set();finder=[];resets=[];protected=[];pending_links=[];verified={};sealed={str(Path(rel).parent) for rel in expected if Path(rel).name=='complete.json'};last=time.monotonic();byte_count=0
 for path in p.walk():
  rel=str(path.relative_to(p.INPUT))
  if path.name=='.DS_Store':finder.append(rel);continue
  if conflicts.classify(path) is not None:continue
  st=p.metadata(path);r=expected.get(rel);seen.add(rel);target=os.readlink(path) if path.is_symlink() else None
  try:
   if rel in registry:
    current=next((a for a in states[rel]['physical'] if a['physical_path']==str(path)),None)
    if current is None or any(current.get('metadata',{}).get(k)!=v for k,v in st.items()):
     current=conditional.assess_one(path,registry[rel]);states[rel]['physical']=[a for a in states[rel]['physical'] if a['physical_path']!=str(path)]+[current]
    digest=current.get('sha256','CONDITIONAL_REGISTERED_CONTENT_HASH_EXCLUDED')
   else:
    if r is None:assert not any(str(q.relative_to(p.INPUT)) in sealed for q in path.parents if q!=p.INPUT and q.is_relative_to(p.INPUT)),('Unregistered addition inside sealed historical run',rel)
    assert not st['flags']&DATALESS,('Unregistered dataless path',rel)
    if r:
     assert all(st[k]==int(r[k]) for k in ['size','mode']),('Unregistered historical size/mode changed',rel)
     if r['kind']=='symlink':
      assert path.is_symlink() and target==r['target'];pending_links.append(path)
      if st['mtime_ns']!=int(r['mtime_ns']):resets.append((rel,int(r['mtime_ns']),st['mtime_ns']))
     else:assert target is None and st['mtime_ns']==int(r['mtime_ns']),('Unregistered regular-file metadata changed',rel)
    digest=hashlib.sha256(os.fsencode(target)).hexdigest() if target is not None else sha(path)
    if r:assert digest==r['sha256'],('Unregistered historical content changed',rel)
    if target is None:byte_count+=st['size']
   record=dict(path=rel,**st,target=target,sha256=digest)
   if os.environ.get('ROU_PREFLIGHT_ONLY')!='1':protected.append(record)
   if target is None and rel not in registry:verified[rel]=record
  except (AssertionError,OSError) as exc:errors.append(dict(path=rel,error=str(exc)))
  if time.monotonic()-last>20:print('Historical exact files',len(seen),'hashed GiB',round(byte_count/2**30,2),flush=True);last=time.monotonic()
 for rel in expected.keys()-seen-pruned.keys()-registry.keys():errors.append(dict(path=rel,error='Unregistered historical absence'))
 covered=dict(pruned);covered.update(states);p._TARGET_HASH_CACHE=verified;p._TARGET_CACHE_ROOT=p.INPUT;p._REGISTERED_CURRENT_STATES=states
 if not errors:
  for path in pending_links:
   try:p.target_check(path,expected,covered)
   except (AssertionError,OSError) as exc:errors.append(dict(path=str(path.relative_to(p.INPUT)),error=str(exc)))
 try:
  extra,more=p.additional_frozen_symlinks(expected,covered);resets.extend(more)
 except (AssertionError,OSError) as exc:extra=0;errors.append(dict(path='additional frozen symlinks',error=str(exc)))
 proof=conditional.persist(label+'_verified',states)
 result=dict(status='PASS_WITH_REGISTERED_CONDITIONALS' if not errors else 'BLOCKED',errors=errors,remaining_checked=len(seen),fresh_unregistered_hashes=len(verified),fresh_unregistered_bytes=byte_count,registered_prune_absences=len((expected.keys()-seen)&pruned.keys()),registered_conditionals=proof,additional_symlinks_checked=extra,finder_exclusions=finder,ignored_conflict_copies=conflicts.persist(label))
 report_path=RUN/f'audit/historical_{label}_{time.time_ns()}.json';write(report_path,result)
 p._IGNORED_CONFLICT_COPIES_PROOF=result['ignored_conflict_copies']
 p._TARGET_HASH_CACHE={}
 assert not errors,('Historical verification failed outside conditional policy',len(errors),str(report_path),errors[:5])
 for args in resets:p.log_reset(*args)
 if os.environ.get('ROU_PREFLIGHT_ONLY')!='1':
  first=cached('runtime/protected_before')
  if first is None:save('runtime/protected_before',protected)
  else:
   old={r['path']:r for r in first};new={r['path']:r for r in protected};names=set(registry);approved_absences=set(pruned)-new.keys()
   from approved_prompt_addition import approve_additions,differences
   approved_additions,addition_proof=approve_additions(old,new)
   if addition_proof is not None:write(event_path('approved_prompt_addition'),addition_proof)
   path_diff=differences(old,new,names,approved_absences,approved_additions)
   assert not path_diff['removed'] and not path_diff['added'],('Unregistered protected path set changed',path_diff)
   for rel,a in old.items():
    if rel in names or rel in approved_absences:continue
    b=new[rel]
    if rel=='experiments/15_pair_trading/PRUNE_REGISTRY.json' and a!=b:
     from approved_A_pruning import approve_control
     proof=approve_control(a,b);write(event_path('approved_A_registry_transition'),proof);continue
    if rel=='experiments/15_pair_trading/RUN_LOCATIONS.json' and a!=b:
     from approved_bookkeeping import approve
     proof=approve(a,b)
     if proof is not None:
      write(event_path('approved_bookkeeping_transition'),proof);continue
    if a['target'] is not None and a['mtime_ns']!=b['mtime_ns']:a=dict(a,mtime_ns=b['mtime_ns']);p.target_check(p.INPUT/rel,expected,covered)
    delta=a['flags']^b['flags']
    if delta and not delta&~0x8040:
     prior=dict(a,flags=b['flags']);assert prior==b,('Flag change accompanied by another protected change',rel,a,b)
     log=RUN/'flag_only_changes.csv';exists=log.exists()
     with log.open('a',newline='') as f:
      w=csv.writer(f)
      if not exists:w.writerow(['path','sealed_flags','observed_flags','xor','content_check'])
      w.writerow([rel,a['flags'],b['flags'],delta,'SHA256,size,mode,inode,mtime exact; keep flags'])
     a=prior
    assert a==b,('Unregistered protected file changed',rel,a,b)
 return result
