"""Authorized old OU checkpoint trimming; final models/results remain available."""
from pathlib import Path
import os,sys,json,hashlib,stat,gzip,pickle,csv,io,time,subprocess,shutil,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from prune_auth import RUN,ROOT,TARGET,OUT,REGISTRY,sha,read,metadata,authenticate
def write(path,obj):
 raw=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode();path=Path(path)
 if path.exists():assert path.read_bytes()==raw;return
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==raw
def source_seal():
 c=read(TARGET/'complete.json');assert c['status']=='complete' and c['tier']==3 and sha(TARGET/'audit.json')==c['audit_sha256'];a=read(TARGET/'audit.json')
 assert sha(TARGET/'complete.json')=='de95f03dbfe6c6162f25fd01733a040b4678b6b7bb5e3a4b5b9ca1f4ef0c5cc1'
 assert sha(TARGET/'audit.json')=='bbe6dd7f311191aad212f0ed279a82d874758ff87d2975cb9ab758de6c7f10e5'
 assert sha(TARGET/'code_registration.json')=='6ebf76b160226d6251daa7f16bbc74dcef06483224f2ea5f0a30beabf25127a0'
 return dict(complete_sha256=sha(TARGET/'complete.json'),audit_sha256=sha(TARGET/'audit.json'),tier3_audit_sha256=sha(TARGET/'tier_3/audit.json'),code_registration_sha256=sha(TARGET/'code_registration.json'),preregistration_sha256=sha(TARGET/'preregistration.json'))
def apply_registry_intent(intent,registry,after_snapshot,transaction):
 current=sha(registry)
 if current==intent['before_sha256']:
  data=Path(after_snapshot).read_bytes();assert hashlib.sha256(data).hexdigest()==intent['after_sha256']
  with Path(transaction).open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.chmod(transaction,int(intent['before_metadata']['mode']));assert Path(transaction).stat().st_dev==Path(registry).stat().st_dev;os.replace(transaction,registry)
 else:assert current==intent['after_sha256'],'Registry changed outside journaled trim'
 assert sha(registry)==intent['after_sha256'];return metadata(registry)
def audit():
 if (OUT/'plan.json').exists():return read(OUT/'plan.json')
 assert shutil.disk_usage(RUN).free>=8*2**30
 # The original verifier must succeed before any source body is removed.
 for tier in [1,2,3]:
  p=subprocess.run([str(ROOT/'.venv/bin/python'),'-I','-B',str(TARGET/'verify_ou_dp.py'),'--tier',str(tier)],capture_output=True,text=True)
  (OUT/f'verifier_before_tier{tier}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 seals=source_seal();write(OUT/'source_seals.json',seals);a=read(TARGET/'tier_3/audit.json');rows=[];models=[];names=[];world_bodies={};protected={}
 for rel,h in sorted(a['artifacts'].items()):
  p=TARGET/rel;parts=Path(rel).parts
  if parts[:2] not in [('runtime','seeds'),('runtime','worlds')] or not rel.endswith('.pkl.gz'):protected[rel]=h;continue
  s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&0x40000000 and sha(p)==h
  ready=rel[:-7]+'.ready.json';assert read(TARGET/ready)['sha256']==h
  with gzip.open(p,'rb') as f:env=pickle.load(f)
  assert env['preregistration_sha256']==seals['preregistration_sha256'] and hashlib.sha256(env['payload']).hexdigest()==env['sha256'];value=pickle.loads(env['payload'])
  row=dict(path=rel,physical_path=str(p),size=s.st_size,allocated_bytes=s.st_blocks*512,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,ino=s.st_ino,flags=s.st_flags,sha256=h,payload_sha256=env['sha256'],ready_path=ready,ready_sha256=sha(TARGET/ready),role='regenerable training trace; final seedmodel retained' if parts[1]=='seeds' else 'world checkpoint; exact body retained in compact bundle')
  rows.append(row)
  if parts[1]=='seeds':models.append(value);names.append(rel)
  else:world_bodies[rel]=p.read_bytes()
 assert len(models)==5600 and len(world_bodies)==280 and len(rows)==5880
 for name in ['README.md','RESULTS.md','complete.json','audit.json','code_registration.json','preregistration.json','verify_ou_dp.py']:
  p=TARGET/name
  if p.exists():protected[name]=sha(p)
 for name,h in read(TARGET/'code_registration.json')['files'].items():assert sha(TARGET/name)==h;protected[name]=h
 # Preserve every final seed Q table, visit count, uncertainty diagnostic and
 # stopping record. Epoch trajectories remain regenerable checkpoint traces.
 arrays=dict(paths=np.array(names),q=np.stack([v['q'] for v in models]),visits=np.stack([v['visits'] for v in models]),diagnostics=np.stack([v['diagnostics'] for v in models]),epochs=np.array([v['epochs'] for v in models]),converged=np.array([v['converged'] for v in models]),convergence_epoch=np.array([v['convergence_epoch'] for v in models]),learning_steps=np.array([v['learning_steps'] for v in models]))
 b=io.BytesIO();np.savez_compressed(b,**arrays);p=OUT/'retained_seed_models.npz';p.write_bytes(b.getvalue())
 with np.load(p,allow_pickle=False) as check:
  for k,v in arrays.items():assert check[k].dtype==v.dtype and check[k].shape==v.shape and check[k].tobytes()==v.tobytes(),('Final model export not bit-exact',k)
 bundle=gzip.compress(pickle.dumps(world_bodies,protocol=5),mtime=0);p=OUT/'retained_world_bodies.bundle.gz';p.write_bytes(bundle)
 back=pickle.loads(gzip.decompress(p.read_bytes()));assert set(back)==set(world_bodies)
 for rel,body in back.items():assert hashlib.sha256(body).hexdigest()==a['artifacts'][rel]
 text=io.StringIO(newline='');w=csv.DictWriter(text,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);raw=text.getvalue().encode();p=OUT/'prune_manifest.csv.gz';p.write_bytes(gzip.compress(raw,mtime=0))
 with gzip.open(p,'rb') as f:assert f.read()==raw
 write(OUT/'protected_artifacts.json',protected)
 prereg=dict(task='coint_residual_qlearning_v1: prerequisite OU checkpoint trim',authorization='User finalprompt and explicit request: trim ou_dp_qlearning_v1 checkpoint bodies beforeTier1; keepresults,policies,receipts,auditchain; manifest+PRUNE_REGISTRY; nativeverifieracceptsonlyregisteredabsences',target=str(TARGET),source_seals=seals,files=5880,seeds=5600,worlds=280,preserve='all finalpolicies/results/code/complete/audit/receipts and all other source artifacts; finalseedQ/visits/diagnostics/stoppingexport bit-exact; worldbodies losslesslybundled',discard='only the exact authenticatedmanifest bodyfiles; epochtrajectories/curves regenerable from originalkernel and streams',real_fits=0,new_trials=0)
 write(OUT/'preregistration.json',prereg)
 recipe='''# OU checkpoint regeneration and retained science

Source: /Users/maxiyao/FYP_runs/ou_dp_qlearning_v1, sealed code/environment and
stream keys unchanged. This trim never deletes results, policies, any ready
receipt, preregistration, code, complete marker or audit artifact.

retained_seed_models.npz stores every5600finalQtable,visitcount,target-noise
diagnostic andstoppingrecord. Export verified bit-for-bit includingdtype/shape.
retained_world_bodies.bundle.gz stores every280world checkpoint body's exact
bytes keyedbyoriginalrelativepath; rawSHA256 matches the source tier3seal.

Rebuild deleted seed learning trajectories in a NEW outside-iCloud directory,
using the sealed simulation.training, native.train/fit_arrays, source
preregistration, job parameters parsed from the checkpoint path, original
td_seed(j), cost, normalizer and epochcap/earlystop. The final retained q,
visits, diagnostics and epochs must match bit-for-bit. Do not call the sealed
run's RUN_ALL command: it would try to rewrite its immutable checkpoints.
No original source path is restored or metadata repaired automatically.
Restoring originals requires a separately authorized journaled restore.

The original native artifact verifier is run read-only with an adapter that
returns authenticated historical digests ONLY for exact missing registered
checkpoint bodies. All existing files are freshly hashed; every other missing
file fails. The canonical FYP historical checker treats this external source
root separately; these relative paths MUST NOT waive workspace paths.
'''
 (OUT/'regeneration_recipe.md').write_text(recipe)
 plan=dict(status='AUDIT_PASS_AWAITING_EXPLICIT_CONFIRM_FLAG',files=len(rows),source_allocated_bytes=sum(r['allocated_bytes'] for r in rows),source_logical_bytes=sum(r['size'] for r in rows),retained_export_bytes=(OUT/'retained_seed_models.npz').stat().st_size+(OUT/'retained_world_bodies.bundle.gz').stat().st_size,manifest_sha256=sha(OUT/'prune_manifest.csv.gz'),real_fits=0,new_trials=0)
 write(OUT/'plan.json',plan);print(json.dumps(plan),flush=True);return plan
def confirm():
 plan=audit()
 # Recover the exact journaled registry append before considering another
 # append. A crash here must never duplicate the pruning registration.
 if not (OUT/'authorization.json').exists() and (OUT/'registry_intent.json').exists():
  intent=read(OUT/'registry_intent.json');aftermeta=apply_registry_intent(intent,REGISTRY,OUT/'registry.approved_after.json',OUT/'registry.transaction');write(OUT/'authorization.json',dict(intent,after_metadata=aftermeta))
 if not (OUT/'authorization.json').exists():
  assert shutil.disk_usage(RUN).free>=5*2**30
  for path,h in read(OUT/'protected_artifacts.json').items():assert sha(TARGET/path)==h
  before=REGISTRY.read_bytes();old=json.loads(before);beforemeta=metadata(REGISTRY);(OUT/'registry.before.json').write_bytes(before)
  paths=[]
  with gzip.open(OUT/'prune_manifest.csv.gz','rt',newline='') as f:rr=list(csv.DictReader(f))
  for row in rr:
   p=Path(row['physical_path']);s=p.lstat();assert sha(p)==row['sha256'] and s.st_size==int(row['size']) and stat.S_IMODE(s.st_mode)==int(row['mode']) and s.st_mtime_ns==int(row['mtime_ns']);paths.append(row['path'])
  code={name:sha(RUN/'scripts'/name) for name in ['trim_ou_dp.py','prune_auth.py','verify_trimmed_ou.py']};write(OUT/'code_registration.json',dict(files=code))
  auditdoc=dict(status='MANIFEST_READBACK_AND_RETENTION_PROOFS_PASS_DELETION_AUTHORIZED',source_seals=read(OUT/'source_seals.json'),preregistration_sha256=sha(OUT/'preregistration.json'),manifest_sha256=sha(OUT/'prune_manifest.csv.gz'),retained_seed_models_sha256=sha(OUT/'retained_seed_models.npz'),retained_world_bundle_sha256=sha(OUT/'retained_world_bodies.bundle.gz'),code_registration_sha256=sha(OUT/'code_registration.json'),protected_artifacts_sha256=sha(OUT/'protected_artifacts.json'),files=len(rr),real_fits=0,new_trials=0)
  write(OUT/'audit.json',auditdoc)
  entry=dict(status='active_registered_pruning',scope='external_run_checkpoint_bodies',source_root=str(TARGET),audit_directory=str(OUT),audit_sha256=sha(OUT/'audit.json'),files=len(rr),manifest=str(OUT/'prune_manifest.csv.gz'),manifest_sha256=sha(OUT/'prune_manifest.csv.gz'),paths_sha256=hashlib.sha256(('\n'.join(sorted(paths))+'\n').encode()).hexdigest(),preregistration_sha256=sha(OUT/'preregistration.json'),recipe=str(OUT/'regeneration_recipe.md'),recipe_sha256=sha(OUT/'regeneration_recipe.md'),disposition='regenerable checkpoint bodies pruned; final seed models and exact world payloads retained')
  updated=json.loads(json.dumps(old));updated['entries'].append(entry);after=(json.dumps(updated,sort_keys=True,indent=2)+'\n').encode();(OUT/'registry.approved_after.json').write_bytes(after)
  intent=dict(authorization='User taskexplicitly authorizes thisnamedcheckpointtrim andPRUNE_REGISTRY append',entry=entry,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),before_metadata=beforemeta,row_path_interpretation='relative to source_root, never relative to FYP workspace')
  write(OUT/'registry_intent.json',intent)
  assert REGISTRY.read_bytes()==before
  aftermeta=apply_registry_intent(intent,REGISTRY,OUT/'registry.approved_after.json',OUT/'registry.transaction')
  assert REGISTRY.read_bytes()==after
  write(OUT/'authorization.json',dict(intent,after_metadata=aftermeta))
 authority,registered=authenticate();before_free=shutil.disk_usage(RUN).free;removed=0;previous=0
 with (OUT/'deletions.jsonl').open('a') as log:
  for name,row in sorted(registered.items()):
   p=Path(name)
   if not p.exists():previous+=1;continue
   s=p.lstat();assert stat.S_ISREG(s.st_mode) and not s.st_flags&0x40000000 and sha(p)==row['sha256'] and s.st_size==int(row['size']) and stat.S_IMODE(s.st_mode)==int(row['mode']) and s.st_mtime_ns==int(row['mtime_ns'])
   p.unlink();removed+=1;log.write(json.dumps(dict(path=name,sha256=row['sha256']))+'\n')
   if removed%1000==0:log.flush();os.fsync(log.fileno());print('Registered checkpoint bodies removed',removed,flush=True)
  log.flush();os.fsync(log.fileno())
 for tier in [1,2,3]:
  p=subprocess.run([str(ROOT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/verify_trimmed_ou.py'),str(tier)],capture_output=True,text=True);(OUT/f'verifier_after_tier{tier}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 for path,h in read(OUT/'protected_artifacts.json').items():assert sha(TARGET/path)==h
 result=dict(status='PASS_REGISTERED_CHECKPOINT_TRIM',removed_now=removed,already_registered_absent=previous,total_registered_bodies=len(registered),allocated_source_bytes=plan['source_allocated_bytes'],retained_export_bytes=plan['retained_export_bytes'],free_bytes_before_deletion=before_free,free_bytes_after=shutil.disk_usage(RUN).free,df_observed_delta_bytes=shutil.disk_usage(RUN).free-before_free,all_other_tier3_artifacts_unchanged=True,native_tiers_verified=[1,2,3],source_code_changes=0,real_fits=0,new_trials=0)
 write(OUT/'outcome.json',result);print(json.dumps(result),flush=True);print(subprocess.run(['/bin/df','-k',str(RUN)],capture_output=True,text=True).stdout,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--confirm',action='store_true');args=parser.parse_args();confirm() if args.confirm else audit()
