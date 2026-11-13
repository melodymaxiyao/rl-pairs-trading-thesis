"""User-authorized pre-result PCI effective-length bucket amendment."""
import numpy as np,pandas as pd
from io_utils import *
from groups import *
import pci,nulls
BASE=ROOT/'preregistration/amendments/speedup_B1_B2_v1'
def bucket(n):
 assert n>=500
 return 500 if n<600 else 600 if n<700 else 700

def design():
 root=BASE/'bucket_design'
 if done(root):return J(root/'design.json')
 values={500:[],600:[],700:[]}
 for kind in ['RP','RF']:
  for y in range(2004,2026):
   for wi in range(3):
    p=ROOT/f'runtime/part_B1/{kind}_{y}_{wi}';assert done(p)
    for row in J(p/'formation_masks.json'):values[row['bucket']].append(row['missing_share'])
 rows={}
 for lower,v in values.items():
  if not v:raise Pause(f'No eligible series in bucket {lower}; cannot invent its median missing share')
  share=float(np.median(v));length=int(np.ceil(lower/(1-share)));rows[str(lower)]=dict(lower_edge=lower,eligible_series_year_worlds=len(v),median_missing_share=share,calendar_length=length,missing_positions=length-lower,effective_observations=lower,actual_simulated_missing_share=(length-lower)/length)
 result=dict(buckets=rows,replications=20000,seed=20261103,mask_sampling='Exactly missing_positions uniform positions without replacement independently for each replication; Gaussian RW evolves on every calendar step',authority_sha256=H(ROOT/'audit/speedup_B1_B2_v1/authorization.json'),not_precomputation=True,user_attested_pre_result=True)
 finish(root,{'design.json':result},phase='amended_PCI_bucket_design');return result

def chunk(lower,offset):
 info=design()['buckets'][str(lower)];root=ROOT/f'runtime/bucket_null_chunks/{lower}/{offset:05d}'
 if done(root):return dict(root=str(root),cached=True)
 rows=[]
 for i in range(offset,offset+100):
  checkpoint();rng=np.random.default_rng(nulls.seed('RW_LR_BUCKET_V1',lower,i));x=np.cumsum(rng.normal(size=info['calendar_length']));missing=rng.choice(len(x),size=info['missing_positions'],replace=False);x[missing]=np.nan;assert np.isfinite(x).sum()==lower
  fit=pci.fit(x,pooled=False,checkpoint=checkpoint)
  if not fit['converged']:raise Pause(f'Bucket {lower} replication{i} did not converge; no replacement')
  rows.append(dict(index=i,LR=fit['LR'],boundary=fit['boundary'],RW_boundary=fit['RW_boundary']))
 return finish(root,{'LR.json':rows},phase='amended_PCI_bucket_null',bucket=lower)

def critical(lower):
 root=BASE/f'critical_values/{lower}'
 if done(root):return J(root/'critical.json')
 info=design()['buckets'][str(lower)];rows=[]
 for offset in range(0,20000,100):
  chunk(lower,offset);rows+=J(ROOT/f'runtime/bucket_null_chunks/{lower}/{offset:05d}/LR.json')
  if offset%1000==0:print('PCI bucket',lower,offset+100,'/20000',flush=True)
 values=np.array([r['LR'] for r in rows]);assert len(values)==20000 and np.isfinite(values).all()
 result=dict(**info,bucket=lower,replications=20000,seed=20261103,quantile_method='higher',critical_5percent=float(np.quantile(values,.95,method='higher')),design_receipt_sha256=H(BASE/'bucket_design/complete.json'))
 finish(root,{'critical.json':result},phase='amended_numeric_PCI_bucket_critical');return result

def for_mask(mask):return critical(bucket(int(np.asarray(mask,bool).sum())))
def sensitivity():
 root=ROOT/'runtime/bucket_pattern_sensitivity'
 if done(root):return
 rows=[]
 for p in (ROOT/'preregistration/critical_values').glob('*/complete.json'):
  assert done(p.parent);r=J(p.parent/'critical.json');n=r['observed_sessions']
  if n<500:continue
  c=critical(bucket(n));rows.append(dict(pattern=r['mask_sha256'],effective_observations=n,pattern_critical=r['critical_5percent'],bucket=c['bucket'],bucket_critical=c['critical_5percent'],difference_pattern_minus_bucket=r['critical_5percent']-c['critical_5percent'],original_receipt_sha256=H(p)))
 finish(root,{'sensitivity.csv.gz':pd.DataFrame(rows)},phase='existing_pattern_vs_bucket_sensitivity')

def overlay_completed(kind,year,wi):
 old=ROOT/f'runtime/part_B/{kind}_{year}_{wi}';root=ROOT/f'runtime/B2_bucket_overlays/{kind}_{year}_{wi}'
 assert done(old)
 if done(root):return dict(root=str(root),cached=True)
 # Existing MLE/LLT results never refit. Only amended gate classification changes.
 f=pd.read_csv(old/'params.csv.gz',float_precision='round_trip')
 for i,row in f.iterrows():
  if row['converged']:
   c=critical(bucket(int(row['valid_formation_sessions'])));f.loc[i,'critical_value']=c['critical_5percent'];f.loc[i,'critical_bucket']=c['bucket'];f.loc[i,'PCI_pass']=bool(row['LR']>c['critical_5percent'] and row['R2_MR']>=.5)
 if 'critical_mask_sha256' in f:f=f.drop(columns='critical_mask_sha256')
 with np.load(old/'signals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 arrays[f'PCI_{wi}']=f.PCI_pass.to_numpy(bool)
 return finish(root,{'params.csv.gz':f,'signals.npz':array_bytes(**arrays),'provenance.json':dict(original_complete_sha256=H(old/'complete.json'),original_group=str(old.relative_to(ROOT)),no_fits_recomputed=True)},phase='B2_completed_group_bucket_overlay')
