import common as C
import numpy as np,io,gzip,pickle,hashlib,zipfile,time,shutil
from pathlib import Path
C.guard();P=C.P;R=Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1');OUT=R/'outputs/N3_intercept_null/CLEAN';t0=time.monotonic();assert shutil.disk_usage(P).free>=2*1024**3
H={}
for p in (P/'audit').glob('*hashes.json'):
 for k,v in C.js(p).items():
  if k in H:assert H[k]==v
  H[k]=v
# Authenticate original registered code and CLEAN records, without importing their executable modules.
for rel in ['audit/N3_intercept_null/CLEAN_addendum/code_registration.json','audit/N3_intercept_null/CLEAN_addendum/registration.json']:
 reg=C.js(R/rel)
 for name,digest in reg['files'].items():
  C.raw(R/name);assert C.INPUTS[str(R/name)]['sha256']==digest,(name,'registered mismatch')
def receipt(root):
 seal=C.js(root/'complete.json');assert seal['status']=='complete'
 for rel,digest in seal['files'].items():
  C.raw(R/rel);assert C.INPUTS[str(R/rel)]['sha256']==digest,rel
 return seal
receipt(OUT/'report');anc=C.js('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1/audit/source_seals.json')['residual_ou_real_v1'];src=Path(anc['root']);b=C.raw(src/'runtime/data/calendar_v2/panel.pkl.gz');assert C.INPUTS[str(src/'runtime/data/calendar_v2/panel.pkl.gz')]['sha256']==anc['expected']['runtime/data/calendar_v2/panel.pkl.gz'];outer=pickle.loads(gzip.decompress(b));assert hashlib.sha256(outer['payload']).hexdigest()==outer['sha256'];meta=pickle.loads(outer['payload'])
def panelarray(name):
 p=src/f'runtime/data/calendar_v2/{name}.npy';b=C.raw(p);assert C.INPUTS[str(p)]['sha256']==anc['expected'][f'runtime/data/calendar_v2/{name}.npy'];return np.load(io.BytesIO(b),allow_pickle=False)
dates=panelarray('dates');membership=panelarray('membership');Y=dates.astype('datetime64[Y]').astype(int)+1970
rf=C.js('/Users/maxiyao/FYP_runs/coint_residual_qlearning_v1/audit/RF_potential_members.json');allowed=set(rf['symbols']);assert len(allowed)==rf['N']==834
rows=[];invent=[];parameters=[]
for family in ['RP','RF']:
 for year in range(2004,2026):
  ix=np.flatnonzero(Y==year);names=[r['pair']for r in meta['pairs'][str(year)]]if family=='RP'else list(meta['symbols']);assert len(set(names))==len(names)
  if family=='RP':assert len(names)==3000 and all(float(r['ipw'])>0 for r in meta['pairs'][str(year)])
  for wi in range(3):
   sroot=OUT/f'signals/{family}_{year}_{wi}';receipt(sroot);froot=OUT/f'pooled/{family}_{year}_{wi}';receipt(froot);fit=C.js(froot/'fit.json');eligible=C.js(froot/'eligible.json');proof=C.js(sroot/'proof.json');assert proof['pooled_receipt_sha256']==C.INPUTS[str(froot/'complete.json')]['sha256']
   b=C.raw(sroot/'signals.npz')
   with zipfile.ZipFile(io.BytesIO(b))as z:
    assert set(z.namelist())=={'K_clean.npy','roll_clean.npy'}
    shapes={}
    for nm in z.namelist():
     with z.open(nm)as f:
      ver=np.lib.format.read_magic(f);shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f));shapes[nm]=shape
     assert shape==(len(names),len(ix))
   invent.append(dict(family=family,year=year,world=wi,shape=list(shapes['K_clean.npy']),fields=['K_clean','roll_clean'],path=str(sroot/'signals.npz'),sha256=C.INPUTS[str(sroot/'signals.npz')]['sha256'],bytes=len(b),converged=fit['converged'],rho=fit.get('rho'),q=fit.get('q'),n_eligible=len(eligible),date_start=str(dates[ix[0]]),date_end=str(dates[ix[-1]])))
   parameters.append(dict(family=family,year=year,world=wi,converged=fit['converged'],rho=fit.get('rho'),q=fit.get('q'),boundary=fit.get('boundary'),n_scale=len(fit.get('scale',[])),n_eligible=len(eligible)))
   assert len(fit['scale'])==len(eligible)
   if wi!=0:continue
   with np.load(io.BytesIO(b),allow_pickle=False)as z:k=z['K_clean']
   valid=np.isfinite(k);e=np.zeros(len(names),bool);e[eligible]=True;assert not valid[~e].any()
   if family=='RF':
    denom=membership[ix].T;assert denom.shape==valid.shape;assert not valid[~denom].any();assert not valid[[n not in allowed for n in names]].any()
   else:denom=np.ones_like(valid)
   op=int(denom.sum());v=int(valid.sum());row=dict(family=family,year=year,n_series=len(names),n_formation_eligible=len(eligible),n_test_days=len(ix),days_with_any_valid_K=int(valid.any(axis=0).sum()),finite_series_days=v,full_grid_series_days=int(valid.size),target_universe_series_days=op,coverage_target_universe=v/op if op else None,coverage_full_grid=v/valid.size,pooled_converged=fit['converged'],pooled_boundary=fit['boundary'],first_test_date=str(dates[ix[0]]),last_test_date=str(dates[ix[-1]]));rows.append(row)
   print('CHECKED',family,year,'eligible',len(eligible),'days',row['days_with_any_valid_K'],'/',len(ix),'coverage',round(row['coverage_target_universe']*100,3),flush=True)
C.csvout(P/'signal_coverage_by_year.csv',rows);C.csvout(P/'audit/signal_inventory.csv',invent);C.csvout(P/'audit/parameters_inventory.csv',parameters)
C.dump(P/'audit/inspection_checks.json',dict(real_groups=44,null_groups=88,all_years=list(range(2004,2026)),all_signals_have_exact_expected_shape=True,no_embedded_ids_or_dates=True,RP_N=3000,RF_panel_columns=863,RF_capital_members=834,one_sided_filter_inspected_not_executed=True,no_parameter_fits=True,no_backtests=True,elapsed_seconds=time.monotonic()-t0,signals_bytes=sum(r['bytes']for r in invent)))
for p,v in C.INPUTS.items():
 if p in H:assert H[p]==v
 H[p]=v
C.dump(P/'input_hashes_before.json',H);after={}
for p,v in H.items():C.raw(p);after[p]=C.INPUTS[p];assert after[p]==v
C.dump(P/'input_hashes_after.json',after)
print('COMPLETE INPUTS',len(H),'scan seconds',time.monotonic()-t0)
