"""Authenticate source seals and exact authorized bookkeeping; no scientific source edits."""
import ast,builtins,gzip,hashlib,importlib.util,io,json,os,pathlib,subprocess,sys,types
from io_utils import *
ORIGINAL_POPEN=subprocess.Popen
def validate_sources():
 reg=J(ROOT/'code_registration.json')
 for rel,digest in reg['files'].items():assert H(ROOT/rel)==digest,('Gate code changed',rel)
 assert H(ROOT/'preregistration.json')==reg['preregistration_sha256']
 complete=J(CORE/'tier_3/complete.json');assert H(CORE/'tier_3/complete.json')==reg['core_complete_sha256'];assert H(CORE/'tier_3/manifest.json')==complete['manifest_sha256'];assert complete['protocol_complete'] and complete['real_trials']==572
 for name,digest in reg['upstream_code_files'].items():assert H(CORE/name)==digest
 gate=ROOT.parent/'gate_power_v1';assert H(gate/'complete.json')==reg['gate_complete_sha256'];g=J(gate/'complete.json');assert g['status']=='complete' and g['real_trials']==580 and H(gate/'manifest.json')==g['manifest_sha256']
 return reg
class MetaView:
 def __init__(self,value,row):self.value=value;self.row=row
 def __getattr__(self,key):return dict(st_size=self.row['size'],st_mode=stat.S_IFREG|self.row['mode'],st_mtime_ns=self.row['mtime_ns'],st_mtime=self.row['mtime_ns']/1e9,st_ino=self.row['ino'],st_flags=self.row['flags']).get(key,getattr(self.value,key,None))
 def __iter__(self):return iter(self.value)
 def __getitem__(self,key):return self.value[key]
 def __len__(self):return len(self.value)
def location_layer():
 # Authenticate the physical task addition before exposing its exact previous
 # control image to already-sealed native verifiers. No physical hash is substituted.
 p=INPUT/'experiments/15_pair_trading/RUN_LOCATIONS.json';auth=J(ROOT/'audit/location_authorization.json')
 before=ROOT/'audit/run_locations.before.json';after=ROOT/'audit/run_locations.approved_after.json';assert H(before)==auth['before_sha256'] and H(after)==auth['after_sha256'] and H(p)==auth['after_sha256']
 b=J(before);a=J(after);reduced=json.loads(json.dumps(a));assert reduced['runs'].pop('partial_coint_kalman_v1')==auth['addition'];reduced['updated_on']=b['updated_on'];assert reduced==b
 expected=J(ROOT/'audit/run_locations.after_metadata.json');old=J(ROOT/'audit/run_locations.before_metadata.json');raw=before.read_bytes();open0=builtins.open;stat0=os.stat;lstat0=os.lstat
 checking=[False]
 def matches(path):return isinstance(path,(str,bytes,os.PathLike)) and pathlib.Path(os.fsdecode(path)).absolute()==p
 def physical():
  if checking[0]:return
  checking[0]=True
  try:
   s=lstat0(p);assert all(getattr(s,{'size':'st_size','mtime_ns':'st_mtime_ns','ino':'st_ino'}[k])==expected[k] for k in ['size','mtime_ns','ino']) and stat.S_IMODE(s.st_mode)==expected['mode'] and not ((s.st_flags^expected['flags'])&~0x8040)
   with open0(p,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==auth['after_sha256']
  finally:checking[0]=False
 def vopen(path,mode='r',buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None):
  if not matches(path):return open0(path,mode,buffering=buffering,encoding=encoding,errors=errors,newline=newline,closefd=closefd,opener=opener)
  assert not any(c in mode for c in 'wax+');physical();stream=io.BytesIO(raw);return stream if 'b' in mode else io.TextIOWrapper(stream,encoding=encoding or 'utf8',errors=errors,newline=newline)
 def vstat(path,*a,**kw):
  if matches(path):physical();return MetaView(stat0(path,*a,**kw),old)
  return stat0(path,*a,**kw)
 def vlstat(path,*a,**kw):
  if matches(path):physical();return MetaView(lstat0(path,*a,**kw),old)
  return lstat0(path,*a,**kw)
 physical();builtins.open=io.open=vopen;os.stat=vstat;os.lstat=vlstat
 # DirEntry.stat bypasses os.stat. Layer scandir entries before archive helpers capture it.
 scan0=os.scandir
 class Entry:
  def __init__(self,e):self.e=e;self.path=e.path;self.name=e.name
  def __fspath__(self):return self.path
  def stat(self,**kw):return vstat(self.path,**kw)
  def is_file(self,**kw):return stat.S_ISREG(self.stat(**kw).st_mode)
  def is_dir(self,**kw):return stat.S_ISDIR(self.stat(**kw).st_mode)
  def is_symlink(self):return self.e.is_symlink()
  def inode(self):return self.stat().st_ino
 class Scan:
  def __init__(self,items):self.items=iter(items)
  def __iter__(self):return self
  def __next__(self):return next(self.items)
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def close(self):pass
 def vscan(path='.'):
  with scan0(path) as it:items=[Entry(e) if matches(e.path) else e for e in it]
  return Scan(items)
 os.scandir=vscan
 import glob
 glob._StringGlobber.scandir=staticmethod(vscan)
def prior_location_layer():
 # Authenticate and stack the unchanged accepted Gate location view.
 gate=ROOT.parent/'gate_power_v1';p=gate/'scripts/bootstrap.py';seal=J(gate/'complete.json');assert H(gate/'manifest.json')==seal['manifest_sha256'];assert H(p)==J(gate/'manifest.json')['files']['scripts/bootstrap.py']
 node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='location_layer');ns=dict(globals(),ROOT=gate);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p), 'exec'),ns);ns['location_layer']()
def views():
 location_layer();prior_location_layer();sys.path[:0]=[str(CORE/'scripts'),str(CORE/'audit/space_recovery_v1/scripts'),str(CORE/'audit/tier1_disk_policy_20261101'),str(CORE/'post_tier3_review_v1')]
 import prompt_policy_view,prompt_registry
 prompt_policy_view.refresh=prompt_registry.authenticate
 entry=CORE/'continuation_tier23_v1/entry.py';spec=importlib.util.spec_from_file_location('_core_sealed_views',entry);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.journal=journal;module.freeze_check();module.views()
 def popen(command,*args,**kwargs):
  if isinstance(command,(list,tuple)):
   cmd=list(map(str,command))
   if cmd and 'python' in pathlib.Path(cmd[0]).name:
    for i,arg in enumerate(cmd[1:],1):
     p=pathlib.Path(arg)
     if p.is_absolute() and arg.endswith('.py') and (p.is_relative_to(ROOT.parent) or p.is_relative_to(INPUT)):
      if p!=ROOT/'scripts/run_partial.py':cmd=cmd[:i]+[str(ROOT/'scripts/run_partial.py'),'--dispatch']+cmd[i:]
      break
   command=cmd
  return ORIGINAL_POPEN(command,*args,**kwargs)
 subprocess.Popen=popen
def core_modules():
 sys.path.insert(0,str(CORE/'scripts'));import common as c
 import tempfile
 os.environ.update(TMPDIR=str(ROOT/'runtime/tmp'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),MPLCONFIGDIR=str(ROOT/'runtime/mpl'));tempfile.tempdir=str(ROOT/'runtime/tmp')
 c.guard=lambda:None;c.publish=publish;c.write=write;c.replace=replace;c.event=event;c.count=count;c.checkpoint=checkpoint;c.budget_guard=budget;c.space=lambda start=False:df()['available_bytes']
 import data,accounting,missing_prices,kernel
 def no_learning(*a,**kw):raise PermissionError('Part A/B entry forbids real Q-learning/MC policy retraining')
 kernel.fit=no_learning;kernel.build=no_learning
 return c,data,accounting
def historical(label):
 c,data,accounting=core_modules();import historical as h,protected_pilot_and_prompt
 protected_pilot_and_prompt.install();h.RUN=ROOT
 h.publish=publish;h.write=write;h.checkpoint=checkpoint
 return h.historical(label)
def native_verification(label):
 for name in ['ou','residual_ou_real_v1','rf_cost_accounting_v1','ou_realstrength_v1','ou_qfix_v1']:
  checkpoint();p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(CORE/'scripts/check_inputs.py'),name],capture_output=True,text=True);publish(event('native_'+name+'_'+label).with_suffix('.log'),p.stdout+p.stderr);assert p.returncode==0,('Native verifier failed',name,p.stderr[-2000:])
def dispatch(script,args):
 validate_sources();install_guard();views()
 import runpy
 sys.path.insert(0,str(script.parent));sys.argv=[str(script),*args];runpy.run_path(str(script),run_name='__main__')
