"""Validate symlink target content before accepting a timestamp-only difference."""
import csv,gzip,hashlib,json,os,stat,time
from contextlib import contextmanager
from pathlib import Path

DATALESS=0x40000000
def read(p):return json.loads(Path(p).read_text())
def digest_file(p):
 p=Path(p);before=p.stat();assert not before.st_flags&DATALESS,('Dataless target cannot be read',p)
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);n+=len(block)
 after=p.stat();assert n==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,('Target changed while hashing',p)
 return h.hexdigest(),n
def in_roots(rel,trie):
 node=trie
 for part in Path(rel).parts:
  if None in node:return True
  node=node.get(part)
  if node is None:return False
 return None in node
def external_digest(root):
 root=Path(root);h=hashlib.sha256();count=0;size=0
 for base,dirs,files in os.walk(root,followlinks=False):
  base=Path(base);dirs.sort();files.sort()
  for name in dirs+files:
   p=base/name;kind='symlink' if p.is_symlink() else 'directory' if p.is_dir() else 'file'
   if kind=='symlink':digest=hashlib.sha256(os.fsencode(os.readlink(p))).hexdigest()
   elif kind=='file':digest,n=digest_file(p);size+=n
   else:digest=''
   h.update(json.dumps([str(p.relative_to(root)),kind,digest],separators=(',',':')).encode()+b'\n');count+=1
  dirs[:]=[d for d in dirs if not (base/d).is_symlink()]
 return dict(sha256=h.hexdigest(),entries=count,file_bytes=size)
def verify_external_targets(root):
 audit=Path(root)/'experiments/15_pair_trading/fyp_relocation_v1';complete=read(audit/'complete.json');frozen=read(audit/'audit.json')
 assert digest_file(audit/'audit.json')[0]==complete['audit_sha256']
 path=audit/'external_symlink_target_hashes.json';assert digest_file(path)[0]==frozen['artifact_sha256'][path.name]
 result={}
 for name,old in read(path).items():
  p=Path(name);assert p.exists(),('External target disappeared',name)
  if old['kind']=='file_content_sha256':h,n=digest_file(p);value=dict(sha256=h,file_bytes=n)
  else:value=external_digest(p)
  assert value['sha256']==old['sha256'],('External target content changed',name)
  if 'entries' in old:assert value['entries']==old['entries']
  result[name]=value
 return result
def validate(root,paths,output=None,allow_outside_additions=False):
 """Use the authenticated relocation inventory as a target-content baseline.

 The one-time repair requires exact member sets. Later callers can permit
 additions outside historically sealed inputs, as the standing historical
 scope already allows; existing files and link targets always remain exact.
 """
 root=Path(root);audit=root/'experiments/15_pair_trading/fyp_relocation_v1'
 own=Path(__file__).resolve().parents[1]
 exclude_own=own.is_relative_to(root)
 complete=read(audit/'complete.json');assert digest_file(audit/'audit.json')[0]==complete['audit_sha256']
 frozen=read(audit/'audit.json')
 for name in ['tree_manifest.csv.gz','symlinks.json','external_symlink_target_hashes.json']:
  assert digest_file(audit/name)[0]==frozen['artifact_sha256'][name],('Frozen target baseline changed',name)
 links={r['path']:r for r in read(audit/'symlinks.json')};external=read(audit/'external_symlink_target_hashes.json')
 roots=set();outer=[];trie={};external_roots=set()
 for rel in paths:
  assert rel in links,('No frozen target baseline',rel)
  old=links[rel];p=root/rel;st=p.lstat();assert stat.S_ISLNK(st.st_mode),('Link no longer exists as a symlink',rel)
  assert os.readlink(p)==old['target'],('Target string changed',rel)
  assert stat.S_IMODE(st.st_mode)==old['lstat']['mode'],('Symlink mode changed',rel)
  assert p.exists()==old['target_exists_before'] and p.exists(),('Target existence changed',rel)
  q=p.resolve(strict=True);assert str(q)==old['resolved_before'],('Resolved target changed',rel)
  outer.append(dict(path=rel,target=old['target'],resolved=str(q),frozen_mtime_ns=old['lstat']['mtime_ns'],actual_mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode)))
  if q.is_relative_to(root):roots.add(str(q.relative_to(root)))
  else:external_roots.add(str(q))
 for rel in roots:
  node=trie
  for part in Path(rel).parts:node=node.setdefault(part,{})
  node[None]=True
 expected={};sealed=set();finder=[]
 with gzip.open(audit/'tree_manifest.csv.gz','rt',newline='') as f:
  for row in csv.DictReader(f):
   if Path(row['path']).name=='complete.json':sealed.add(str(Path(row['path']).parent))
   if in_roots(row['path'],trie):
    if Path(row['path']).name=='.DS_Store':finder.append(row['path']);continue
    expected[row['path']]=row
 actual={};visited=set();last=time.monotonic();file_bytes=0
 for rel in sorted(roots,key=lambda x:(len(Path(x).parts),x)):
  p=root/rel
  if rel in visited:continue
  nodes=[p]
  if p.is_dir():
   for base,dirs,files in os.walk(p,followlinks=False):
    base=Path(base);dirs.sort();files.sort();nodes.extend(base/n for n in dirs+files)
    dirs[:]=[d for d in dirs if not (base/d).is_symlink() and not (exclude_own and base/d==own)]
  for q in nodes:
   name=str(q.relative_to(root))
   if name in visited or q.name=='.DS_Store' or (exclude_own and q.is_relative_to(own)):continue
   visited.add(name);st=q.lstat();kind='symlink' if q.is_symlink() else 'directory' if q.is_dir() else 'file'
   old=expected.get(name)
   if old is None:
    assert allow_outside_additions and not any(str(p) in sealed for p in Path(name).parents),('Target member added',name)
   else:
    assert kind==old['kind'] and stat.S_IMODE(st.st_mode)==int(old['mode']),('Target member type/mode changed',name)
   if kind=='file':
    digest,n=digest_file(q);file_bytes+=n
    if old:
     assert n==int(old['size']) and digest==old['sha256'],('Target file content or size changed',name)
     assert st.st_mtime_ns==int(old['mtime_ns']),('Regular target file timestamp changed',name)
    actual[name]=dict(kind=kind,sha256=digest,size=n)
   elif kind=='symlink':
    target=os.readlink(q);digest=hashlib.sha256(os.fsencode(target)).hexdigest()
    if old:assert target==old['target'] and digest==old['sha256'],('Nested target string changed',name)
    if name in links:assert q.exists()==links[name]['target_exists_before'],('Nested target existence changed',name)
    actual[name]=dict(kind=kind,sha256=digest,size=st.st_size)
   else:actual[name]=dict(kind=kind,sha256='',size=0)
   if time.monotonic()-last>25:print('Verified symlink target members',len(actual),flush=True);last=time.monotonic()
 assert not expected.keys()-actual.keys(),('Target member removed',sorted(expected.keys()-actual.keys())[:10])
 ext={}
 for name in sorted(external_roots):
  q=Path(name)
  if q.is_file():
   digest,n=digest_file(q);value=dict(sha256=digest,file_bytes=n,kind='file_content_sha256')
  else:value=dict(external_digest(q),kind='directory_member_content_sha256')
  assert name in external and value['sha256']==external[name]['sha256'],('External target content changed',name)
  if 'entries' in external[name]:assert value['entries']==external[name]['entries']
  ext[name]=value
 canonical=hashlib.sha256()
 for name,value in sorted(actual.items()):canonical.update(json.dumps([name,value],sort_keys=True,separators=(',',':')).encode()+b'\n')
 report=dict(status='PASS',symlinks=len(paths),links=outer,internal_target_members=len(actual),internal_file_bytes=file_bytes,
             canonical_verified_target_members_sha256=canonical.hexdigest(),external_targets=ext,finder_excluded=finder,
             frozen_tree_manifest_sha256=complete['tree_manifest_sha256'],regular_target_content_and_size='exact',source_writes=0)
 if output:
  with Path(output).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 return report

class StatView:
 def __init__(self,value,mtime):self.value=value;self.mtime=mtime
 def __getattr__(self,key):
  if key=='st_mtime_ns':return self.mtime
  if key=='st_mtime':return self.mtime/1e9
  return getattr(self.value,key)
 def __getitem__(self,key):
  values=list(self.value);values[8]=self.mtime/1e9;return values[key]
 def __iter__(self):
  values=list(self.value);values[8]=self.mtime/1e9;return iter(values)
 def __len__(self):return len(self.value)

@contextmanager
def canonical_mtimes(root):
 """Task-local read view for old strict verifiers, after full target validation.

 No original verifier/source or filesystem metadata is edited. Only a verified
 symlink's mtime is presented at its frozen value; regular stat results and all
 other fields remain actual. Each use is emitted into the verifier log.
 """
 root=Path(root);audit=root/'experiments/15_pair_trading/fyp_relocation_v1'
 links={str(root/r['path']):r for r in read(audit/'symlinks.json')}
 raw_stat,raw_lstat=os.stat,os.lstat;verified={};busy=False
 def view(path,value,follow=True,dir_fd=None):
  nonlocal busy
  if follow or busy or not stat.S_ISLNK(value.st_mode) or not isinstance(path,(str,bytes,os.PathLike)):return value
  if dir_fd is not None and not Path(os.fsdecode(path)).is_absolute():return value
  key=os.path.abspath(os.fsdecode(path));old=links.get(key)
  if old is None or value.st_mtime_ns==old['lstat']['mtime_ns']:return value
  if key not in verified:
   busy=True
   try:proof=validate(root,[old['path']],allow_outside_additions=True)
   finally:busy=False
   verified[key]=True
   print('SYMLINK_TIMESTAMP_ACCEPTED '+json.dumps(dict(path=old['path'],frozen_mtime_ns=old['lstat']['mtime_ns'],observed_mtime_ns=value.st_mtime_ns,
       target_members_sha256=proof['canonical_verified_target_members_sha256'],external_targets=proof['external_targets'],frozen_tree_manifest_sha256=proof['frozen_tree_manifest_sha256']),sort_keys=True),flush=True)
  return StatView(value,old['lstat']['mtime_ns'])
 def stat_call(path,*args,**kw):return view(path,raw_stat(path,*args,**kw),kw.get('follow_symlinks',True),kw.get('dir_fd'))
 def lstat_call(path,*args,**kw):return view(path,raw_lstat(path,*args,**kw),False,kw.get('dir_fd'))
 os.stat,os.lstat=stat_call,lstat_call
 try:yield
 finally:os.stat,os.lstat=raw_stat,raw_lstat
