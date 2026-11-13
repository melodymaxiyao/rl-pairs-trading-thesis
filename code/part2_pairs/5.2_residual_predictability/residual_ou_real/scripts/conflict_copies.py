"""Metadata/manifest-only iCloud conflict-copy recognition. Never open the copy."""
from common import *
import re,csv
CONTROL=Path('/Users/maxiyao/FYP_runs/fyp_icloud_conflict_copies_v1')
PATTERN=re.compile(r'^(?P<stem>.+) (?P<number>[0-9]+)(?P<extension>\.[^/\s]+)$')
def original_name(name):
 m=PATTERN.fullmatch(name)
 return (m['stem']+m['extension'],int(m['number'])) if m and int(m['number'])>=2 else None
class Index:
 def __init__(self,root,frozen,inventory_proof,completion_snapshots=None,known_seals=None,additional_sealed=()):
  self.root=Path(root);self.frozen=frozen;self.inventory_proof=inventory_proof;self.snapshots=completion_snapshots or {};self.known_seals=known_seals or {};self.cache={};self.ignored={};self.additional_sealed=set(additional_sealed)
 def _manifest(self,root):
  if root in self.cache:return self.cache[root]
  complete=root/'complete.json';rel=str(complete.relative_to(self.root));expected=self.frozen.get(rel,{}).get('sha256') or self.known_seals.get(root.name)
  selected=None
  try:s=complete.lstat()
  except FileNotFoundError:s=None
  if s is not None and not s.st_flags&DATALESS:selected=complete
  elif rel in self.snapshots:
   p,digest=self.snapshots[rel];assert expected is None or digest==expected;selected=Path(p);expected=digest
  elif s is None and expected is None:self.cache[root]=None;return None
  if selected is None or expected is None:self.cache[root]=False;return False
  assert sha(selected)==expected,('Conflict-rule completion seal mismatch',rel)
  c=json.loads(selected.read_text());assert c['status']=='complete';name=c.get('manifest_file',c.get('manifest','manifest.json'));assert isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts
  manifest=root/name
  try:s=manifest.lstat()
  except FileNotFoundError:self.cache[root]=False;return False
  if s.st_flags&DATALESS:self.cache[root]=False;return False
  assert sha(manifest)==c['manifest_sha256'],('Conflict-rule manifest seal mismatch',str(manifest))
  value=json.loads(manifest.read_text());files=value['files'];assert isinstance(files,dict)
  result=dict(root=root,files=files,complete=str(selected),complete_sha256=expected,manifest=str(manifest),manifest_sha256=c['manifest_sha256']);self.cache[root]=result;return result
 def classify(self,path,st=None):
  path=Path(path);match=original_name(path.name)
  if match is None:return None
  rel=str(path.relative_to(self.root))
  if rel in self.frozen or rel in self.additional_sealed:return None # A numeric suffix never excuses a sealed file.
  if rel in self.ignored:
   st=path.lstat() if st is None else st
   return self.ignored[rel] if stat.S_ISREG(st.st_mode) else None
  st=path.lstat() if st is None else st
  if not stat.S_ISREG(st.st_mode):return None
  original=path.with_name(match[0]);original_rel=str(original.relative_to(self.root));proofs=[];sealed_original=None;unresolved=False
  for parent in path.parents:
   if not parent.is_relative_to(self.root):break
   m=self._manifest(parent)
   if m is False:unresolved=True;continue
   if m:
    copy_key=str(path.relative_to(parent));original_key=str(original.relative_to(parent))
    if copy_key in m['files']:return None
    if original_key in m['files']:
     v=m['files'][original_key];sealed_original=v.get('sha256') if isinstance(v,dict) else v;proofs.append({k:v for k,v in m.items() if k not in ['files','root']})
   if parent==self.root:break
  if unresolved:return None # No guess about an unavailable sealed manifest.
  if sealed_original is None:
   r=self.frozen.get(original_rel)
   if not r or r.get('kind')!='file':return None
   sealed_original=r['sha256'];proofs.append(dict(sealed_inventory=self.inventory_proof))
  record=dict(path=rel,original_path=original_rel,copy_number=match[1],status='iCloud conflict copy; non-sealed; ignored; never read',original_sealed_sha256=sealed_original,sealed_membership_proof=proofs,copy_absent_from_authenticated_manifests=True,metadata=dict(size=st.st_size,mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns,ino=st.st_ino,flags=st.st_flags),copy_content_read=False,copy_content_hash=None,deleted=False)
  self.ignored[rel]=record;deny_conflict_read(path);return record
 def persist(self,label):
  p=RUN/f'audit/icloud_conflict_copies_{label}_{time.time_ns()}.json';write(p,dict(rule='non-sealed numeric-suffix copy; same-directory original sealed',copies=list(self.ignored.values()),count=len(self.ignored),copy_reads=0,copy_deletions=0));return dict(file=str(p.relative_to(RUN)),sha256=sha(p),count=len(self.ignored))
def index_for(root,frozen,inventory_proof):
 registration=json.loads((RUN/'audit/input_registration.json').read_text())['source_snapshots'];snapshots={rel:(RUN/'vendor'/rel,r['sha256']) for rel,r in registration.items() if rel.endswith('/complete.json')}
 from registered_scope import rows
 protected=[rel for rel,r in rows().items() if r.get('sealed_sha256')]
 return Index(root,frozen,inventory_proof,snapshots,spec()['input_seals'],protected)
@contextmanager
def ignored_inventory_view(proof_path=None,expected_sha=None):
 """Exact authorized omissions for unchanged legacy artifact verifiers."""
 if proof_path is None:yield;return
 proof_path=Path(proof_path);assert proof_path.resolve().is_relative_to(RUN) and sha(proof_path)==expected_sha
 proof=json.loads(proof_path.read_text());names=set()
 for row in proof['copies']:
  assert row['status']=='iCloud conflict copy; non-sealed; ignored; never read' and not row['copy_content_read'] and row['copy_absent_from_authenticated_manifests']
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
  path=INPUT/rel;names.add(os.path.abspath(path));deny_conflict_read(path)
 glob,rglob,walk=Path.glob,Path.rglob,os.walk
 def filtered_glob(path,*a,**kw):return (p for p in glob(path,*a,**kw) if os.path.abspath(p) not in names)
 def filtered_rglob(path,*a,**kw):return (p for p in rglob(path,*a,**kw) if os.path.abspath(p) not in names)
 def filtered_walk(*a,**kw):
  for parent,dirs,files in walk(*a,**kw):yield parent,dirs,[n for n in files if os.path.abspath(Path(parent)/n) not in names]
 Path.glob,Path.rglob,os.walk=filtered_glob,filtered_rglob,filtered_walk
 try:yield
 finally:Path.glob,Path.rglob,os.walk=glob,rglob,walk
