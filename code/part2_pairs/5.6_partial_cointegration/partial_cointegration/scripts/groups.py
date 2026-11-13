"""Immutable compact group receipts; no source writes."""
import gzip,io,resource
import numpy as np,pandas as pd
from io_utils import *
def done(root):
 p=root/'complete.json'
 if not p.exists():return False
 seal=J(p);assert seal['preregistration_sha256']==H(ROOT/'preregistration.json')
 assert {str(p.relative_to(ROOT)) for p in root.rglob('*') if p.is_file()}==set(seal['files'])|{str(p.relative_to(ROOT))}
 for rel,digest in seal['files'].items():assert H(ROOT/rel)==digest,rel
 return True
def finish(root,outputs,**metadata):
 files={}
 for name,value in outputs.items():
  if isinstance(value,pd.DataFrame):raw=value.to_csv(index=False,float_format='%.17g').encode();raw=gzip.compress(raw,compresslevel=6,mtime=0) if name.endswith('.gz') else raw
  elif isinstance(value,bytes):raw=value
  else:raw=encode(value)
  publish(root/name,raw);files[str((root/name).relative_to(ROOT))]=H(root/name)
 write(root/'complete.json',dict(status='complete',files=files,preregistration_sha256=H(ROOT/'preregistration.json'),**metadata));journal('group_committed',group=str(root.relative_to(ROOT)),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,df=df())
 return dict(root=str(root))
def array_bytes(**arrays):
 stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()
def sealed_frame(kind,year,learner,name):
 rel=f'outputs/cells/{kind}_{year}_G0_c5_{learner}/{name}';mf=J(CORE/'tier_3/manifest.json')['files'];assert H(CORE/rel)==mf[rel];return pd.read_csv(CORE/rel,float_precision='round_trip')
