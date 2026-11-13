"""Copy only authenticated resident numerical source bytes; never repair original paths."""
from pathlib import Path
import json,hashlib,stat,ast
RUN=Path(__file__).resolve().parents[1];ROOT=Path('/Users/maxiyao/Desktop/FYP')
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def checked(p,digest):
 assert stat.S_ISREG(p.lstat().st_mode) and not p.lstat().st_flags&0x40000000
 assert H(p)==digest,('Source hash mismatch',str(p));return p.read_bytes()
def main():
 sources=json.loads((RUN/'audit/input_seals.json').read_text());records={}
 for name,select in {
 'ou':['scripts/simulation.py','scripts/native.py','scripts/evaluation.py','scripts/dp.py','scripts/common.py','scripts/rollout.c','runtime/ou_kernel.c','runtime/ou_kernel.so','vendor/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/original_q_kernel.c','audit/pip_freeze_all.txt','audit/kernel_import_proof.json'],
 'residual':['scripts/models.py','scripts/stats.py','scripts/aggregate.py','outputs/tables/residual_transfer.json','audit/pip_freeze_all.txt'],
 'cost':['scripts/common.py','scripts/integrity.py','scripts/standing_wheel.py','scripts/verify.py']}.items():
  seal=sources[name];root=Path(seal['root']);assert H(root/'complete.json')==seal['sealed_completion_sha256'] and H(root/'audit.json')==seal['sealed_audit_sha256'] and H(root/'code_registration.json')==seal['sealed_code_registration_sha256']
  a=json.loads((root/'audit.json').read_text());c=json.loads((root/'code_registration.json').read_text());table=dict(a.get('artifacts',{}))|c['files']
  if name=='cost':
   m=json.loads((root/'manifest.json').read_text());assert H(root/'manifest.json')==json.loads((root/'complete.json').read_text())['manifest_sha256'];table.update(m['files']);assert table['code_registration.json']==H(root/'code_registration.json')
  else:assert a['code_registration_sha256']==H(root/'code_registration.json')
  for rel in select:
   raw=checked(root/rel,table[rel]);dest=RUN/'vendor'/name/rel;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw);records[name+'/'+rel]=dict(source=str(root/rel),sha256=table[rel],size=len(raw))
 # Already authenticated SPI receipt from the final real-mechanism seal.
 receipt=json.loads((RUN/'audit/spi_inference_receipt.json').read_text());row=receipt['files']['experiments/15_pair_trading/td_mechanism_spi_v1/scripts/inference.py'];p=RUN/'vendor/spi/inference.py';assert H(p)==row['sha256'];records['spi/inference.py']=dict(source=row['snapshot'],sha256=row['sha256'],size=row['size'],authority='real_mechanism final audit source_snapshot_receipt SHA256')
 for name,digest in [('learning.py','f5996643672cf8b7f6d9675ff558592db6f42308795805c1128720d7d27c4c6d'),('policies.py','922971f189973a545c86f0c0a8f62584a16ab2afd9cbcda3b06fea61407d437e')]:
  p=RUN/'vendor/spi'/name;assert H(p)==digest;records['spi/'+name]=dict(source='authenticated resident session log, line722',sha256=digest,size=p.stat().st_size)
 (RUN/'audit/import_registration.json').write_text(json.dumps(dict(status='PASS_EXACT_SEALED_NUMERICAL_BYTES',files=records,source_writes=0,placeholder_reads=0),sort_keys=True,indent=2)+'\n')
 # IO facade inherits the same guards and sealed interpreter checks as the preceding audit.
 text=(RUN/'vendor/cost/scripts/common.py').read_text();text=text.replace("p=spec();self.complete=", "p=read(RUN/'audit/input_seals.json')['residual'];self.complete=")
 # No production cache before the design has actually been frozen.
 text+='''\nimport numpy as np\nfrom collections import Counter\nCOUNTS=Counter()\ndef count(operation,n=1):\n assert isinstance(n,int) and n>=0\n scope=os.environ.get('OU_SCOPE','preparation')\n with (RUN/'audit'/('counts_'+scope+'_'+str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps([operation,n])+'\\n')\ndef checkpoint_guard():checkpoint()\n'''
 ou=(RUN/'vendor/ou/scripts/common.py').read_text();nodes=ast.parse(ou).body
 for name in ['stable_seed','td_seed']:
  node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name);text+='\n'+ast.get_source_segment(ou,node)+'\n'
 (RUN/'scripts/common.py').write_text(text)
 (RUN/'runtime/ou_kernel.so').write_bytes((RUN/'vendor/ou/runtime/ou_kernel.so').read_bytes())
 print('Authenticated imports:',len(records),'old trainer binary unchanged')
if __name__=='__main__':main()
