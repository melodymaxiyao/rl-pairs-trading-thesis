"""Exactly four user-authorized provenance exclusions, never computation inputs."""
from common import *
FOUR={
 'experiments/15_pair_trading/td_mechanism_spi_v1/complete.json',
 'experiments/15_pair_trading/td_mechanism_spi_v1/verify_td_mechanism_spi.py',
 'experiments/15_pair_trading/trend_anchor_td_tuned_v1/complete.json',
 'experiments/15_pair_trading/trend_anchor_td_tuned_v1/verify_td_tuned_resume_v3.py'}
def policy():
 p=RUN/'addendum_002_four_upstream_provenance.json';a=json.loads(p.read_text());rows={r['path']:r for r in a['entries']}
 assert len(a['entries'])==len(rows)==4 and set(rows)==FOUR and a['status']=='AUTHORIZED_BY_USER'
 registry=json.loads((RUN/'audit/input_registration.json').read_text())['source_snapshots']
 for rel,row in rows.items():
  assert row['provenance_only'] and not row['numerical_computation_import'] and not row['numerical_computation_read']
  assert row['sealed_sha256']==registry[rel]['sha256'] and row['size']==registry[rel]['size']
  copy=RUN/'vendor'/rel;assert not copy.is_symlink() and copy.stat().st_size==row['size'] and sha(copy)==row['sealed_sha256']
  if rel.endswith('complete.json'):assert row['sealed_sha256']==spec()['input_seals'][Path(rel).parent.name]
 return rows
def assess(path,row):
 """The later standing rule now logs visible metadata drift for these four too."""
 from registered_scope import assess_one
 result=assess_one(path,row);result['status']='CONDITIONALLY_UNAVAILABLE_PROVENANCE' if result['content_hash_excluded'] else 'MATERIALIZED_SEALED_HASH_PASS';result['size']=row['size'];return result
def snapshot():return {rel:assess(INPUT/rel,row) for rel,row in policy().items()}
def unavailable():return {rel:r for rel,r in snapshot().items() if r['content_hash_excluded']}
