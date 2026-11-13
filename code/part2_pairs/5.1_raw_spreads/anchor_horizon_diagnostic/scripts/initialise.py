"""First full protected snapshot; no price parsing or research computation."""
from aio import *
from protection import protected
from verification import finder_inventory

def main():
    guard();start=time.monotonic();value=protected()
    write(HERE/'audit/protected_before.json',value)
    # Register all source/prereg/config and fresh-data files plus sampled rows.
    wanted={k:v for k,v in value['files'].items() if k.endswith('.py') or k.endswith('/preregistration.json')
            or k.startswith('configs/') or k.startswith(str(FRESH.relative_to(ROOT))+'/')
            or k.startswith(str(PROPERTY.relative_to(ROOT))+'/runtime/checkpoints/sample_')}
    seals=str((BASELINE/'source_code_seals.json').relative_to(ROOT));wanted[seals]=value['files'][seals]
    write(HERE/'audit/source_baseline.json',dict(files=wanted))
    write(HERE/'audit/finder_before.json',finder_inventory())
    write(HERE/'audit/README_before_numeric_checks.json',dict(fingerprint=fp(HERE/'README.md')))
    print('Initial protected files',len(value['files']),'seconds',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
