"""Run N3 only after the authorized Item3 process seals its final report."""
import argparse,hashlib,json,os,pathlib,time
P=pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1')
HERE=P/'audit/N3_intercept_null'
parser=argparse.ArgumentParser();parser.add_argument('--wait-for-pid',type=int);args=parser.parse_args()
reg=json.loads((HERE/'code_registration.json').read_text())
for rel,digest in reg['files'].items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==digest,rel
receipt=P/'outputs/A1_sanity_v2/report/complete.json'
while not receipt.exists():
    if args.wait_for_pid is None:raise SystemExit('Item3 is not sealed; finish Item3 first. No N3 computation started.')
    try:os.kill(args.wait_for_pid,0)
    except ProcessLookupError:raise SystemExit('Item3 stopped without a final receipt. N3 not started; completed work retained.')
    print('N3 waiting for Item3 final receipt; no diagnostic fits started.',flush=True)
    time.sleep(30)
value=json.loads(receipt.read_text())
assert value['status']=='complete'
for rel,digest in value['files'].items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==digest,rel
print('Item3 final report verified; starting registered N3 diagnostic. Part C remains closed; trials=580.',flush=True)
os.execv('/Users/maxiyao/Desktop/FYP/.venv/bin/python',['/Users/maxiyao/Desktop/FYP/.venv/bin/python','-I','-B',str(HERE/'run.py')])
