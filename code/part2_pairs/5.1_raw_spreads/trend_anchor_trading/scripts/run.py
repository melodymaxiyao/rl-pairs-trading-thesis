"""One user-launched resumable pipeline; never called by preparation."""
from aio import *
import argparse
from control import registered,preflight,tests,finish,preserve_failure

def lock_pipeline():
    # Never write to an existing workspace task's lock or start another task.
    for p in PT.glob('*/pipeline.lock'):
        if p.parent==HERE:continue
        with p.open('rb') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another workspace task holds '+str(p))
            fcntl.flock(f,fcntl.LOCK_UN)
    f=(HERE/'pipeline.lock').open('rb');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);parser.add_argument('mode',nargs='?',choices=['run','verify'],default='run');args=parser.parse_args()
    assert args.workers==1,'This registered pipeline uses one worker; numerical libraries use one thread'
    if args.mode=='verify' or (HERE/'complete.json').exists():
        from verify_trend_anchor import verify
        return verify()
    guard();handle=lock_pipeline();stage='preflight'
    try:
        preflight();tests('before',full=True)
        from accounting import catalogue
        catalogue()
        from plans import WORLD
        from execution import job,RULES,BENCH,LADDER
        from reporting import part1,verify_part1,final
        from reproduction import continuity
        stage='B2_continuity'
        continuity()
        stage='Part1'
        for pool in ['P_EXP','P_RAND','P_QUAL']:
            pool_start=time.monotonic()
            for i,year in enumerate(YEARS):
                start=time.monotonic()
                for cell in RULES+BENCH:
                    for variant in ['primary','no_forced','fresh_beta']:
                        for world,rep in WORLD:job(pool,year,cell,world,rep,variant)
                if i==0:print('Measured Part1',pool,'first-fold seconds',round(time.monotonic()-start,1),'remaining estimate seconds',round((time.monotonic()-pool_start)*21,1),flush=True)
        continuity();part1();verify_part1()
        print('Part1 tables sealed. Part2 starts now.',flush=True)
        stage='Part2'
        for pool in ['P_RAND','P_EXP','P_QUAL']:
            for i,year in enumerate(YEARS):
                start=time.monotonic();verify_part1()
                for cell in LADDER:
                    for variant in ['primary','no_forced','fresh_beta']:
                        for world,rep in WORLD:job(pool,year,cell,world,rep,variant)
                if i==0:print('Measured Part2',pool,'first-fold seconds',round(time.monotonic()-start,1),'remaining estimate seconds',round((time.monotonic()-start)*21,1),'(excludes final DSR/PBO and protected checks)',flush=True)
        stage='report';final();tests('after',full=True)
        stage='seal';finish()
        from verify_trend_anchor import verify
        verify(check_current=False)
        print(str(HERE/'README.md'),flush=True)
        for line in read(HERE/'summary.json')['lines']:print(line,flush=True)
    except BaseException:preserve_failure(stage);raise
    finally:handle.close()

if __name__=='__main__':
    sys.path.insert(0,str(HERE));main()
