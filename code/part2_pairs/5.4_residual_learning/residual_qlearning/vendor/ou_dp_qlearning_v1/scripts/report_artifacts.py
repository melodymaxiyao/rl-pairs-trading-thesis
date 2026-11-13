"""Immutable compact report publication with bounded temporary storage."""
from common import *
from disk_policy import require_space

def stream_policy_json(path, policies):
    # Encode one world at a time, using C's compact JSON encoder. All values
    # match clean()/write(); only whitespace differs from earlier tier exports.
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending_' + str(time.time_ns()))
    try:
        with pending.open('xb') as f:
            f.write(b'{')
            for index, key in enumerate(sorted(policies)):
                checkpoint_guard()
                data = ((',' if index else '') + json.dumps(str(key)) + ':' +
                        json.dumps(clean(policies[key]), sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)).encode()
                # Reserve the approved floor before each bounded world write.
                if require_space('checkpoint') - len(data) < 5 * 2**30:
                    raise Pause('Report would cross5GiB; checkpoints retained')
                f.write(data)
            f.write(b'}\n'); f.flush(); os.fsync(f.fileno())
        checkpoint_guard()
        try:
            os.link(pending, path)
        except FileExistsError:
            assert sha(path) == sha(pending), ('Existing artifact differs; not overwritten', str(path))
    finally:
        # Only our uncommitted report fragment is removed. Checkpoints and
        # already published artifacts are never overwritten or deleted.
        if pending.exists():
            pending.unlink()

def share_final_report(base):
    base = inside(base)
    paths = [base / 'RESULTS.md'] + sorted(p for p in (base / 'outputs').rglob('*') if p.is_file())
    for source in paths:
        checkpoint_guard()
        target = inside(RUN / source.relative_to(base))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            assert sha(source) == sha(target), ('Existing root report differs; not overwritten', str(target))
