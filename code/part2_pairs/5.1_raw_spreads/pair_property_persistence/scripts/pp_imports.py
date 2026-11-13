"""Load the universe run's screen, event, F/statistics and null-generator code unchanged.

The universe modules import `universe_io`, `train_reader` and `checkpoints` by
name. Only those namespaces are substituted: `universe_io` is this task's IO
module, and the universe's test-date reader and checkpoint loaders are replaced
by stubs that raise, so no universe code path can read a test year or a
universe runtime cache from this process.
"""
from ppio import *
import types

UNIVERSE_MODULES = ['screen', 'events', 'numeric', 'worker']

@lru_cache(None)
def universe():
    reader = types.ModuleType('train_reader')
    for name in ['read_window', 'complete_test', 'admit', 'adjclose_present']:
        setattr(reader, name, forbidden)
    checkpoints = types.ModuleType('checkpoints')
    for name in ['ready', 'save_rows', 'load_rows', 'load_context', 'save_context', 'receipt']:
        setattr(checkpoints, name, forbidden)
    io = sys.modules['ppio']
    loaded = {}
    with aliases(universe_io=io, train_reader=reader, checkpoints=checkpoints):
        for name in UNIVERSE_MODULES:
            loaded[name] = load(UNIV / 'scripts' / (name + '.py'), name)
    numeric = loaded['numeric']
    for name in ['FoldData', 'evaluate_pair']:
        setattr(numeric, name, forbidden)
    for name in ['real_chunk', 'null_chunk', 'context']:
        setattr(loaded['worker'], name, forbidden)
    return types.SimpleNamespace(
        make_episodes=numeric.make_episodes,
        episode_statistics=numeric.episode_statistics,
        screen_events=loaded['events'].screen_events,
        summarize=loaded['events'].summarize,
        km_quantiles=loaded['screen'].km_quantiles,
        simulate_logs=loaded['worker'].simulate_logs,
        modules=loaded)
