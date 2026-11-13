"""Date-first CSV admission, with boolean-only test adjusted-close inspection."""
import csv
import io
import mmap
from pathlib import Path

MISSING = frozenset((b'',b'na',b'n/a',b'nan',b'null',b'none',b'#n/a',b'<na>',
                     b'#n/a n/a',b'#na',b'-1.#ind',b'-1.#qnan',b'-nan',b'1.#ind',b'1.#qnan'))

def present_token(token):
    # Presence only: no float conversion, price comparison or volume access.
    return token.strip(b' \r\n"').lower() not in MISSING

def adjclose_present(data, start, end, column):
    position = start
    for _ in range(column):
        comma = data.find(b',', position, end)
        if comma < 0:
            return False
        position = comma + 1
    stop = data.find(b',', position, end)
    if stop < 0:
        stop = end
    return present_token(data[position:stop])

def admit(path, test_year, cutoff_year=None):
    """Return admitted train CSV bytes and (date, presence) test rows.

    Test rows are never copied into a dataframe/parser. Delimiter scans locate
    only the date and adjusted-close field. Numeric admission can be cut off at
    Y−1 for the inner screen, before any log, eligibility or F computation.
    """
    cutoff = test_year if cutoff_year is None else cutoff_year
    assert cutoff <= test_year
    admitted, test = [], []
    with Path(path).open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            first = data.find(b'\n')
            assert first >= 0
            header = data[:first + 1]
            columns = next(csv.reader([header.decode().strip()]))
            assert columns[0] == 'Date' and 'Adj Close' in columns
            adj = columns.index('Adj Close')
            position = first + 1
            while position < len(data):
                end = data.find(b'\n', position)
                end = len(data) if end < 0 else end
                comma = data.find(b',', position, end)
                if comma < 0:
                    assert end == position, 'Malformed date row'
                    position = end + 1
                    continue
                date = data[position:comma].decode().strip('"\r ')
                year = int(date[:4])
                if year == test_year:
                    test.append((date, adjclose_present(data, position, end, adj)))
                if 2000 <= year < cutoff:
                    admitted.append(data[position:end] + b'\n')
                position = end + 1
    return header + b''.join(admitted), test

def read_window(path, test_year, cutoff_year=None):
    import pandas as pd
    raw, test = admit(path, test_year, cutoff_year)
    frame = pd.read_csv(io.BytesIO(raw), parse_dates=['Date'], float_precision='round_trip').set_index('Date')
    if frame.empty:
        # A stored ticker can have no pre-cutoff prices. Pandas otherwise leaves
        # header-only columns as object, including their reindexed NaNs. Keep
        # those missing observations numeric so the original eligibility check
        # can reject insufficient history without np.log raising a TypeError.
        frame = frame.astype({name: 'float64' for name in ('Adj Close', 'Close', 'Volume')})
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    assert frame.index.is_unique, 'Duplicate train dates'
    assert (frame.index.year < (test_year if cutoff_year is None else cutoff_year)).all()
    return frame.sort_index(), test

def complete_test(rows, sessions):
    dates = [d[:10] for d, _ in rows]
    expected = set(str(d)[:10] for d in sessions)
    present = {d[:10] for d, p in rows if p}
    return bool(expected) and len(dates) == len(set(dates)) and expected <= present
