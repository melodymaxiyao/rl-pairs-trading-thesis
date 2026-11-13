"""Train-only price admission: stop at the first row of the test year.

For fold Y the parser compares only the four-character year prefix of each row.
At the first row whose year is >= Y it stops; that row and every later row are
never decoded, parsed, stored or counted. Rows before 2000 are skipped exactly
as the universe reader did, and the admitted bytes are parsed with the same
pandas call, so admitted frames equal the universe's train frames.
Correctness relies on the sealed fresh data_quality.csv dates_sorted=True;
admitted dates are asserted strictly increasing.
"""
import csv
import io
from pathlib import Path

def admit_train(path, test_year):
    admitted, previous = [], None
    with Path(path).open('rb') as stream:
        header = stream.readline()
        assert header.endswith(b'\n')
        columns = next(csv.reader([header.decode().strip()]))
        assert columns[0] == 'Date' and 'Adj Close' in columns
        for line in stream:
            if b',' not in line:
                assert not line.strip(b'\r\n'), 'Malformed date row'
                continue
            year = int(line.lstrip(b'"')[:4])
            if year >= test_year:
                break
            if year < 2000:
                continue
            date = line[:line.index(b',')].strip(b'"\r ')
            assert previous is None or date > previous, ('Unsorted or duplicate train dates', str(path))
            previous = date
            admitted.append(line if line.endswith(b'\n') else line + b'\n')
    return header, admitted

def read_train(path, test_year):
    import pandas as pd
    header, rows = admit_train(path, test_year)
    frame = pd.read_csv(io.BytesIO(header + b''.join(rows)), parse_dates=['Date'], float_precision='round_trip').set_index('Date')
    if frame.empty:
        frame = frame.astype({name: 'float64' for name in ('Adj Close', 'Close', 'Volume')})
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    assert frame.index.is_unique, 'Duplicate train dates'
    assert (frame.index.year < test_year).all()
    return frame.sort_index(), len(rows)
