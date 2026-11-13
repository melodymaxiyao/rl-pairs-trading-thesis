"""Shared turning-point detection metrics."""
from __future__ import annotations

import numpy as np


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Return standard event-detection metrics with explicit empty handling."""
    if min(tp, fp, fn) < 0:
        raise ValueError("tp, fp, and fn must be non-negative")
    predicted = tp + fp
    actual = tp + fn
    precision = float(tp / predicted) if predicted else (np.nan if actual == 0 else 0.0)
    recall = float(tp / actual) if actual else (np.nan if predicted == 0 else 0.0)
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "F1": f1}

