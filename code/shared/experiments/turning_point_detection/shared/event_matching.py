"""Deterministic one-to-one matching for ordered turning-point events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMatchResult:
    """Matched index pairs and the unmatched indices from both sequences."""

    matched_pairs: tuple[tuple[int, int], ...]
    unmatched_predicted: tuple[int, ...]
    unmatched_oracle: tuple[int, ...]
    total_absolute_error: int


def monotone_one_to_one_match(
    predicted_positions: np.ndarray | list[int],
    oracle_positions: np.ndarray | list[int],
    *,
    tolerance: int,
) -> EventMatchResult:
    """Match ordered events, maximizing count before minimizing total error.

    Inputs must be strictly increasing trading-day positions.  The dynamic
    program preserves order, so crossing matches cannot occur.  Ties after the
    two stated objectives prefer a direct match, then skipping a prediction,
    which makes the result deterministic.
    """
    predicted = np.asarray(predicted_positions, dtype=np.int64)
    oracle = np.asarray(oracle_positions, dtype=np.int64)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if len(predicted) > 1 and np.any(np.diff(predicted) <= 0):
        raise ValueError("predicted_positions must be strictly increasing")
    if len(oracle) > 1 and np.any(np.diff(oracle) <= 0):
        raise ValueError("oracle_positions must be strictly increasing")

    n, m = len(predicted), len(oracle)
    match_count = np.zeros((n + 1, m + 1), dtype=np.int32)
    total_cost = np.zeros((n + 1, m + 1), dtype=np.int64)
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    # 1 = skip prediction, 2 = skip Oracle, 3 = match.
    choice[1:, 0] = 1
    choice[0, 1:] = 2

    def score(matches: int, cost: int, priority: int) -> tuple[int, int, int]:
        return matches, -cost, priority

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [
                (
                    score(int(match_count[i - 1, j]), int(total_cost[i - 1, j]), 1),
                    1,
                    int(match_count[i - 1, j]),
                    int(total_cost[i - 1, j]),
                ),
                (
                    score(int(match_count[i, j - 1]), int(total_cost[i, j - 1]), 0),
                    2,
                    int(match_count[i, j - 1]),
                    int(total_cost[i, j - 1]),
                ),
            ]
            error = abs(int(predicted[i - 1]) - int(oracle[j - 1]))
            if error <= tolerance:
                matches = int(match_count[i - 1, j - 1]) + 1
                cost = int(total_cost[i - 1, j - 1]) + error
                options.append((score(matches, cost, 2), 3, matches, cost))
            _, selected_choice, selected_matches, selected_cost = max(
                options, key=lambda item: item[0]
            )
            choice[i, j] = selected_choice
            match_count[i, j] = selected_matches
            total_cost[i, j] = selected_cost

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        selected = int(choice[i, j])
        if selected == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif selected == 1:
            i -= 1
        elif selected == 2:
            j -= 1
        else:  # only reachable at the origin
            break
    pairs.reverse()
    matched_predicted = {pair[0] for pair in pairs}
    matched_oracle = {pair[1] for pair in pairs}
    return EventMatchResult(
        matched_pairs=tuple(pairs),
        unmatched_predicted=tuple(i for i in range(n) if i not in matched_predicted),
        unmatched_oracle=tuple(j for j in range(m) if j not in matched_oracle),
        total_absolute_error=int(total_cost[n, m]),
    )

