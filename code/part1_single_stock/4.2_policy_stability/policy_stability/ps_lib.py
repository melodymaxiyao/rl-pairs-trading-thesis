"""Pure functions for policy_stability_v1 (definitions fixed in preregistration.md)."""
from __future__ import annotations

from itertools import combinations

import numpy as np

CASH, LONG = 0, 1  # action index order from config "actions": ["TargetCash", "TargetLong"]
Z_BUCKETS = ("oversold", "neutral", "overbought")


def greedy_action(q_row) -> int:
    """TargetLong only if its Q value is strictly larger; exact ties -> TargetCash."""
    return LONG if q_row[LONG] > q_row[CASH] else CASH


def modal_action(seeds_long: int, n_seeds: int = 10) -> str:
    if n_seeds != 10:
        raise ValueError(f"expected 10 seeds, got {n_seeds}")
    if seeds_long >= 6:
        return "long"
    if seeds_long <= 4:
        return "cash"
    return "tie"


def z_bucket(label: str) -> str:
    """Map a stored state label to its z bucket; refuse anything ambiguous."""
    head = label.split("/")[0]
    if head not in Z_BUCKETS:
        raise ValueError(f"cannot map state label {label!r} to a z bucket")
    return head


def pattern_label(modal_by_bucket: dict[str, str]) -> str:
    o, n, b = (modal_by_bucket[z] for z in Z_BUCKETS)
    if o == n == b == "long":
        return "always long"
    if o == "long" and b == "cash":
        return "MR exit"
    if o == "cash" and b == "long":
        return "momentum exit"
    return "other"


def direction_index(seeds_long_oversold: int, seeds_long_overbought: int, n_seeds: int = 10) -> float:
    return seeds_long_oversold / n_seeds - seeds_long_overbought / n_seeds


def stable_label(labels: list[str]) -> str | None:
    """Label shared by at least 3 of the 4 folds, else None."""
    if len(labels) != 4:
        raise ValueError("expected 4 folds")
    for lab in set(labels):
        if labels.count(lab) >= 3:
            return lab
    return None


def _agree(a: str, b: str) -> bool:
    return a == b and a != "tie"


def all_folds_agree_share(modal: list[list[str]]) -> float:
    """modal[fold][state]; share of states with the same non-tie action in every fold."""
    n_states = len(modal[0])
    hits = sum(1 for s in range(n_states)
               if modal[0][s] != "tie" and all(modal[f][s] == modal[0][s] for f in range(len(modal))))
    return hits / n_states


def mean_pairwise_agreement(modal: list[list[str]]) -> float:
    n_states = len(modal[0])
    pairs = list(combinations(range(len(modal)), 2))
    return float(np.mean([sum(_agree(modal[i][s], modal[j][s]) for s in range(n_states)) / n_states
                          for i, j in pairs]))


def fold_shares(modal_fold: list[str]) -> tuple[float, float, float]:
    n = len(modal_fold)
    return (modal_fold.count("long") / n, modal_fold.count("cash") / n, modal_fold.count("tie") / n)


def chance_all_agree(shares: list[tuple[float, float, float]]) -> float:
    return float(np.prod([s[0] for s in shares]) + np.prod([s[1] for s in shares]))


def chance_pairwise(shares: list[tuple[float, float, float]]) -> float:
    pairs = list(combinations(range(len(shares)), 2))
    return float(np.mean([shares[i][0] * shares[j][0] + shares[i][1] * shares[j][1] for i, j in pairs]))


def simulate_chance(shares, n_states: int, n_sims: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Monte Carlo version of the chance baseline: independent draws per fold and state."""
    rng = np.random.default_rng(seed)
    cats = np.array(["long", "cash", "tie"])
    all_sh, pw = [], []
    for _ in range(n_sims):
        draw = [list(rng.choice(cats, size=n_states, p=list(s))) for s in shares]
        all_sh.append(all_folds_agree_share(draw))
        pw.append(mean_pairwise_agreement(draw))
    return float(np.mean(all_sh)), float(np.mean(pw))


def exp1_fold_agreement(exp1_modal: list[str], fold_modal: list[str]) -> float:
    return sum(_agree(a, b) for a, b in zip(exp1_modal, fold_modal)) / len(exp1_modal)
