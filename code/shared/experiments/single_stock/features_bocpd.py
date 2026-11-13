"""Causal BOCPD features for z20.

The recursive run-length posterior is delegated to the pinned
``sdt.changepoint.BayesOnline`` implementation.  At decision row ``t`` we use
only the posterior produced after feeding observations through ``t``.  A
configurable confirmation lag ``d`` turns ``P(r_t=d)`` into the probability
that a changepoint occurred at ``t-d``; labels are emitted at ``t`` and are
never back-filled onto their source row.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fyp_bocpd_matplotlib")
)
from sdt.changepoint import BayesOnline


NO_CONFIRMED_TURN = 0
CONFIRMED_PEAK = 1
CONFIRMED_TROUGH = 2
TURN_STATE_LABELS = {
    NO_CONFIRMED_TURN: "no_confirmed_turn",
    CONFIRMED_PEAK: "at_confirmed_peak",
    CONFIRMED_TROUGH: "at_confirmed_trough",
}


@dataclass(frozen=True)
class BOCPDOnlineResult:
    """Tabular features plus the complete posterior available at every row."""

    frame: pd.DataFrame
    run_length_posteriors: tuple[np.ndarray | None, ...]


def installed_dependency_version() -> str:
    return metadata.version("sdt-python")


def _validate_inputs(
    z20: pd.Series,
    *,
    hazard_lambda: float,
    probability_threshold: float,
    confirmation_lag_days: int,
    direction_pre_window_days: int,
    observation_model: dict[str, Any],
    expected_dependency_version: str,
) -> pd.Series:
    if installed_dependency_version() != expected_dependency_version:
        raise RuntimeError(
            "BOCPD dependency mismatch: expected sdt-python=="
            f"{expected_dependency_version}, found {installed_dependency_version()}"
        )
    if not z20.index.is_monotonic_increasing or z20.index.has_duplicates:
        raise ValueError("z20 index must be unique and increasing")
    if not np.isfinite(hazard_lambda) or hazard_lambda <= 1.0:
        raise ValueError("hazard_lambda must be finite and greater than 1")
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be in (0, 1)")
    if confirmation_lag_days < 1:
        raise ValueError("confirmation_lag_days must be at least 1")
    if direction_pre_window_days < 1:
        raise ValueError("direction_pre_window_days must be at least 1")
    required_prior = {"name", "mu", "kappa", "alpha", "beta"}
    if not required_prior.issubset(observation_model):
        raise ValueError(
            f"observation_model is missing {sorted(required_prior - set(observation_model))}"
        )
    if observation_model["name"] != "student_t":
        raise ValueError("sdt BayesOnline must use the preregistered student_t model")
    for name in ("mu", "kappa", "alpha", "beta"):
        if not np.isfinite(float(observation_model[name])):
            raise ValueError(f"observation_model.{name} must be finite")
    if min(
        float(observation_model["kappa"]),
        float(observation_model["alpha"]),
        float(observation_model["beta"]),
    ) <= 0.0:
        raise ValueError("kappa, alpha and beta must be positive")
    values = pd.to_numeric(z20, errors="coerce")
    finite = values.notna().to_numpy()
    if finite.any():
        first = int(np.flatnonzero(finite)[0])
        if not finite[first:].all():
            raise ValueError("z20 may have leading NaNs but no internal missing values")
    return values


def run_bocpd_online(
    z20: pd.Series,
    *,
    hazard_lambda: float,
    probability_threshold: float,
    confirmation_lag_days: int,
    direction_pre_window_days: int,
    observation_model: dict[str, Any],
    dependency_engine: str,
    expected_dependency_version: str,
) -> BOCPDOnlineResult:
    """Sequentially update BOCPD and return causal turn-state features."""
    values = _validate_inputs(
        z20,
        hazard_lambda=hazard_lambda,
        probability_threshold=probability_threshold,
        confirmation_lag_days=confirmation_lag_days,
        direction_pre_window_days=direction_pre_window_days,
        observation_model=observation_model,
        expected_dependency_version=expected_dependency_version,
    )
    if dependency_engine not in {"python", "numba"}:
        raise ValueError("dependency_engine must be 'python' or 'numba'")

    detector = BayesOnline(
        hazard="const",
        obs_likelihood="student_t",
        hazard_params={"time_scale": float(hazard_lambda)},
        obs_params={
            "mu": float(observation_model["mu"]),
            "kappa": float(observation_model["kappa"]),
            "alpha": float(observation_model["alpha"]),
            "beta": float(observation_model["beta"]),
        },
        engine=dependency_engine,
    )

    n_rows = len(values)
    states = np.full(n_rows, NO_CONFIRMED_TURN, dtype=np.int8)
    probabilities = np.full(n_rows, np.nan, dtype=np.float64)
    posterior_modes = np.full(n_rows, np.nan, dtype=np.float64)
    posterior_means = np.full(n_rows, np.nan, dtype=np.float64)
    source_locs = np.full(n_rows, -1, dtype=np.int64)
    pre_means = np.full(n_rows, np.nan, dtype=np.float64)
    post_means = np.full(n_rows, np.nan, dtype=np.float64)
    posteriors: list[np.ndarray | None] = [None] * n_rows
    numeric = values.to_numpy(dtype=np.float64)

    for decision_loc, observation in enumerate(numeric):
        if not np.isfinite(observation):
            continue
        detector.update(float(observation))
        posterior = np.asarray(detector.probabilities[-1], dtype=np.float64).copy()
        if not np.isclose(posterior.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError("Third-party BOCPD returned a non-normalized posterior")
        posteriors[decision_loc] = posterior
        posterior_modes[decision_loc] = float(np.argmax(posterior))
        posterior_means[decision_loc] = float(
            np.dot(np.arange(len(posterior), dtype=np.float64), posterior)
        )

        source_loc = decision_loc - confirmation_lag_days
        if (
            source_loc < direction_pre_window_days
            or len(posterior) <= confirmation_lag_days
        ):
            continue
        probability = float(posterior[confirmation_lag_days])
        probabilities[decision_loc] = probability
        pre = numeric[source_loc - direction_pre_window_days : source_loc]
        post = numeric[source_loc : decision_loc + 1]
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            continue
        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        pre_means[decision_loc] = pre_mean
        post_means[decision_loc] = post_mean
        if probability <= probability_threshold or post_mean == pre_mean:
            continue
        states[decision_loc] = (
            CONFIRMED_TROUGH if post_mean > pre_mean else CONFIRMED_PEAK
        )
        source_locs[decision_loc] = source_loc

    source_dates = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    valid_sources = source_locs >= 0
    if valid_sources.any():
        source_dates.iloc[np.flatnonzero(valid_sources)] = values.index[
            source_locs[valid_sources]
        ].to_numpy()
    frame = pd.DataFrame(
        {
            "bocpd_changepoint_probability": probabilities,
            "bocpd_run_length_mode": posterior_modes,
            "bocpd_run_length_mean": posterior_means,
            "bocpd_turn_source_date": source_dates,
            "bocpd_direction_pre_mean": pre_means,
            "bocpd_direction_post_mean": post_means,
            "confirmed_turn_state": states,
            "confirmed_turn_label": pd.Series(states, index=values.index).map(
                TURN_STATE_LABELS
            ),
        },
        index=values.index,
    )
    return BOCPDOnlineResult(frame=frame, run_length_posteriors=tuple(posteriors))


def build_bocpd_features(
    z20: pd.Series,
    *,
    hazard_lambda: float,
    probability_threshold: float,
    confirmation_lag_days: int,
    direction_pre_window_days: int,
    observation_model: dict[str, Any],
    dependency_engine: str,
    expected_dependency_version: str,
) -> pd.DataFrame:
    """Convenience wrapper returning only the feature frame."""
    return run_bocpd_online(
        z20,
        hazard_lambda=hazard_lambda,
        probability_threshold=probability_threshold,
        confirmation_lag_days=confirmation_lag_days,
        direction_pre_window_days=direction_pre_window_days,
        observation_model=observation_model,
        dependency_engine=dependency_engine,
        expected_dependency_version=expected_dependency_version,
    ).frame
