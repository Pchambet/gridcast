"""Forecast scores and their sampling uncertainty.

Hourly errors are strongly autocorrelated (a mis-forecast cold spell lasts days), so
naive i.i.d. standard errors would be far too narrow. Confidence intervals resample
whole calendar weeks (a block bootstrap), which keeps the within-week dependence.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def ape(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Absolute percentage error per observation (as a fraction)."""
    return np.abs(np.asarray(yhat) - np.asarray(y)) / np.abs(np.asarray(y))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(ape(y, yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))


def hits(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where the outcome falls inside the closed interval [lo, hi].

    A missing bound counts as a miss, so callers must drop rows without an interval
    before scoring rather than let them bias coverage down.
    """
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return (y >= lo) & (y <= hi)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hits(y, lo, hi)))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    """Winkler / interval score (Gneiting & Raftery, 2007), per observation.

    Width plus 2/alpha times the distance by which y falls outside the interval:
    a proper scoring rule, so it rewards narrow intervals only if they also cover.
    """
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return (hi - lo) + (2.0 / alpha) * (below + above)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Quantile (pinball) loss of a tau-quantile forecast, per observation."""
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def week_blocks(times: pd.DatetimeIndex) -> np.ndarray:
    """Integer block id per row: the ISO week (local calendar) the row belongs to."""
    days = pd.DatetimeIndex(times).tz_convert("Europe/Paris").tz_localize(None).normalize()
    monday = days - pd.to_timedelta(days.dayofweek, unit="D")
    return pd.factorize(monday)[0]


def block_bootstrap(
    stat: Callable[[np.ndarray], float],
    blocks: np.ndarray,
    reps: int = 1000,
    seed: int = 7,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat(row_indices)`` under block resampling.

    ``stat`` receives an integer index array into the evaluated rows, so any metric (or
    difference of metrics between two models on the same rows) can be bootstrapped.
    """
    if len(blocks) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    order = np.argsort(blocks, kind="stable")
    ids, starts = np.unique(blocks[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    members = [order[s:e] for s, e in zip(starts, ends, strict=True)]
    estimate = stat(np.arange(len(blocks)))
    draws = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(ids), size=len(ids))
        draws[r] = stat(np.concatenate([members[i] for i in pick]))
    tail = (1 - ci) / 2
    lo, hi = np.quantile(draws, [tail, 1 - tail])
    return float(estimate), float(lo), float(hi)
