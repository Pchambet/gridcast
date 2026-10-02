"""Conformal prediction intervals for a stream of day-ahead forecasts.

Three ingredients, combined into the methods compared in the report:

* **Score**: absolute residual |y - yhat| (split conformal, symmetric intervals) or the
  CQR score max(lo - y, y - hi) of Romano et al. (2019), which inherits the
  heteroscedasticity of the quantile model.
* **Calibration set**: a frozen burn-in year ("static") or the trailing
  ``window_days`` of scored days ("rolling").
* **ACI** (Gibbs & Candes, 2021): the miscoverage level used to read the calibration
  quantile is updated online, alpha_{t+1} = alpha_t + gamma (alpha - err_t), which
  guarantees long-run coverage under arbitrary distribution shift.

Feedback is delayed: errors of day D are only used from day D + ``delay_days`` on,
because the forecast for D + 1 is issued before D is over.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.models import qcol


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.

    Returns +inf when the calibration set is too small for the requested level, which
    is the exact (infinite-width) conformal answer.
    """
    n = len(scores)
    k = ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return np.inf
    if k < 1:
        return -np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def abs_scores(y: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.abs(y - point)


def cqr_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def aci_step(alpha_t: float, target: float, gamma: float, errors: np.ndarray) -> float:
    """One delayed ACI update from the miscoverage indicators of one day."""
    return alpha_t + gamma * (target - float(np.mean(errors)))


@dataclass(frozen=True)
class Method:
    name: str
    score: str  # "abs" | "cqr" | "raw"
    calibration: str  # "static" | "rolling" | "none"
    aci: bool = False
    label: str = ""


METHODS: tuple[Method, ...] = (
    Method("qr_raw", "raw", "none", label="Quantile LightGBM, uncalibrated"),
    Method("split_static", "abs", "static", label="Split conformal, static"),
    Method("split_static_aci", "abs", "static", aci=True, label="Split conformal + ACI"),
    Method("split_rolling", "abs", "rolling", label="Split conformal, rolling 90 d"),
    Method("cqr_rolling", "cqr", "rolling", label="CQR, rolling 90 d"),
    Method("cqr_rolling_aci", "cqr", "rolling", aci=True, label="CQR rolling + ACI"),
)
METHOD_BY_NAME = {m.name: m for m in METHODS}


def _bands(pred: pd.DataFrame, level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1 - level
    return pred[qcol(alpha / 2)].to_numpy(), pred[qcol(1 - alpha / 2)].to_numpy()


def online_intervals(
    pred: pd.DataFrame,
    method: Method,
    level: float,
    *,
    gamma: float = config.ACI_GAMMA,
    window_days: int = config.ROLLING_WINDOW_DAYS,
    delay_days: int = config.FEEDBACK_DELAY_DAYS,
    static_end: pd.Timestamp = config.CALIBRATION_END,
    min_days: int = 30,
) -> pd.DataFrame:
    """Intervals for every row of ``pred``, produced day by day as they would be live.

    ``pred`` needs columns ``day``, ``load``, ``point`` and the quantile columns. Rows
    are hours; all hours of a day share one calibration quantile. Days without enough
    calibration history get NaN bounds.
    """
    pred = pred.sort_index()
    y = pred["load"].to_numpy()
    point = pred["point"].to_numpy()
    lo_q, hi_q = _bands(pred, level)
    if method.score == "raw":
        return pd.DataFrame({"lo": lo_q, "hi": hi_q, "alpha_t": np.nan}, index=pred.index)

    base_lo, base_hi = (point, point) if method.score == "abs" else (lo_q, hi_q)
    scores = abs_scores(y, point) if method.score == "abs" else cqr_scores(y, lo_q, hi_q)

    day_values = pred["day"].to_numpy()
    days, starts = np.unique(day_values, return_index=True)
    ends = np.append(starts[1:], len(pred))
    day_index = pd.DatetimeIndex(days)
    delay = pd.Timedelta(days=delay_days)
    window = pd.Timedelta(days=window_days)

    target = 1 - level
    alpha_t = target
    lo = np.full(len(pred), np.nan)
    hi = np.full(len(pred), np.nan)
    alphas = np.full(len(pred), np.nan)
    produced: list[int] = []  # days with an interval whose error is not yet fed back
    for i, day in enumerate(day_index):
        newest = day - delay
        if method.aci:
            while produced and day_index[produced[0]] <= newest:
                j = produced.pop(0)
                sl = slice(starts[j], ends[j])
                miss = (y[sl] < lo[sl]) | (y[sl] > hi[sl])
                ok = ~np.isnan(y[sl])
                if ok.any():
                    alpha_t = aci_step(alpha_t, target, gamma, miss[ok])

        if method.calibration == "static":
            in_cal = day_index <= min(newest, static_end)
        else:
            in_cal = (day_index <= newest) & (day_index > newest - window)
        if in_cal.sum() < min_days:
            continue
        cal = np.concatenate([scores[starts[j] : ends[j]] for j in np.flatnonzero(in_cal)])
        cal = cal[~np.isnan(cal)]
        # ACI may push alpha_t outside (0, 1); clip to the widest/narrowest finite
        # interval the calibration set supports instead of returning +-inf.
        eff = float(np.clip(alpha_t, 1.0 / (len(cal) + 1), 1.0)) if method.aci else target
        q = conformal_quantile(cal, eff)
        if q == np.inf:
            q = float(np.max(cal))
        elif q == -np.inf:
            q = float(np.min(cal))
        sl = slice(starts[i], ends[i])
        lo[sl] = base_lo[sl] - q
        hi[sl] = base_hi[sl] + q
        alphas[sl] = alpha_t
        produced.append(i)
    return pd.DataFrame({"lo": lo, "hi": hi, "alpha_t": alphas}, index=pred.index)


def all_intervals(pred: pd.DataFrame, gamma: float = config.ACI_GAMMA) -> pd.DataFrame:
    """Long table: one row per (hour, method, level) with lo / hi bounds."""
    parts = []
    for method in METHODS:
        for level in config.LEVELS:
            iv = online_intervals(pred, method, level, gamma=gamma)
            iv["method"] = method.name
            iv["level"] = level
            parts.append(iv.reset_index())
    return pd.concat(parts, ignore_index=True)
