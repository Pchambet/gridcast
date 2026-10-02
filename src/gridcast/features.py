"""Feature matrix for day-ahead demand forecasting.

Every feature for target day D is a function of (a) the calendar, (b) demand known at
the issue time (see :mod:`gridcast.infoset`) and (c) temperature. Two weather modes:

* ``"observed"`` uses reanalysis temperature everywhere. It is used to *train* (the
  past is known exactly when a model is refit) and for an oracle ablation.
* ``"forecast"`` uses only what a forecaster had at issue time: observed temperature
  up to the issue time, then archived day-ahead forecasts. It is used to *predict*.

Demand lags are taken on the local wall clock (same local hour two and seven days
earlier) so that DST changes never shift the daily profile by one hour.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Literal

import holidays
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from gridcast import infoset
from gridcast.config import PARIS

WeatherMode = Literal["observed", "forecast"]
EWM_HALFLIVES_H = (12, 48)
MOS_WINDOW_DAYS = 60

FEATURES = [
    "hour",
    "dow",
    "doy_sin",
    "doy_cos",
    "utc_offset",
    "is_holiday",
    "is_bridge",
    "xmas",
    "offday_d1",
    "offday_d2",
    "offday_d7",
    "load_d2",
    "load_d7",
    "load_d1_morning",
    "load_d2_morning",
    "load_d1_last",
    "load_d7_mean",
    "week_ratio",
    "anchor_scaled",
    "temp",
    "temp_daymean",
    "temp_daymin",
    "temp_ewm12",
    "temp_ewm48",
    "temp_d7_daymean",
    "temp_delta7",
]


# --- Calendar -------------------------------------------------------------------------


@cache
def _french_holidays(first: int, last: int) -> frozenset[pd.Timestamp]:
    return frozenset(pd.Timestamp(d) for d in holidays.France(years=range(first, last + 1)))


def day_calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per local day: holiday, bridge day ("pont") and off-day flags."""
    days = pd.DatetimeIndex(days).normalize()
    hol = _french_holidays(days.year.min() - 1, days.year.max() + 1)
    is_hol = np.array([d in hol for d in days])
    dow = days.dayofweek.to_numpy()
    prev_hol = np.array([(d - pd.Timedelta(days=1)) in hol for d in days])
    next_hol = np.array([(d + pd.Timedelta(days=1)) in hol for d in days])
    # A Monday before a Tuesday holiday, or a Friday after a Thursday holiday.
    bridge = ((dow == 0) & next_hol) | ((dow == 4) & prev_hol)
    xmas = ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day <= 1))
    offday = is_hol | bridge | (dow >= 5)
    return pd.DataFrame(
        {"is_holiday": is_hol, "is_bridge": bridge, "xmas": np.asarray(xmas), "offday": offday},
        index=days,
    ).astype("int8")


# --- Demand on the local wall clock ---------------------------------------------------


def to_local_wallclock(series: pd.Series) -> pd.Series:
    """Re-index a UTC hourly series on naive local time.

    The repeated hour of the autumn DST change is averaged; the non-existent hour of the
    spring change is the mean of its two neighbours, so every local day has 24 slots.
    No other gap is filled: interpolating a missing value from a later one could leak
    information past the issue time.
    """
    local = series.copy()
    local.index = series.index.tz_convert(PARIS).tz_localize(None)
    local = local.groupby(level=0).mean()
    full = pd.date_range(local.index.min(), local.index.max(), freq="h")
    local = local.reindex(full)
    nonexistent = full.tz_localize(PARIS, nonexistent="NaT", ambiguous=True).isna()
    neighbours = (local.shift(1) + local.shift(-1)) / 2
    return local.where(~nonexistent, neighbours)


def _daily_stat(local: pd.Series, hours: slice | None, how: str) -> pd.Series:
    sub = (
        local
        if hours is None
        else local[(local.index.hour >= hours.start) & (local.index.hour <= hours.stop)]
    )
    grouped = sub.groupby(sub.index.normalize())
    stat = grouped.agg(how)
    # Days with missing hours would give a biased mean: require complete coverage.
    expected = 24 if hours is None else hours.stop - hours.start + 1
    return stat.where(grouped.count() == expected)


# --- Temperature ----------------------------------------------------------------------


def _alpha(halflife_h: float) -> float:
    return 1.0 - 0.5 ** (1.0 / halflife_h)


def ewm(values: np.ndarray, halflife_h: float, init: float | None = None) -> np.ndarray:
    """Exponentially weighted mean y_t = a x_t + (1 - a) y_{t-1} (thermal inertia proxy)."""
    a = _alpha(halflife_h)
    x = np.asarray(values, dtype=float)
    y0 = x[0] if init is None else init
    out, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=np.array([(1.0 - a) * y0]))
    return out


def mos_correct(table: pd.DataFrame, window_days: int = MOS_WINDOW_DAYS) -> pd.DataFrame:
    """Remove the recent systematic error of each forecast lead, per UTC hour of day.

    Raw GFS 2 m temperature runs ~1.3 degC warmer than ERA5 at night and is unbiased in
    the afternoon. A model trained on ERA5 would read that as a warm bias in demand
    terms, so forecasts are corrected by their mean error over a trailing window
    (model output statistics). Errors are only used once the valid time is at least
    two days old, which is always before the issue time of any day they correct.
    """
    out = pd.DataFrame(index=table.index)
    hour = table.index.hour
    for lead in ("d1", "d2"):
        col = f"temp_fc_{lead}"
        err = table[col] - table["temp_obs"]
        bias = err.groupby(hour).transform(
            lambda s: s.rolling(window_days, min_periods=window_days // 3).mean().shift(2)
        )
        out[col] = table[col] - bias.fillna(0.0)
    return out


def forecast_temperature(table: pd.DataFrame, day: pd.Timestamp, times: pd.DatetimeIndex):
    """Lead-matched forecast for ``times`` (falls back to the other lead if missing).

    ``table`` must hold the (MOS-corrected) ``temp_fc_d1`` / ``temp_fc_d2`` columns.
    """
    d1 = table["temp_fc_d1"].reindex(times).to_numpy()
    d2 = table["temp_fc_d2"].reindex(times).to_numpy()
    use_d2 = infoset.use_day2_forecast(times, day)
    primary = np.where(use_d2, d2, d1)
    backup = np.where(use_d2, d1, d2)
    return np.where(np.isnan(primary), backup, primary)


def observed_or_proxy(table: pd.DataFrame) -> pd.Series:
    """Observed temperature, with gaps filled by the bias-corrected 24 h-ahead forecast.

    Reanalysis arrives ~5 days late, so in live operation the most recent days are only
    known through forecasts issued a day before them. In the backtest window the
    reanalysis is complete and this is exactly ``temp_obs``.
    """
    return table["temp_obs"].fillna(mos_correct(table)["temp_fc_d1"])


def _observed_ewm(table: pd.DataFrame) -> pd.DataFrame:
    obs = observed_or_proxy(table)
    valid = obs.notna().to_numpy()
    out = {}
    for h in EWM_HALFLIVES_H:
        vals = np.full(len(obs), np.nan)
        if valid.any():
            first = int(np.argmax(valid))
            last = len(valid) - int(np.argmax(valid[::-1]))
            # A forward fill is causal, so an isolated gap cannot leak future values; it
            # only stops one missing hour from turning the whole recursion into NaN.
            vals[first:last] = ewm(obs.iloc[first:last].ffill().to_numpy(), h)
        out[f"temp_ewm{h}"] = vals
    return pd.DataFrame(out, index=table.index)


def _temperature_block(
    table: pd.DataFrame, days: Sequence[pd.Timestamp], targets: pd.DatetimeIndex, mode: WeatherMode
) -> pd.DataFrame:
    obs_ewm = _observed_ewm(table)
    if mode == "observed":
        block = pd.DataFrame({"temp": table["temp_obs"].reindex(targets).to_numpy()}, index=targets)
        for h in EWM_HALFLIVES_H:
            block[f"temp_ewm{h}"] = obs_ewm[f"temp_ewm{h}"].reindex(targets).to_numpy()
        return block

    corrected = mos_correct(table)
    parts = []
    for day in days:
        issue = infoset.issue_time(day)
        hours = infoset.target_hours(day)
        # Hours from the issue time to the end of D: everything here is a forecast.
        path = pd.date_range(issue, hours[-1], freq="h")
        fc = forecast_temperature(corrected, day, path)
        block = pd.DataFrame({"temp": fc[-len(hours) :]}, index=hours)
        before = issue - pd.Timedelta(hours=1)
        for h in EWM_HALFLIVES_H:
            init = obs_ewm[f"temp_ewm{h}"].get(before, np.nan)
            if np.isnan(fc).any() or np.isnan(init):
                block[f"temp_ewm{h}"] = np.nan
                continue
            # Start from the observed state at the issue time, continue on forecasts.
            path_ewm = ewm(fc, h, init=init)
            block[f"temp_ewm{h}"] = path_ewm[-len(hours) :]
        parts.append(block)
    return pd.concat(parts).reindex(targets)


# --- Assembly -------------------------------------------------------------------------


def build_features(
    table: pd.DataFrame, days: Sequence, mode: WeatherMode = "forecast"
) -> pd.DataFrame:
    """Feature rows for every hour of every local day in ``days``.

    ``table`` is the hourly UTC table from :func:`gridcast.data.build_hourly`. The
    result is indexed by UTC hour-start and also carries the target ``load`` and RTE's
    day-ahead forecast ``rte_j1`` (never used as a feature) for evaluation.
    """
    days = [pd.Timestamp(d).normalize() for d in days]
    targets = infoset.target_hours_many(days)
    naive = targets.tz_convert(PARIS).tz_localize(None)
    local_day = naive.normalize()

    load = to_local_wallclock(table["load"])
    last = infoset.LAST_KNOWN_LOCAL_HOUR
    morning = _daily_stat(load, slice(0, last), "mean")
    daymean = _daily_stat(load, None, "mean")

    one, two, seven = (pd.Timedelta(days=k) for k in (1, 2, 7))
    cal_days = pd.date_range(local_day.min() - 2 * seven, local_day.max(), freq="D")
    cal = day_calendar(cal_days)

    doy = naive.dayofyear.to_numpy()
    frame = pd.DataFrame(index=targets)
    frame["day"] = local_day
    frame["hour"] = naive.hour
    frame["dow"] = naive.dayofweek
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    offsets = targets.tz_convert(PARIS).map(lambda t: t.utcoffset().total_seconds() / 3600)
    frame["utc_offset"] = np.asarray(offsets, dtype=float)
    for col in ("is_holiday", "is_bridge", "xmas"):
        frame[col] = cal[col].reindex(local_day).to_numpy()
    for k, delta in (("d1", one), ("d2", two), ("d7", seven)):
        frame[f"offday_{k}"] = cal["offday"].reindex(local_day - delta).to_numpy()

    frame["load_d2"] = load.reindex(naive - two).to_numpy()
    frame["load_d7"] = load.reindex(naive - seven).to_numpy()
    frame["load_d1_morning"] = morning.reindex(local_day - one).to_numpy()
    frame["load_d2_morning"] = morning.reindex(local_day - two).to_numpy()
    frame["load_d1_last"] = load.reindex(local_day - one + pd.Timedelta(hours=last)).to_numpy()
    frame["load_d7_mean"] = daymean.reindex(local_day - seven).to_numpy()
    # Similar-day anchor: last week's same hour, or two weeks back when only that one
    # matches the target's work/off-day status (holidays break weekly persistence).
    off = cal["offday"]
    off_target = off.reindex(local_day).to_numpy()
    load_d14 = load.reindex(naive - 2 * seven).to_numpy()
    use_d14 = (frame["offday_d7"].to_numpy() != off_target) & (
        off.reindex(local_day - 2 * seven).to_numpy() == off_target
    )
    anchor = np.where(use_d14, load_d14, frame["load_d7"].to_numpy())
    # Week-over-week drift of the level, measured on the latest known mornings (D-1 vs
    # D-8, same weekday): lets the trees rescale the anchor without having to
    # extrapolate absolute levels they have never seen (e.g. the 2022 demand drop).
    # Undefined when one of the two mornings is an off-day and the other is not.
    d8_morning = morning.reindex(local_day - pd.Timedelta(days=8)).to_numpy()
    comparable = (
        frame["offday_d1"].to_numpy() == off.reindex(local_day - pd.Timedelta(days=8)).to_numpy()
    )
    ratio = frame["load_d1_morning"].to_numpy() / d8_morning
    frame["week_ratio"] = np.where(comparable, ratio, 1.0)
    frame["anchor_scaled"] = anchor * frame["week_ratio"].to_numpy()

    temps = _temperature_block(table, days, targets, mode)
    frame["temp"] = temps["temp"].to_numpy()
    by_day = frame.groupby("day")["temp"]
    frame["temp_daymean"] = by_day.transform("mean")
    frame["temp_daymin"] = by_day.transform("min")
    for h in EWM_HALFLIVES_H:
        frame[f"temp_ewm{h}"] = temps[f"temp_ewm{h}"].to_numpy()
    obs_daily = _daily_stat(to_local_wallclock(observed_or_proxy(table)), None, "mean")
    frame["temp_d7_daymean"] = obs_daily.reindex(local_day - seven).to_numpy()
    frame["temp_delta7"] = frame["temp_daymean"] - frame["temp_d7_daymean"]

    frame["load"] = table["load"].reindex(targets).to_numpy()
    frame["rte_j1"] = table["rte_j1"].reindex(targets).to_numpy()
    return frame
