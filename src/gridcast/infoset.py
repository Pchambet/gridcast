"""The information set of a day-ahead forecast, written down once and tested.

A forecast for local day D is issued at 12:00 Europe/Paris on D-1. Everything else in
the project (feature building, conformal feedback, the live job) derives its notion of
"what was known when" from the functions below, so leakage can only enter here.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from gridcast.config import ISSUE_HOUR_LOCAL, PARIS, PUBLICATION_LAG, RUN_AVAILABILITY_DELAY

# Last local hour h such that the demand of [h, h+1) is published by the issue time.
LAST_KNOWN_LOCAL_HOUR = ISSUE_HOUR_LOCAL - 1 - int(PUBLICATION_LAG / pd.Timedelta(hours=1))


def _as_date(day: date | pd.Timestamp) -> date:
    return day.date() if isinstance(day, pd.Timestamp) else day


def local_midnight(day: date | pd.Timestamp) -> pd.Timestamp:
    """Start of local day ``day`` as a UTC timestamp (handles DST offsets)."""
    return pd.Timestamp(_as_date(day)).tz_localize(PARIS).tz_convert("UTC")


def issue_time(day: date | pd.Timestamp) -> pd.Timestamp:
    """When the forecast for local day ``day`` is issued, in UTC."""
    d = pd.Timestamp(_as_date(day)) - pd.Timedelta(days=1)
    return (d + pd.Timedelta(hours=ISSUE_HOUR_LOCAL)).tz_localize(PARIS).tz_convert("UTC")


def demand_cutoff(day: date | pd.Timestamp) -> pd.Timestamp:
    """Hourly demand values whose hour *starts* strictly before this are known.

    The hour [h, h+1) is published at h + 1 + lag; it is known at issue time iff
    h + 1 + lag <= issue, i.e. h < issue - lag.
    """
    return issue_time(day) - PUBLICATION_LAG


def target_hours(day: date | pd.Timestamp) -> pd.DatetimeIndex:
    """The 23, 24 or 25 UTC hour-starts that make up local day ``day``."""
    start = local_midnight(day)
    end = local_midnight(pd.Timestamp(_as_date(day)) + pd.Timedelta(days=1))
    return pd.date_range(start, end, freq="h", inclusive="left", name="time")


def target_hours_many(days: Iterable[date | pd.Timestamp]) -> pd.DatetimeIndex:
    """Union of :func:`target_hours` over ``days``, sorted."""
    wanted = pd.DatetimeIndex([pd.Timestamp(_as_date(d)) for d in days]).unique()
    if len(wanted) == 0:
        return pd.DatetimeIndex([], tz="UTC", name="time")
    hours = pd.date_range(
        local_midnight(wanted.min()),
        local_midnight(wanted.max() + pd.Timedelta(days=1)),
        freq="h",
        inclusive="left",
        name="time",
    )
    return hours[local_day_of(hours).isin(wanted)]


def use_day2_forecast(valid: pd.DatetimeIndex, day: date | pd.Timestamp) -> np.ndarray:
    """True where the 24 h-ahead archived forecast was not yet available at issue time.

    The value forecast 24 h before ``valid`` comes from a run started at most
    ``valid - 24 h`` and finished ``RUN_AVAILABILITY_DELAY`` later. Hours for which that
    run could finish after the issue time fall back to the 48 h-ahead value. In
    practice the first ~8 local hours of D use d1 and the rest d2: slightly pessimistic
    about weather skill, never optimistic.
    """
    latest_d1_valid = issue_time(day) - RUN_AVAILABILITY_DELAY + pd.Timedelta(hours=24)
    return np.asarray(valid > latest_d1_valid)


def local_day_of(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar day (naive midnight) of each UTC timestamp."""
    return times.tz_convert(PARIS).tz_localize(None).normalize()
