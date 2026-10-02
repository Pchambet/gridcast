"""Download and assemble the hourly modelling table.

Sources (all open, no API key):

* RTE eco2mix national demand via ODRE (Licence Ouverte / Etalab 2.0): consolidated
  history and the real-time feed, including RTE's own day-ahead forecast (J-1).
* Open-Meteo: ERA5 reanalysis for observed temperature, and the previous-runs archive
  for temperature *forecasts* as they were issued 24 h / 48 h ahead (CC BY 4.0).

Downloads are cached in ``data/raw``; ``refresh=True`` re-fetches only the parts that
grow over time (real-time demand, recent weather).
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gridcast import config
from gridcast.config import CITIES, RAW

_SESSION = requests.Session()


def _get(url: str, params: dict | None = None, timeout: int = 300) -> requests.Response:
    """GET with retries; Open-Meteo answers 429 when the per-minute quota is spent.

    Every retry is logged, so a stalled scheduled job shows why it is waiting, and the
    last error is chained to the final exception.
    """
    last: Exception | str = "no attempt"
    for attempt in range(8):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last, wait = exc, 10 * (attempt + 1)
        else:
            if resp.status_code < 500 and resp.status_code != 429:
                resp.raise_for_status()
                return resp
            last = f"HTTP {resp.status_code}"
            wait = 65 if resp.status_code == 429 else 10 * (attempt + 1)
        print(f"  retry {attempt + 1}/8 in {wait} s: {last} ({url})", flush=True)
        time.sleep(wait)
    cause = last if isinstance(last, Exception) else None
    raise RuntimeError(f"giving up on {url} after repeated failures: {last}") from cause


# --- Demand ---------------------------------------------------------------------------


def download_eco2mix(dataset: str, path: Path) -> None:
    params = {"select": "date_heure,consommation,prevision_j1,prevision_j"}
    resp = _get(config.ODRE_EXPORT.format(dataset=dataset), params=params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)


def fetch_demand(refresh: bool = False) -> None:
    """Consolidated history is immutable: fetched once. The real-time feed is refreshed."""
    def_path = RAW / "eco2mix_def.parquet"
    tr_path = RAW / "eco2mix_tr.parquet"
    if not def_path.exists():
        print("downloading eco2mix consolidated history (~7 MB)...")
        download_eco2mix(config.ECO2MIX_DEF, def_path)
    if refresh or not tr_path.exists():
        print("downloading eco2mix real-time feed...")
        download_eco2mix(config.ECO2MIX_TR, tr_path)


def to_hourly_demand(raw: pd.DataFrame) -> pd.DataFrame:
    """Average 15/30-min eco2mix values into hourly MW (the mean of what is present).

    Consumption is reported every 30 min in the consolidated file and every 15 min in
    the real-time feed; forecasts every 15 min. The hourly mean of available points is
    the hourly energy in MWh, so both feeds land on the same scale.
    """
    df = raw.rename(
        columns={"consommation": "load", "prevision_j1": "rte_j1", "prevision_j": "rte_j"}
    )
    df["time"] = pd.to_datetime(df["date_heure"], utc=True).dt.floor("h")
    cols = ["load", "rte_j1", "rte_j"]
    hourly = df.groupby("time")[cols].mean()
    return hourly.astype("float64")


def load_demand() -> pd.DataFrame:
    """Hourly demand, consolidated values taking precedence over real-time ones.

    ``realtime`` flags hours that come from the real-time feed: RTE's J-1 forecast is
    scored differently against the two series (see the README), so the distinction is kept.
    """
    parts = []
    for name, realtime in (("eco2mix_def.parquet", 0.0), ("eco2mix_tr.parquet", 1.0)):
        path = RAW / name
        if path.exists():
            hourly = to_hourly_demand(pd.read_parquet(path))
            hourly["realtime"] = realtime
            parts.append(hourly)
    if not parts:
        raise FileNotFoundError("no eco2mix data in data/raw; run `gridcast data` first")
    hourly = parts[0]
    for extra in parts[1:]:
        hourly = hourly.combine_first(extra)
    return hourly.sort_index()


# --- Weather --------------------------------------------------------------------------


def _hourly_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize("UTC")
    return frame.set_index("time")


def fetch_observed(city: config.City, start: date, end: date) -> pd.Series:
    """ERA5 2 m temperature (reanalysis: what actually happened, a few days late)."""
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m",
        "models": "era5",
        "timezone": "UTC",
    }
    frame = _hourly_frame(_get(config.OPEN_METEO_ARCHIVE, params).json())
    return frame["temperature_2m"].rename(city.name)


def fetch_forecasts(city: config.City, start: date, end: date, model: str) -> pd.DataFrame:
    """Archived forecasts valid at each hour, issued 24 h (d1) and 48 h (d2) earlier."""
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
        "models": model,
        "timezone": "UTC",
    }
    frame = _hourly_frame(_get(config.OPEN_METEO_PREVIOUS_RUNS, params).json())
    frame.columns = ["fc_d1", "fc_d2"]
    return frame


def _merge_cached(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        new = new.combine_first(old)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_index().to_parquet(path)
    return new


def fetch_weather(refresh: bool = False, today: date | None = None) -> None:
    """Observed and forecast temperature for every city, cached and incremental."""
    today = today or pd.Timestamp.now(tz="UTC").date()
    obs_path = RAW / "weather_observed.parquet"
    fc_path = RAW / "weather_forecast.parquet"

    if refresh or not obs_path.exists():
        start = config.HISTORY_START.date()
        if obs_path.exists():
            last = pd.read_parquet(obs_path).dropna(how="all").index.max()
            start = (last - pd.Timedelta(days=10)).date()
        cols = []
        for city in CITIES:
            print(f"observed temperature {city.name} {start} -> {today}")
            cols.append(fetch_observed(city, start, today))
        _merge_cached(obs_path, pd.concat(cols, axis=1))

    if refresh or not fc_path.exists():
        start = date(2021, 1, 1)
        if fc_path.exists():
            last = pd.read_parquet(fc_path).dropna(how="all").index.max()
            start = (last - pd.Timedelta(days=10)).date()
        end = today + timedelta(days=2)
        frames = []
        for city in CITIES:
            print(f"forecast temperature {city.name} {start} -> {end}")
            primary = fetch_forecasts(city, start, end, config.FORECAST_MODELS[0])
            primary = _fill_gaps(city, primary)
            frames.append(primary.add_prefix(f"{city.name}|"))
        _merge_cached(fc_path, pd.concat(frames, axis=1))


def _fill_gaps(city: config.City, primary: pd.DataFrame) -> pd.DataFrame:
    """Fill holes in the primary model archive with the secondary model, day by day.

    GFS day-ahead forecasts are missing from late December 2023 to mid-January 2024 in
    the archive; JMA GSM is the only archived model covering that window. Only
    backtest-relevant gaps are filled, which keeps the API quota small. The ``filled``
    column (1.0 where an hour holds a gap-filler value) keeps the substitution auditable.
    """
    past = primary.index < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
    in_scope = primary.index >= config.BACKTEST_START.tz_localize("UTC")
    missing = primary[past & in_scope].isna().any(axis=1)
    days = sorted({t.date() for t in missing[missing].index})
    before = primary.isna()
    for lo, hi in _contiguous_ranges(days):
        print(f"  filling {city.name} {lo} -> {hi} from {config.FORECAST_MODELS[1]}")
        backup = fetch_forecasts(city, lo, hi, config.FORECAST_MODELS[1])
        primary = primary.combine_first(backup)
    filled = (before & primary.notna().reindex(before.index)).any(axis=1)
    return primary.assign(filled=filled.astype(float))


def _contiguous_ranges(days: list[date]) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    for day in days:
        if ranges and day - ranges[-1][1] == timedelta(days=1):
            ranges[-1] = (ranges[-1][0], day)
        else:
            ranges.append((day, day))
    return ranges


def weighted_mean(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Population-weighted mean that renormalises weights over cities with data."""
    values = frame[weights.index].to_numpy(dtype=float)
    w = np.broadcast_to(weights.to_numpy(), values.shape)
    present = ~np.isnan(values)
    total = np.where(present, w, 0.0).sum(axis=1)
    acc = np.where(present, values * w, 0.0).sum(axis=1)
    out = np.where(total > 0.5, acc / np.where(total > 0, total, 1.0), np.nan)
    return pd.Series(out, index=frame.index)


def national_temperature() -> pd.DataFrame:
    """National population-weighted temperature: observed, and 24 h / 48 h forecasts."""
    weights = config.city_weights()
    obs = pd.read_parquet(RAW / "weather_observed.parquet")
    fc = pd.read_parquet(RAW / "weather_forecast.parquet")
    out = pd.DataFrame({"temp_obs": weighted_mean(obs, weights)})
    for lead in ("fc_d1", "fc_d2"):
        sub = fc[[f"{c}|{lead}" for c in weights.index]]
        sub.columns = list(weights.index)
        out = out.join(weighted_mean(sub, weights).rename(f"temp_{lead}"), how="outer")
    flags = [f"{c}|filled" for c in weights.index if f"{c}|filled" in fc]
    if flags:
        # 1.0 where any city's forecast comes from the gap-filler model.
        out = out.join(fc[flags].max(axis=1).rename("temp_fc_filled"), how="outer")
    return out


def build_hourly() -> pd.DataFrame:
    """Join demand and weather on a complete hourly UTC grid and cache it."""
    demand = load_demand()
    weather = national_temperature()
    start = config.HISTORY_START.tz_localize(config.PARIS).tz_convert("UTC")
    end = max(demand.index.max(), weather.index.max())
    grid = pd.date_range(start, end, freq="h", tz="UTC", name="time")
    table = demand.reindex(grid).join(weather.reindex(grid))
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.INTERIM / "hourly.parquet")
    return table


def load_hourly() -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM / "hourly.parquet")
