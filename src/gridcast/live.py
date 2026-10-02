"""The daily live job: forecast tomorrow, keep a public log, score it honestly.

The job is stateless apart from the forecast log. Each run

1. retrains the model on every complete past day (observed weather),
2. hindcasts any day since the end of the backtest that has no forecast yet (first run,
   or after an outage) with the backtest protocol, so calibration never runs dry,
3. forecasts tomorrow under the day-ahead information set,
4. rebuilds the conformal intervals by replaying CQR + ACI over seed + log, and
5. back-fills realised demand and RTE's forecast into past log rows.

Because ACI is a deterministic function of the scored history, replaying it each day
gives exactly the intervals a stateful service would have produced, with no state
file to corrupt. Hindcast rows stay in the log (``kind == "hindcast"``) so the replay
is reproducible, but they never enter the live track record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import config, infoset, metrics
from gridcast.conformal import METHOD_BY_NAME, online_intervals
from gridcast.features import build_features
from gridcast.models import DemandModel, bias_adjusted, qcol

LOG_NAME = "forecast_log.csv"
QCOLS = [qcol(q) for q in config.QUANTILES]
BANDS = ["lo80", "hi80", "lo95", "hi95"]
LOG_COLUMNS = [
    "time", "day", "hour", "issued_at", "kind", "point_raw", "point", *QCOLS,
    *BANDS, "rte_j1", "load",
]  # fmt: skip
# Calibration looks 90 days back, so hindcasting further than this is wasted compute.
HINDCAST_MAX_DAYS = 2 * config.ROLLING_WINDOW_DAYS


def log_path() -> Path:
    return config.LIVE / LOG_NAME


def read_log(path: Path | None = None) -> pd.DataFrame:
    path = path or log_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(path)
    log["time"] = pd.to_datetime(log["time"], utc=True)
    log["issued_at"] = pd.to_datetime(log["issued_at"], utc=True)
    log["day"] = pd.to_datetime(log["day"])
    return log


def write_log(log: pd.DataFrame, path: Path | None = None) -> None:
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = log[LOG_COLUMNS].sort_values("time").copy()
    out["day"] = pd.to_datetime(out["day"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, float_format="%.1f", date_format="%Y-%m-%dT%H:%M:%SZ")


def append_forecast(log: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Add forecast days; a re-run for the same target day replaces the earlier one."""
    days = set(forecast["day"])
    kept = log[~log["day"].isin(days)]
    parts = [frame for frame in (kept, forecast[LOG_COLUMNS]) if not frame.empty]
    return pd.concat(parts, ignore_index=True).sort_values("time", ignore_index=True)


def score_log(log: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Fill realised demand (and RTE's J-1 forecast) for every hour now published."""
    out = log.copy()
    idx = pd.DatetimeIndex(out["time"])
    realised = table["load"].reindex(idx).to_numpy()
    rte = table["rte_j1"].reindex(idx).to_numpy()
    out["load"] = out["load"].astype(float).fillna(pd.Series(realised, index=out.index))
    out["rte_j1"] = out["rte_j1"].astype(float).fillna(pd.Series(rte, index=out.index))
    return out


def track_record(log: pd.DataFrame) -> dict:
    """Live scores over hours whose outcome is known.

    Only forecasts actually issued live count. Hours without a finite interval are
    reported, not scored: counting them as misses would bias coverage down.
    """
    done = log[log["kind"] == "live"].dropna(subset=["load", "rte_j1"])
    if done.empty:
        return {"days": 0}
    banded = done[np.isfinite(done[BANDS].to_numpy(dtype=float)).all(axis=1)]
    y = banded["load"].to_numpy()

    def cov(tag: int) -> float:
        if banded.empty:
            return float("nan")
        return metrics.coverage(y, banded[f"lo{tag}"].to_numpy(), banded[f"hi{tag}"].to_numpy())

    return {
        "days": int(done["day"].nunique()),
        "first_day": str(done["day"].min().date()),
        "last_day": str(done["day"].max().date()),
        "coverage80": cov(80),
        "coverage95": cov(95),
        "hours_without_bands": int(len(done) - len(banded)),
        "mape": metrics.mape(done["load"].to_numpy(), done["point"].to_numpy()),
        "mape_rte": metrics.mape(done["load"].to_numpy(), done["rte_j1"].to_numpy()),
    }


def calibrated_intervals(history: pd.DataFrame, today: pd.DataFrame) -> pd.DataFrame:
    """Replay CQR + ACI over (seed + log) and return 80/95 bounds for ``today``."""
    frame = pd.concat([history, today]).set_index("time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    method = METHOD_BY_NAME["cqr_rolling_aci"]
    out = pd.DataFrame(index=pd.DatetimeIndex(today["time"]))
    for level in config.LEVELS:
        iv = online_intervals(frame, method, level)
        tag = round(level * 100)
        out[f"lo{tag}"] = iv["lo"].reindex(out.index).to_numpy()
        out[f"hi{tag}"] = iv["hi"].reindex(out.index).to_numpy()
    return out


def training_rows(table: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Observed-weather feature rows for every day complete at the issue time of ``day``."""
    history_days = pd.date_range(
        config.HISTORY_START + pd.Timedelta(days=15), day - pd.Timedelta(days=2), freq="D"
    )
    return build_features(table, history_days, mode="observed")


def _past_rows(seed: pd.DataFrame, log: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    frames = [frame for frame in (seed.reset_index(), log) if not frame.empty]
    history = pd.concat(frames, ignore_index=True)
    return history[history["day"] < before]


def _finish(raw: pd.DataFrame, history: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Level-correct raw model output and attach conformal bands, as in the backtest.

    ``raw`` holds model output (``point`` and quantiles) for one or more target days,
    plus ``time``, ``day``, ``hour``, ``load`` and ``rte_j1``; ``history`` holds every
    earlier seed or log row. Both steps only use errors at least two days old, so a
    multi-day ``raw`` is processed exactly as if it had arrived one day at a time.
    """
    pred = raw.copy()
    pred["issued_at"] = [infoset.issue_time(d) for d in pred["day"]]
    pred["kind"] = kind
    hist, new = history.set_index("time"), pred.set_index("time")
    series = pd.concat([hist["point_raw"], new["point"]]).sort_index()
    actual = pd.concat([hist["load"], new["load"]]).sort_index()
    offset = (series - bias_adjusted(series, actual)).reindex(pred["time"]).fillna(0.0)
    pred["point_raw"] = pred["point"]
    for col in ("point", *QCOLS):
        pred[col] = pred[col] - offset.to_numpy()
    cols = ["time", "day", "load", "point", *QCOLS]
    bands = calibrated_intervals(history[cols], pred[cols])
    for col in bands:
        pred[col] = bands[col].to_numpy()
    return pred[LOG_COLUMNS]


def missing_days(seed: pd.DataFrame, log: pd.DataFrame, day: pd.Timestamp) -> pd.DatetimeIndex:
    """Days after the seed and before ``day`` (at most 180 back) that the log lacks."""
    first = max(
        seed["day"].max() + pd.Timedelta(days=1), day - pd.Timedelta(days=HINDCAST_MAX_DAYS)
    )
    candidates = pd.date_range(first, day - pd.Timedelta(days=1), freq="D")
    return candidates[~candidates.isin(pd.DatetimeIndex(log["day"]))]


def hindcast(
    table: pd.DataFrame,
    day: pd.Timestamp,
    log: pd.DataFrame,
    seed: pd.DataFrame,
    train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast the days before ``day`` that the log lacks, with the backtest protocol.

    One refit per month, trained on days up to the first of the month minus two, and
    forecast-mode features: what the backtest would have produced for those days.
    Returns log rows with ``kind == "hindcast"`` (none if nothing is missing).
    """
    gap = missing_days(seed, log, day)
    if gap.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)
    train = training_rows(table, day) if train is None else train
    rows = build_features(table, gap, mode="forecast")
    parts = []
    for month in gap.to_period("M").unique():
        model = DemandModel().fit(train[train["day"] <= month.start_time - pd.Timedelta(days=2)])
        parts.append(model.predict(rows[rows["day"].dt.to_period("M") == month]))
    raw = pd.concat(parts)
    raw[["day", "hour", "load", "rte_j1"]] = rows[["day", "hour", "load", "rte_j1"]]
    out = _finish(raw.reset_index(), _past_rows(seed, log, day), kind="hindcast")
    print(f"hindcast {len(gap)} days without a forecast: {gap.min().date()} to {gap.max().date()}")
    return out


def forecast_day(
    table: pd.DataFrame,
    day: pd.Timestamp,
    log: pd.DataFrame,
    seed: pd.DataFrame,
    train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Train, forecast and calibrate one target day. Returns the new log rows.

    Raises rather than publish a forecast without intervals, which happens when seed
    and log hold fewer than 30 scored days in the calibration window.
    """
    train = training_rows(table, day) if train is None else train
    model = DemandModel().fit(train)
    rows = build_features(table, [day], mode="forecast")
    if rows[["load_d2", "load_d7", "temp"]].isna().any().any():
        raise RuntimeError(f"incomplete inputs for {day.date()}: demand or weather missing")
    raw = model.predict(rows)
    raw["day"], raw["hour"], raw["rte_j1"] = rows["day"], rows["hour"], rows["rte_j1"]
    raw["load"] = float("nan")
    pred = _finish(raw.reset_index(), _past_rows(seed, log, day), kind="live")
    if pred[BANDS].isna().any().any():
        raise RuntimeError(
            f"no calibrated interval for {day.date()}: too few scored days in the "
            "calibration window; hindcast the missing days first"
        )
    return pred


def run(table: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Forecast tomorrow (Europe/Paris), update the log, return the track record."""
    now = now or pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp(now.tz_convert(config.PARIS).date()) + pd.Timedelta(days=1)
    seed = pd.read_parquet(config.RESULTS / "calibration_seed.parquet")
    log = score_log(read_log(), table)
    train = training_rows(table, day)
    missed = hindcast(table, day, log, seed, train)
    if not missed.empty:
        log = append_forecast(log, missed)
    new = forecast_day(table, day, log, seed, train)
    log = append_forecast(log, new)
    write_log(log)
    record = track_record(log)
    print(f"forecast for {day.date()}: peak {new['point'].max() / 1000:.1f} GW; record {record}")
    return record
