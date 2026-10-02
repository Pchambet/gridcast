"""The daily live job: forecast tomorrow, keep a public log, score it honestly.

The job is stateless apart from the forecast log. Each run

1. retrains the model on every complete past day (observed weather),
2. forecasts tomorrow under the day-ahead information set,
3. rebuilds the conformal intervals by replaying CQR + ACI over the log, seeded with
   the end of the backtest so that calibration exists on day one, and
4. back-fills realised demand and RTE's forecast into past log rows.

Because ACI is a deterministic function of the scored history, replaying it each day
gives exactly the intervals a stateful service would have produced, with no state
file to corrupt.
"""

from __future__ import annotations

import pandas as pd

from gridcast import config, infoset
from gridcast.conformal import METHOD_BY_NAME, online_intervals
from gridcast.features import build_features
from gridcast.models import DemandModel, qcol

LOG_NAME = "forecast_log.csv"
QCOLS = [qcol(q) for q in config.QUANTILES]
LOG_COLUMNS = [
    "time", "day", "hour", "issued_at", "point", *QCOLS,
    "lo80", "hi80", "lo95", "hi95", "rte_j1", "load",
]  # fmt: skip


def log_path():
    return config.LIVE / LOG_NAME


def read_log(path=None) -> pd.DataFrame:
    path = path or log_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(path, parse_dates=["time", "issued_at"])
    log["time"] = pd.to_datetime(log["time"], utc=True)
    log["issued_at"] = pd.to_datetime(log["issued_at"], utc=True)
    log["day"] = pd.to_datetime(log["day"])
    return log


def write_log(log: pd.DataFrame, path=None) -> None:
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = log[LOG_COLUMNS].sort_values("time")
    out.to_csv(path, index=False, float_format="%.1f", date_format="%Y-%m-%dT%H:%M:%SZ")


def append_forecast(log: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Add a day's forecast; a re-run for the same target day replaces the earlier one."""
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
    """Live scores over hours whose outcome is known."""
    done = log.dropna(subset=["load", "rte_j1"])
    if done.empty:
        return {"days": 0}
    y = done["load"]
    return {
        "days": int(done["day"].nunique()),
        "first_day": str(done["day"].min().date()),
        "last_day": str(done["day"].max().date()),
        "coverage80": float(((y >= done["lo80"]) & (y <= done["hi80"])).mean()),
        "coverage95": float(((y >= done["lo95"]) & (y <= done["hi95"])).mean()),
        "mape": float(((done["point"] - y).abs() / y).mean()),
        "mape_rte": float(((done["rte_j1"] - y).abs() / y).mean()),
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


def forecast_day(table: pd.DataFrame, day: pd.Timestamp, log: pd.DataFrame, seed: pd.DataFrame):
    """Train, forecast and calibrate one target day. Returns the new log rows."""
    history_days = pd.date_range(
        config.HISTORY_START + pd.Timedelta(days=15), day - pd.Timedelta(days=2), freq="D"
    )
    train = build_features(table, history_days, mode="observed")
    model = DemandModel().fit(train)
    rows = build_features(table, [day], mode="forecast")
    if rows[["load_d2", "load_d7", "temp"]].isna().any().any():
        raise RuntimeError(f"incomplete inputs for {day.date()}: demand or weather missing")
    pred = model.predict(rows)
    pred["day"], pred["hour"] = rows["day"], rows["hour"]
    pred["rte_j1"] = rows["rte_j1"]
    pred["load"] = float("nan")
    pred["issued_at"] = infoset.issue_time(day)
    pred = pred.reset_index()

    history = pd.concat(
        [frame for frame in (seed.reset_index(), log) if not frame.empty], ignore_index=True
    )
    history = history[history["day"] < day][["time", "day", "load", "point", *QCOLS]]
    bands = calibrated_intervals(history, pred[["time", "day", "load", "point", *QCOLS]])
    for col in bands:
        pred[col] = bands[col].to_numpy()
    return pred[LOG_COLUMNS]


def run(table: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Forecast tomorrow (Europe/Paris), update the log, return the track record."""
    now = now or pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp(now.tz_convert(config.PARIS).date()) + pd.Timedelta(days=1)
    seed = pd.read_parquet(config.RESULTS / "calibration_seed.parquet")
    log = score_log(read_log(), table)
    new = forecast_day(table, day, log, seed)
    log = append_forecast(log, new)
    write_log(log)
    record = track_record(log)
    print(f"forecast for {day.date()}: peak {new['point'].max() / 1000:.1f} GW; record {record}")
    return record
