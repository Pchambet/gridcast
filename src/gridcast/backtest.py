"""Rolling-origin backtest: refit every month, forecast every day of that month.

The model for month M is trained on target days up to the first day of M minus two
days (the last complete day at the first issue time of M), with observed weather.
Forecasts for M use the day-ahead information set only. The same models are also fed
two other information sets, to bracket the published backtest:

* ``*_oracle``: observed weather for the target day (what weather-forecast error costs);
* ``*_era5lag``: reanalysis masked over the last days before issue, as in the live job
  (what the backtest's complete reanalysis is worth, see
  :func:`gridcast.features.build_features_era5_delayed`).
"""

from __future__ import annotations

import time

import pandas as pd

from gridcast import config
from gridcast.features import build_features, build_features_era5_delayed
from gridcast.models import DemandModel

PREDICTIONS = config.INTERIM / "predictions.parquet"
CHECKPOINTS = config.INTERIM / "backtest_months"


def run_backtest(
    table: pd.DataFrame,
    start: pd.Timestamp = config.BACKTEST_START,
    end: pd.Timestamp = config.BACKTEST_END,
    resume: bool = False,
) -> pd.DataFrame:
    """Run all monthly refits; ``resume`` reuses months checkpointed by an interrupted run."""
    if not resume:
        for old in CHECKPOINTS.glob("*.parquet"):
            old.unlink()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    history_days = pd.date_range(config.HISTORY_START + pd.Timedelta(days=8), end, freq="D")
    observed = build_features(table, history_days, mode="observed")
    test_days = pd.date_range(start, end, freq="D")
    forecast = build_features(table, test_days, mode="forecast")
    oracle = observed[observed["day"] >= start]
    era5lag = build_features_era5_delayed(table, test_days)

    chunks = []
    for month in pd.period_range(start, end, freq="M"):
        checkpoint = CHECKPOINTS / f"{month}.parquet"
        if checkpoint.exists():
            chunks.append(pd.read_parquet(checkpoint))
            continue
        tic = time.perf_counter()
        first = month.start_time
        train = observed[observed["day"] <= first - pd.Timedelta(days=2)]
        model = DemandModel().fit(train)
        in_month = forecast["day"].dt.to_period("M") == month
        rows = forecast[in_month]
        pred = model.predict(rows)
        for name, frame in (("oracle", oracle), ("era5lag", era5lag)):
            alt = model.predict(frame.loc[rows.index])
            pred[[f"{col}_{name}" for col in alt]] = alt.to_numpy()
        pred["refit"] = first
        pred[["day", "hour", "load", "rte_j1"]] = rows[["day", "hour", "load", "rte_j1"]]
        pred["naive"] = rows["load_d7"]
        pred.to_parquet(checkpoint)
        chunks.append(pred)
        print(f"  {month}: trained on {len(train):,} rows, {time.perf_counter() - tic:4.1f} s")

    out = pd.concat(chunks)
    out.index.name = "time"
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS)
    return out


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS)
