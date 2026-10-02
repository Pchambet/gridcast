import numpy as np
import pandas as pd
import pytest

from gridcast import live
from gridcast.models import qcol


def _forecast(day: str, point: float = 50_000.0) -> pd.DataFrame:
    times = pd.date_range(
        pd.Timestamp(day).tz_localize("Europe/Paris"), periods=24, freq="h"
    ).tz_convert("UTC")
    frame = pd.DataFrame({"time": times, "day": pd.Timestamp(day), "hour": range(24)})
    frame["issued_at"] = times[0] - pd.Timedelta(hours=12)
    frame["kind"] = "live"
    frame["point"] = point
    frame["point_raw"] = point
    for q, off in zip((0.025, 0.1, 0.9, 0.975), (-3000, -1500, 1500, 3000), strict=True):
        frame[qcol(q)] = point + off
    frame[["lo80", "hi80", "lo95", "hi95"]] = [
        point - 2000,
        point + 2000,
        point - 4000,
        point + 4000,
    ]
    frame["rte_j1"] = np.nan
    frame["load"] = np.nan
    return frame


def test_append_is_idempotent_per_target_day():
    log = live.append_forecast(pd.DataFrame(columns=live.LOG_COLUMNS), _forecast("2026-10-03"))
    log = live.append_forecast(log, _forecast("2026-10-04"))
    rerun = live.append_forecast(log, _forecast("2026-10-04", point=60_000.0))
    assert len(rerun) == 48
    assert rerun.loc[rerun["day"] == "2026-10-04", "point"].eq(60_000.0).all()
    assert rerun["time"].is_monotonic_increasing


def test_scoring_fills_only_published_hours_and_never_overwrites():
    log = _forecast("2026-10-03")
    log.loc[0, "load"] = 1.0  # already scored earlier: must be kept
    table = pd.DataFrame(
        {"load": 51_000.0, "rte_j1": 50_500.0}, index=pd.DatetimeIndex(log["time"][:10])
    )
    scored = live.score_log(log, table)
    assert scored["load"].iloc[0] == 1.0
    assert scored["load"].iloc[1:10].eq(51_000.0).all()
    assert scored["load"].iloc[10:].isna().all()


def test_track_record_counts_only_scored_hours():
    log = _forecast("2026-10-03")
    log["load"] = 50_000.0 + np.where(np.arange(24) < 6, 3_000.0, 500.0)  # 6 misses at 80 %
    log["rte_j1"] = 50_000.0
    rec = live.track_record(pd.concat([log, _forecast("2026-10-04")]))
    assert rec["days"] == 1
    assert rec["coverage80"] == pytest.approx(18 / 24)
    assert rec["coverage95"] == pytest.approx(1.0)
    assert rec["mape"] == pytest.approx(rec["mape_rte"])


def test_log_roundtrip(tmp_path):
    log = live.append_forecast(pd.DataFrame(columns=live.LOG_COLUMNS), _forecast("2026-10-03"))
    path = tmp_path / "log.csv"
    live.write_log(log, path)
    back = live.read_log(path)
    pd.testing.assert_series_equal(back["time"], log["time"], check_dtype=False)
    assert back["point"].eq(50_000.0).all()


def test_calibrated_intervals_widen_when_history_is_noisier():
    rng = np.random.default_rng(0)
    days = [str(d.date()) for d in pd.date_range("2026-06-01", "2026-09-30")]
    history = pd.concat([_forecast(d) for d in days], ignore_index=True)
    today = _forecast("2026-10-02")[["time", "day", "load", "point", *live.QCOLS]]
    cols = ["time", "day", "load", "point", *live.QCOLS]

    def bands(noise: float) -> pd.DataFrame:
        h = history.copy()
        h["load"] = h["point"] + rng.normal(0, noise, len(h))
        return live.calibrated_intervals(h[cols], today)

    calm, wild = bands(500.0), bands(4000.0)
    assert (calm["hi80"] - calm["lo80"]).mean() < (wild["hi80"] - wild["lo80"]).mean()
    assert ((wild["hi95"] - wild["lo95"]) > (wild["hi80"] - wild["lo80"])).all()


def test_track_record_skips_hours_without_bands_and_hindcasts():
    scored = _forecast("2026-10-03")
    scored["load"], scored["rte_j1"] = 50_000.0, 50_000.0  # every hour inside the bands
    scored.loc[:5, ["lo80", "hi80", "lo95", "hi95"]] = np.nan
    hind = _forecast("2026-10-02")
    hind["kind"] = "hindcast"
    hind["load"], hind["rte_j1"] = 99_000.0, 50_000.0  # would be misses if counted
    rec = live.track_record(pd.concat([hind, scored], ignore_index=True))
    assert rec["days"] == 1
    assert rec["coverage80"] == 1.0 and rec["coverage95"] == 1.0
    assert rec["hours_without_bands"] == 6


def _seed(table: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """Backtest-like rows: a decent point forecast with quantiles around it."""
    rng = np.random.default_rng(5)
    rows = table.loc[
        pd.Timestamp(first).tz_localize("Europe/Paris") : pd.Timestamp(last).tz_localize(
            "Europe/Paris"
        )
        + pd.Timedelta(hours=23)
    ]
    seed = pd.DataFrame(index=rows.index)
    seed["day"] = rows.index.tz_convert("Europe/Paris").tz_localize(None).normalize()
    seed["load"] = rows["load"]
    seed["point"] = rows["load"] + rng.normal(0, 800, len(rows))
    seed["point_raw"] = seed["point"]
    for q, z in zip((0.025, 0.1, 0.9, 0.975), (-1.96, -1.28, 1.28, 1.96), strict=True):
        seed[qcol(q)] = seed["point"] + z * 700
    return seed


def test_stale_seed_fails_loudly_instead_of_publishing_nan_bands(table, fast_model):
    # The seed ends 70 days before the target day: fewer than 30 calibration days left.
    seed = _seed(table, "2023-10-01", "2023-12-31")
    empty = pd.DataFrame(columns=live.LOG_COLUMNS)
    with pytest.raises(RuntimeError, match="no calibrated interval"):
        live.forecast_day(table, pd.Timestamp("2024-03-10"), empty, seed)


def test_hindcast_closes_the_gap_so_a_late_first_run_gets_finite_bands(table, fast_model):
    seed = _seed(table, "2023-10-01", "2023-12-31")
    day = pd.Timestamp("2024-03-10")
    log = pd.DataFrame(columns=live.LOG_COLUMNS)
    missed = live.hindcast(table, day, log, seed)
    gap = pd.date_range("2024-01-01", "2024-03-09")
    assert set(missed["day"]) == set(gap) and (missed["kind"] == "hindcast").all()
    log = live.append_forecast(log, missed)
    new = live.forecast_day(table, day, log, seed)
    assert np.isfinite(new[live.BANDS].to_numpy(dtype=float)).all()
    assert (new["lo95"] < new["lo80"]).all() and (new["hi80"] < new["hi95"]).all()
    assert (new["kind"] == "live").all()
    # Once filled, nothing is missing any more, and hindcasts never count as live.
    assert live.missing_days(seed, log, day).empty
    assert live.track_record(log)["days"] == 0


def test_live_forecast_ignores_everything_after_issue_time(table, poison, fast_model):
    seed = _seed(table, "2023-12-01", "2024-02-27")
    day = pd.Timestamp("2024-03-01")
    log = pd.DataFrame(columns=live.LOG_COLUMNS)
    clean = live.forecast_day(table, day, log, seed)
    dirty = live.forecast_day(poison(table, day), day, log, seed)
    cols = ["point_raw", "point", *live.QCOLS, *live.BANDS]
    pd.testing.assert_frame_equal(clean[cols], dirty[cols])
