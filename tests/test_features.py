import numpy as np
import pandas as pd
import pytest

from gridcast import features, infoset
from gridcast.features import FEATURES, build_features, build_features_era5_delayed

DAYS = ["2023-10-29", "2023-12-12", "2024-03-31", "2024-04-01"]


@pytest.mark.parametrize("day", DAYS)
def test_forecast_features_ignore_everything_after_issue_time(table, poison, day):
    day = pd.Timestamp(day)
    clean = build_features(table, [day], mode="forecast")[FEATURES]
    dirty = build_features(poison(table, day), [day], mode="forecast")[FEATURES]
    pd.testing.assert_frame_equal(clean, dirty)
    assert clean.notna().all().all()


def test_observed_mode_does_use_observed_temperature(table, poison):
    # Sanity check of the leakage test itself: the oracle mode must react to poison.
    day = pd.Timestamp("2023-12-12")
    clean = build_features(table, [day], mode="observed")["temp"]
    dirty = build_features(poison(table, day), [day], mode="observed")["temp"]
    assert not np.allclose(clean, dirty)


def test_era5_delayed_features_ignore_recent_reanalysis(table):
    day = pd.Timestamp("2023-12-12")
    delayed = build_features_era5_delayed(table, [day])[FEATURES]
    # With no delay the 120-day slice reproduces the full-history features exactly.
    full = build_features(table, [day], mode="forecast")[FEATURES]
    pd.testing.assert_frame_equal(
        build_features_era5_delayed(table, [day], delay=pd.Timedelta(0))[FEATURES], full
    )
    # Reanalysis inside the delay window must not matter...
    recent = table.copy()
    window = (recent.index >= infoset.issue_time(day) - pd.Timedelta(days=5)) & (
        recent.index < infoset.issue_time(day)
    )
    recent.loc[window, "temp_obs"] += 30.0
    pd.testing.assert_frame_equal(build_features_era5_delayed(recent, [day])[FEATURES], delayed)
    # ... while the complete-reanalysis backtest features do use it.
    assert not build_features(recent, [day], mode="forecast")[FEATURES].equals(full)


def test_features_are_deterministic(table):
    days = pd.date_range("2023-11-01", "2023-11-10")
    a = build_features(table, days, mode="forecast")
    b = build_features(table, days, mode="forecast")
    pd.testing.assert_frame_equal(a, b)


def test_weekly_lag_matches_local_wall_clock_across_dst(table):
    # 2023-11-02 08:00 local; one week earlier (2023-10-26) was still in CEST.
    frame = build_features(table, [pd.Timestamp("2023-11-02")], mode="forecast")
    row = frame[frame["hour"] == 8].iloc[0]
    expected = table.loc[pd.Timestamp("2023-10-26 06:00", tz="UTC"), "load"]  # 08:00 CEST
    assert row["load_d7"] == pytest.approx(expected)


def test_spring_gap_is_filled_and_autumn_hour_averaged(table):
    local = features.to_local_wallclock(table["load"])
    assert not np.isnan(local[pd.Timestamp("2024-03-31 02:00")])
    both = table.loc["2023-10-29 00:00":"2023-10-29 01:00", "load"]  # the two 02:00 local
    assert local[pd.Timestamp("2023-10-29 02:00")] == pytest.approx(both.mean())


def test_calendar_flags_holidays_and_bridges():
    days = pd.DatetimeIndex(["2024-05-08", "2024-05-09", "2024-05-10", "2024-05-13"])
    cal = features.day_calendar(days)
    assert cal["is_holiday"].tolist() == [1, 1, 0, 0]  # 8 May, Ascension
    assert cal["is_bridge"].tolist() == [0, 0, 1, 0]  # Friday after Thursday holiday
    assert cal["offday"].tolist() == [1, 1, 1, 0]


def test_ewm_matches_recursive_definition():
    x = np.array([0.0, 10.0, 10.0, 0.0])
    a = 1 - 0.5 ** (1 / 2)
    expected = [0.0]
    for v in x[1:]:
        expected.append(a * v + (1 - a) * expected[-1])
    np.testing.assert_allclose(features.ewm(x, 2), expected)


def test_mos_removes_constant_forecast_bias(table):
    corrected = features.mos_correct(table)
    late = table.index >= table.index[0] + pd.Timedelta(days=70)
    raw_bias = (table["temp_fc_d1"] - table["temp_obs"])[late].mean()
    new_bias = (corrected["temp_fc_d1"] - table["temp_obs"])[late].mean()
    assert raw_bias == pytest.approx(0.8, abs=0.1)
    assert abs(new_bias) < 0.1
