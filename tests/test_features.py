import numpy as np
import pandas as pd
import pytest

from gridcast import features, infoset
from gridcast.features import FEATURES, build_features

DAYS = ["2023-10-29", "2023-12-12", "2024-03-31", "2024-04-01"]


def _poison_future(table: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Replace everything unknowable at the issue time of ``day`` by garbage."""
    poisoned = table.copy()
    rng = np.random.default_rng(1)
    after_cutoff = poisoned.index >= infoset.demand_cutoff(day)
    after_issue = poisoned.index >= infoset.issue_time(day)
    poisoned.loc[after_cutoff, "load"] = rng.normal(1e6, 1e5, after_cutoff.sum())
    poisoned.loc[after_issue, "temp_obs"] = rng.normal(80, 10, after_issue.sum())
    # Forecasts are legitimate inputs only if issued before the issue time; corrupt the
    # 24 h-ahead values that were not yet available (valid more than 20 h after issue).
    late = poisoned.index > infoset.issue_time(day) + pd.Timedelta(hours=20)
    poisoned.loc[late, "temp_fc_d1"] = rng.normal(80, 10, late.sum())
    return poisoned


@pytest.mark.parametrize("day", DAYS)
def test_forecast_features_ignore_everything_after_issue_time(table, day):
    day = pd.Timestamp(day)
    clean = build_features(table, [day], mode="forecast")[FEATURES]
    dirty = build_features(_poison_future(table, day), [day], mode="forecast")[FEATURES]
    pd.testing.assert_frame_equal(clean, dirty)
    assert clean.notna().all().all()


def test_observed_mode_does_use_observed_temperature(table):
    # Sanity check of the leakage test itself: the oracle mode must react to poison.
    day = pd.Timestamp("2023-12-12")
    clean = build_features(table, [day], mode="observed")["temp"]
    dirty = build_features(_poison_future(table, day), [day], mode="observed")["temp"]
    assert not np.allclose(clean, dirty)


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
