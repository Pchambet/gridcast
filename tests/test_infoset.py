import pandas as pd
import pytest

from gridcast import infoset


@pytest.mark.parametrize(
    ("day", "expected_utc"),
    [
        ("2024-01-15", "2024-01-14 11:00"),  # winter: CET = UTC+1
        ("2024-07-15", "2024-07-14 10:00"),  # summer: CEST = UTC+2
        ("2024-03-31", "2024-03-30 11:00"),  # issued the day before the spring change
        ("2024-04-01", "2024-03-31 10:00"),  # issued on the change day itself, in CEST
    ],
)
def test_issue_time_is_noon_paris_the_day_before(day, expected_utc):
    assert infoset.issue_time(pd.Timestamp(day)) == pd.Timestamp(expected_utc, tz="UTC")


@pytest.mark.parametrize(
    ("day", "n_hours"), [("2024-03-31", 23), ("2024-10-27", 25), ("2024-06-12", 24)]
)
def test_target_hours_follow_dst(day, n_hours):
    hours = infoset.target_hours(pd.Timestamp(day))
    assert len(hours) == n_hours
    local = hours.tz_convert("Europe/Paris")
    assert local[0].hour == 0
    assert (local.normalize().tz_localize(None) == pd.Timestamp(day)).all()


def test_target_hours_many_is_union_of_days():
    days = pd.date_range("2024-03-30", "2024-04-01")
    many = infoset.target_hours_many(days)
    assert len(many) == 24 + 23 + 24
    assert many.is_monotonic_increasing and many.is_unique


def test_demand_cutoff_leaves_last_known_hour_ten_local():
    day = pd.Timestamp("2024-01-15")
    cutoff = infoset.demand_cutoff(day)
    assert cutoff == pd.Timestamp("2024-01-14 10:00", tz="UTC")  # 11:00 local
    assert infoset.LAST_KNOWN_LOCAL_HOUR == 10


def test_day1_weather_forecast_only_when_its_run_existed_at_issue_time():
    day = pd.Timestamp("2024-01-15")
    hours = infoset.target_hours(day)
    use_d2 = infoset.use_day2_forecast(hours, day)
    issue = infoset.issue_time(day)
    # Every d1 value used was forecast at least 4 h before the issue time.
    d1_issued = hours[~use_d2] - pd.Timedelta(hours=24)
    assert (d1_issued <= issue - pd.Timedelta(hours=4)).all()
    # ... and the switch is a single cut: early hours d1, later hours d2.
    assert use_d2.sum() > 0 and (~use_d2).sum() > 0
    assert (pd.Series(use_d2).diff().fillna(0) >= 0).all()
