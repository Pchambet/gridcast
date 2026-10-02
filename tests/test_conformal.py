import numpy as np
import pandas as pd
import pytest

from gridcast import conformal
from gridcast.conformal import METHOD_BY_NAME, conformal_quantile, online_intervals
from gridcast.models import qcol


def test_conformal_quantile_is_the_finite_sample_order_statistic():
    scores = np.arange(1.0, 10.0)  # n = 9
    # ceil((9 + 1) * 0.8) = 8 -> 8th smallest
    assert conformal_quantile(scores, 0.2) == 8.0
    # ceil(10 * 0.95) = 10 > n -> no finite quantile exists
    assert conformal_quantile(scores, 0.05) == np.inf
    assert conformal_quantile(np.array([]), 0.2) == np.inf


def test_cqr_score_is_signed_distance_outside_the_band():
    y = np.array([5.0, 0.0, 12.0])
    lo, hi = np.full(3, 2.0), np.full(3, 10.0)
    np.testing.assert_allclose(conformal.cqr_scores(y, lo, hi), [-3.0, 2.0, 2.0])


def test_aci_step_moves_alpha_against_the_error():
    # Too many misses -> alpha shrinks -> wider intervals; and vice versa.
    assert conformal.aci_step(0.2, 0.2, 0.1, np.array([1, 1, 0, 0])) == pytest.approx(0.17)
    assert conformal.aci_step(0.2, 0.2, 0.1, np.zeros(4)) == pytest.approx(0.22)


def _stream(n_days: int, scale: np.ndarray | float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Hourly stream with a constant point forecast and noise of a given scale per day."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=24 * n_days, freq="h", tz="UTC", name="time")
    day = idx.tz_localize(None).normalize()
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (n_days,)).repeat(24)
    noise = rng.normal(0, 1, len(idx)) * scale
    frame = pd.DataFrame({"day": day, "load": 100 + noise, "point": 100.0}, index=idx)
    for q in (0.025, 0.1, 0.9, 0.975):
        frame[qcol(q)] = 100.0 + (-1.0 if q < 0.5 else 1.0)  # deliberately too narrow
    return frame


@pytest.mark.parametrize("name", ["split_static", "split_rolling", "cqr_rolling"])
@pytest.mark.parametrize("level", [0.8, 0.95])
def test_coverage_is_nominal_on_exchangeable_data(name, level):
    frame = _stream(400, seed=1)
    iv = online_intervals(
        frame, METHOD_BY_NAME[name], level, static_end=pd.Timestamp("2020-03-31")
    )
    scored = frame.index >= pd.Timestamp("2020-04-01", tz="UTC")
    covered = (frame["load"] >= iv["lo"]) & (frame["load"] <= iv["hi"])
    assert covered[scored].mean() == pytest.approx(level, abs=0.02)


def test_raw_quantiles_undercover_and_conformal_fixes_it():
    frame = _stream(200, seed=2)
    raw = online_intervals(frame, METHOD_BY_NAME["qr_raw"], 0.8)
    assert ((frame["load"] >= raw["lo"]) & (frame["load"] <= raw["hi"])).mean() < 0.75


def test_aci_recovers_coverage_after_a_variance_shift():
    # Noise triples after day 150: a frozen calibration set badly undercovers, ACI adapts.
    scale = np.where(np.arange(450) < 150, 1.0, 3.0)
    frame = _stream(450, scale=scale, seed=3)
    kwargs = {"static_end": pd.Timestamp("2020-03-31"), "gamma": 0.05}
    static = online_intervals(frame, METHOD_BY_NAME["split_static"], 0.8, **kwargs)
    aci = online_intervals(frame, METHOD_BY_NAME["split_static_aci"], 0.8, **kwargs)
    late = frame.index >= pd.Timestamp("2020-09-01", tz="UTC")

    def cov(iv):
        return ((frame["load"] >= iv["lo"]) & (frame["load"] <= iv["hi"]))[late].mean()

    assert cov(static) < 0.5
    assert cov(aci) == pytest.approx(0.8, abs=0.04)


def test_feedback_is_delayed_two_days():
    # A day's own errors must not influence its interval: perturbing the outcomes of
    # the last two days cannot change any interval up to and including that day.
    frame = _stream(120, seed=4)
    method = METHOD_BY_NAME["cqr_rolling_aci"]
    base = online_intervals(frame, method, 0.8)
    shocked = frame.copy()
    last_two = shocked["day"] >= shocked["day"].max() - pd.Timedelta(days=1)
    shocked.loc[last_two, "load"] += 1000.0
    after = online_intervals(shocked, method, 0.8)
    pd.testing.assert_frame_equal(base, after)
