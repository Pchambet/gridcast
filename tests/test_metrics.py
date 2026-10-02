import numpy as np
import pandas as pd
import pytest

from gridcast import metrics


def test_point_metrics_hand_computed():
    y = np.array([100.0, 200.0])
    yhat = np.array([110.0, 180.0])
    assert metrics.mape(y, yhat) == pytest.approx(0.10)
    assert metrics.mae(y, yhat) == pytest.approx(15.0)
    assert metrics.rmse(y, yhat) == pytest.approx(np.sqrt((100 + 400) / 2))


def test_interval_score_hand_computed():
    y = np.array([5.0, 0.0, 12.0])
    lo, hi = np.full(3, 2.0), np.full(3, 10.0)
    # width 8; misses of 2 below and 2 above, penalised by 2 / 0.2 = 10.
    np.testing.assert_allclose(metrics.interval_score(y, lo, hi, 0.2), [8.0, 28.0, 28.0])
    assert metrics.coverage(y, lo, hi) == pytest.approx(1 / 3)
    assert metrics.width(lo, hi) == 8.0


def test_pinball_hand_computed():
    y = np.array([10.0, 10.0])
    q = np.array([8.0, 12.0])
    np.testing.assert_allclose(metrics.pinball(y, q, 0.9), [1.8, 0.2])


def test_block_bootstrap_ci_contains_truth_and_respects_blocks():
    rng = np.random.default_rng(0)
    times = pd.date_range("2023-01-01 23:00", periods=24 * 7 * 60, freq="h", tz="UTC")  # Mon 00:00 CET
    blocks = metrics.week_blocks(times)
    assert blocks.max() == 59
    # Strongly autocorrelated noise: one shared shock per week.
    x = 1.0 + rng.normal(0, 1, 60)[blocks] + rng.normal(0, 0.1, len(times))
    est, lo, hi = metrics.block_bootstrap(lambda i: x[i].mean(), blocks, reps=400)
    assert lo < 1.0 < hi
    # The block CI must be much wider than a naive i.i.d. one would be.
    naive_half_width = 1.96 * x.std() / np.sqrt(len(x))
    assert (hi - lo) / 2 > 5 * naive_half_width
    assert lo <= est <= hi
